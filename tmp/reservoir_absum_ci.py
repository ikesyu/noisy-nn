"""absum was the ONE task where the bump won big (+60.8%, 2-3x). Confirm at 6 seeds.
absum = |u(t-2)-c| + |u(t-5)-c| : sum of EVEN, CENTRED, non-monotone functions
of individual lags -- 'locality/centring', NOT products (product/square refuted)."""
import numpy as np
from reservoir.fields import LDNField
from reservoir_lambda_local import masks
from reservoir_gamma_schedule import train_eval as ge

G = (0.3, 0.5, 0.7, 1.0, 1.5); T = 3000; S = 6
rng = np.random.default_rng(0); u = rng.uniform(0.0, 0.5, size=T)
lag = lambda L: np.concatenate([np.zeros(L), u[:T-L]])
y = np.abs(lag(2)-0.25) + np.abs(lag(5)-0.25); y = y - y.mean()
tr, te = masks(T); A = LDNField(H=48, theta=60.0).run(u)
print("absum = |u(t-2)-0.25| + |u(t-5)-0.25|, LDN, depth 1, 6 seeds")
res = {}
for act in ("threshold", "crossing", "tanh"):
    best = None
    for g in G:
        e = [ge(A, y, tr, te, 1, act, [g], seed=s) for s in range(S)]
        if best is None or np.mean(e) < best[0]: best = (np.mean(e), np.std(e), g, e)
    res[act] = best
    print(f"  {act:<10} {best[0]:.4f} +/- {best[1]:.4f}  (g={best[2]})")
c = np.array(res["crossing"][3]); t = np.array(res["tanh"][3])
d = t - c; se = d.std(ddof=1)/np.sqrt(S)
print(f"\n  tanh - crossing: {d.mean():+.4f} +/- {d.std():.4f}; "
      f"crossing better in {int((d>0).sum())}/{S};  mean/se = {d.mean()/se:+.2f}")
print(f"  -> {'REAL' if abs(d.mean()/se) > 2.5 else 'not resolvable'}")
# depth 2 as well: does it survive depth?
for depth in (2,):
    row = {}
    for act in ("threshold", "crossing", "tanh"):
        row[act] = min(np.mean([ge(A, y, tr, te, depth, act, [g]*depth, seed=s)
                                for s in range(3)]) for g in G)
    print(f"  depth{depth} (3 seeds): " + "  ".join(f"{k}={v:.3f}" for k, v in row.items()))
