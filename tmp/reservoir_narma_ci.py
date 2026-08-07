"""Is the NARMA depth-1 bump advantage (sec10.34: 0.34 vs 0.37) real or seed noise?
6 seeds with std, each activation at its own best gamma."""
import numpy as np
from reservoir.tasks import narma_x
from reservoir.fields import LDNField
from reservoir_lambda_local import masks
from reservoir_gamma_schedule import train_eval as ge

G = (0.3, 0.5, 0.7, 1.0); T = 3000; S = 6
u, y = narma_x(T, 10, seed=0); y = y - y.mean()
tr, te = masks(T); A = LDNField(H=48, theta=60.0).run(u)
print("NARMA-10 + LDN, depth 1, 6 seeds, per-activation best gamma")
res = {}
for act in ("threshold", "crossing", "tanh"):
    best = None
    for g in G:
        e = [ge(A, y, tr, te, 1, act, [g], seed=s) for s in range(S)]
        if best is None or np.mean(e) < best[0]: best = (np.mean(e), np.std(e), g, e)
    res[act] = best
    print(f"  {act:<10} {best[0]:.4f} +/- {best[1]:.4f}  (g={best[2]}, n={S})")
c = res["crossing"]; t = res["tanh"]; th = res["threshold"]
d = np.array(t[3]) - np.array(c[3])
print(f"\n  tanh - crossing per-seed diff: mean {d.mean():+.4f} +/- {d.std():.4f}")
print(f"  crossing better in {int((d>0).sum())}/{S} seeds")
se = d.std(ddof=1)/np.sqrt(S)
print(f"  paired t-ish: mean/se = {d.mean()/se:+.2f}  -> "
      f"{'real' if abs(d.mean()/se) > 2.5 else 'NOT resolvable at this n'}")
