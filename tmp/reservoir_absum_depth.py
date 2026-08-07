"""Is the bump's absum win a real activation advantage, or the SAME depth-1
optimisation failure of monotone nets seen in sec10.32 (6d depth1 tanh 0.97 vs
depth2 tanh 0.016 on a local target)?  If deeper monotone nets close the gap,
the claim is only 'the bump wins where depth-1 monotone nets fail to optimise'."""
import numpy as np
from reservoir.fields import LDNField
from reservoir_lambda_local import masks
from reservoir_gamma_schedule import train_eval as ge

G = (0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0); T = 3000; S = 3
rng = np.random.default_rng(0); u = rng.uniform(0.0, 0.5, size=T)
lag = lambda L: np.concatenate([np.zeros(L), u[:T-L]])
y = np.abs(lag(2)-0.25) + np.abs(lag(5)-0.25); y = y - y.mean()
tr, te = masks(T); A = LDNField(H=48, theta=60.0).run(u)
print("absum: does DEPTH close the monotone gap?  (3 seeds, min over gamma incl. 3.0)")
print(f"  {'depth':>5} {'threshold':>10} {'crossing':>9} {'tanh':>7}")
for depth in (1, 2, 3, 4):
    row = []
    for act in ("threshold", "crossing", "tanh"):
        row.append(min(np.mean([ge(A, y, tr, te, depth, act, [g]*depth, seed=s)
                                for s in range(S)]) for g in G))
    print(f"  {depth:5d} {row[0]:10.3f} {row[1]:9.3f} {row[2]:7.3f}", flush=True)
