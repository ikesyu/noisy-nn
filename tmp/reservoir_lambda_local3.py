"""(b) v3: isolate the ACTUAL hypothesis by removing the alignment confound.

v2 result: on a 6-tap field, depth 1, NOTHING works (all lambdas AND tanh sit at
0.83-1.01, i.e. barely better than predicting the mean; 2-layer tanh gets 0.016).
Diagnosis: a depth-1 unit must simultaneously (1) align w to the ONE relevant tap
out of 6 and (2) build a lobe.  The alignment is what SGD fails at, and it swamps
any activation-shape effect.  Any "lambda wins by 10%" read off that table would
be tea leaves inside a failure regime.

The hypothesis was never about alignment; it is about REPRESENTATION EFFICIENCY:
  a bump makes a lobe with ONE unit, a monotone sigmoid needs a PAIR.
So give the net the relevant coordinate directly (1-D input) and sweep the unit
budget H.  Prediction: high lambda solves at H~3 (one bump per lobe) while
lambda=0 needs H~6+ (two opposed sigmoids per lobe) or fails.

This is a MECHANISM test of the activation, not a reservoir task.
"""
import numpy as np
import torch

import reservoir as R
from reservoir.readout import standardize_fit, nrmse

from reservoir_lambda_local import masks, DelayField
from reservoir_lambda_local2 import Net, local_task


def train_eval(X_np, y, tr, te, H, lam, act="lam", numT=100, steps=3000, bs=256,
               lr=3e-3, gain0=1.0, seed=0):
    mu, sd = standardize_fit(X_np[tr]); Xs = (X_np - mu) / sd
    X = torch.tensor(Xs, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    net = Net(X.shape[1], H, lam, numT=numT, act=act, gain0=gain0, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(steps):
        b = torch.tensor(rng.choice(tr_idx, size=bs, replace=False))
        net.train(); opt.zero_grad()
        loss = ((net(X[b]) - yt[b]) ** 2).mean()
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pr = [net(X[torch.tensor(te_idx[i:i+1000])]).numpy()
              for i in range(0, len(te_idx), 1000)]
    return nrmse(y[te_idx], np.concatenate(pr))


def main():
    T = 3000; u, y = local_task(T, lag=2); tr, te = masks(T)
    A = DelayField(H=6).run(u)
    X1 = A[:, 2:3]                      # the relevant lagged input, alone
    lams = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    Hs = [2, 3, 4, 6, 8, 16]
    SEEDS = 3
    print("=== (b v3) 3-lobe target, 1-D input (no alignment confound), depth 1 ===")
    print("    3 lobes: a bump needs ~3 units, a monotone sigmoid needs ~6")
    for gain0 in (1.0, 4.0):
        print(f"\n--- BN gain init = {gain0} ---")
        print("  H\\lam " + "  ".join(f"{l:5.2f}" for l in lams) + "  | tanh")
        for H in Hs:
            vals = [np.mean([train_eval(X1, y, tr, te, H, lam, gain0=gain0, seed=s)
                             for s in range(SEEDS)]) for lam in lams]
            tanh_e = np.mean([train_eval(X1, y, tr, te, H, 0.0, act="tanh",
                                         gain0=gain0, seed=s) for s in range(SEEDS)])
            print(f"  {H:<3d} " + "  ".join(f"{v:5.2f}" for v in vals)
                  + f"  | {tanh_e:5.2f}", flush=True)


if __name__ == "__main__":
    main()
