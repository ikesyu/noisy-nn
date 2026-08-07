"""(b) Does the lambda knob have an INTERIOR OPTIMUM?  I.e. is there a task where
reading the 2nd moment (bump) genuinely BEATS reading the 1st (monotone)?

reservoir_lambda.py showed lambda only ever COSTS on deep parity (lam=0 best).
If lambda has no upside anywhere, "modulate the moment order" has no purpose.

The place a bump should win is LOCAL representation under a UNIT BUDGET:
a monotone sigmoid needs 2 opposed units per local band, a bump needs 1.
So: multimodal target (3 Gaussian lobes) of one lagged input, depth 1,
sweep hidden width H (the budget) x lambda.

PREDICTION: at small H, high lambda (bump) wins; as H grows the advantage
vanishes (both are universal).  An interior/edge optimum at lambda>0 that
appears only under budget pressure = the honest existence proof.

Same LambdaAct as reservoir_lambda.py (forward MC over numT, backward analytic).
"""
import numpy as np
import torch
import torch.nn as nn

import reservoir as R
from reservoir.readout import standardize_fit, nrmse
# shared harness (moved verbatim to the package; re-exported here for the
# scripts that import masks/DelayField/LambdaAct/local_task from this file)
from reservoir.moment import masks, DelayField, LambdaAct, local_task


class Net(nn.Module):
    def __init__(self, Hin, H, lam, numT=100, h=0.0, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.lin = nn.Linear(Hin, H)
        self.bn = nn.BatchNorm1d(H, affine=True)   # affine => effective gain is learnable
        self.out = nn.Linear(H, 1)
        self.lam, self.numT, self.h = lam, numT, h
    def forward(self, x):
        z = LambdaAct.apply(self.bn(self.lin(x)), self.lam, self.h, self.numT)
        return self.out(z).squeeze(-1)


def train_eval(A, y, tr, te, H, lam, numT=100, steps=1200, bs=256, lr=3e-3, seed=0):
    mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
    X = torch.tensor(As, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    net = Net(A.shape[1], H, lam, numT=numT, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        b = torch.tensor(rng.choice(tr_idx, size=bs, replace=False))
        net.train(); opt.zero_grad()
        loss = ((net(X[b]) - yt[b]) ** 2).mean()
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        preds = [net(X[torch.tensor(te_idx[i:i + 1000])]).numpy()
                 for i in range(0, len(te_idx), 1000)]
    return nrmse(y[te_idx], np.concatenate(preds))


def main():
    T = 3000; u, y = local_task(T); tr, te = masks(T)
    A = DelayField(H=24).run(u)
    lams = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    Hs = [3, 4, 6, 10, 16]
    print("=== (b) local 3-lobe task, depth 1: NRMSE vs lambda x hidden width H ===")
    print("   (lam 0 = threshold/monotone, 1 = crossing/bump; 3 seeds, numT=100)")
    print("  lam:   " + "  ".join(f"{l:5.2f}" for l in lams))
    best = {}
    for H in Hs:
        row = f"  H={H:<3d}:"
        vals = []
        for lam in lams:
            es = [train_eval(A, y, tr, te, H, lam, seed=s) for s in range(3)]
            vals.append(np.mean(es)); row += f"  {np.mean(es):5.2f}"
        i = int(np.argmin(vals)); best[H] = (lams[i], vals[i], vals[0])
        row += f"   | best lam={lams[i]:.1f} ({vals[i]:.2f}) vs lam0 ({vals[0]:.2f})"
        print(row, flush=True)
    print("\n=== summary: does bump-reading win under budget pressure? ===")
    for H in Hs:
        lam_b, v_b, v_0 = best[H]
        gain = (v_0 - v_b) / max(v_0, 1e-9) * 100
        print(f"  H={H:<3d}  best lam={lam_b:.1f}  NRMSE {v_b:.3f} vs lam0 {v_0:.3f}"
              f"  ({gain:+.1f}% vs monotone)")


if __name__ == "__main__":
    main()
