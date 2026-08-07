"""lambda-interpolation between THRESHOLD (1st moment) and CROSSING (2nd moment).

One readout gate lambda in [0,1] mixes, on the SAME binarised noisy signal
b_t = 1[d + eta_t > h], the two counters that idea_core sec2.5 already produces:

    z_lambda = mean_t[ (1-lambda) * b_t  +  lambda * |b_{t+1} - b_t| ]

Statistic level (adjacent samples independent):

    zbar_lambda(s) = (1-lambda) Q(s) + lambda * 2 Q(s)(1-Q(s))
                   = Q(s) [1 + lambda - 2 lambda Q(s)]

    lambda = 0  ->  Q(s)            = THRESHOLD activation (monotone sigmoid, 2018)
    lambda = 1  ->  2Q(1-Q)         = CROSSING activation (bump, 2021)

PREDICTION (from d/ds sign analysis): zbar_lambda is monotone on the whole axis
iff 1 - 3 lambda >= 0, i.e. **lambda <= 1/3**.  Above 1/3 a non-monotone (bump)
lobe appears.  Since deep stability tracks monotonicity (sec2.7.5), deep parity
should stay healthy for lambda <~ 1/3 and collapse above it.

Forward = MC over numT samples (real sample-level statistic).
Backward = analytic slope  [(1-lambda) + 2 lambda (1 - 2 Phi)] * phi   (straight-through).
Task = 3-way sign parity on a sign field, depths 1-4 (same harness as
reservoir_crossing_deep.py / reservoir_depth.py).
"""
import numpy as np
import torch
import torch.nn as nn

import reservoir as R
from reservoir.readout import standardize_fit, nrmse
# shared harness moved verbatim to the package (this script defined it first)
from reservoir.moment import masks, SignField, parity_task, LambdaAct, _Phi, _phi


class Net(nn.Module):
    def __init__(self, Hin, H, depth, lam, numT=100, h=0.0, seed=0):
        super().__init__(); torch.manual_seed(seed)
        d = [Hin] + [H] * depth
        self.ls = nn.ModuleList([nn.Linear(d[i], d[i + 1]) for i in range(depth)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(d[i + 1], affine=True) for i in range(depth)])
        self.out = nn.Linear(d[-1], 1)
        self.lam, self.numT, self.h = lam, numT, h
    def forward(self, x):
        for L, bn in zip(self.ls, self.bns):
            x = LambdaAct.apply(bn(L(x)), self.lam, self.h, self.numT)
        return self.out(x).squeeze(-1)


def train_eval(A, y, tr, te, depth, lam, numT=100, H=48, steps=900, bs=256,
               lr=3e-3, seed=0):
    mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
    X = torch.tensor(As, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    net = Net(A.shape[1], H, depth, lam, numT=numT, seed=seed)
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


def shape_check(lams):
    """Print the analytic monotonicity boundary of zbar_lambda for the record."""
    print("=== analytic shape check: min_s d(zbar)/ds sign  (monotone iff 1-3lam>=0) ===")
    s = torch.linspace(-6, 6, 2001)
    P = _Phi(s); ph = _phi(s)
    for lam in lams:
        sl = ((1 - lam) + lam * 2 * (1 - 2 * P)) * ph
        mono = "monotone" if sl.min() >= -1e-12 else "NON-monotone (bump)"
        print(f"  lam={lam:4.2f}  min slope={sl.min():+.4f}   {mono}")
    print(f"  -> predicted critical lambda = 1/3 = {1/3:.3f}\n")


def main():
    T = 3000; u, y = parity_task(T); tr, te = masks(T)
    A = SignField(H=32, gain=8.0).run(u)
    lams = [0.0, 0.15, 0.30, 1 / 3, 0.40, 0.55, 0.75, 1.0]
    shape_check(lams)
    print("=== 3-way sign parity NRMSE, sample numT=100, 2 seeds ===")
    print("  lam:   " + "  ".join(f"{l:5.2f}" for l in lams)
          + "     (0=threshold/1st, 1=crossing/2nd)")
    for depth in (1, 2, 3, 4):
        row = f"  depth{depth}:"
        for lam in lams:
            es = [train_eval(A, y, tr, te, depth, lam, seed=s) for s in range(2)]
            row += f"  {np.mean(es):5.2f}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
