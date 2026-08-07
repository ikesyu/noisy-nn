"""(b) FIXED: does lambda have an interior optimum on a LOCAL task?

v1 (reservoir_lambda_local.py) was invalid: a 24-tap delay field turned a sharp
3-lobe function of ONE lagged input into a feature-selection problem, and even a
plain tanh MLP collapsed (NRMSE 1.6 with the field vs 0.016 with the right column
alone).  Fix: small field (6 taps) so distractor dimensions cannot dominate.

Question unchanged: a monotone sigmoid needs 2 opposed units per local band, a
bump needs 1 -> under a UNIT BUDGET the 2nd-moment reading (high lambda) should
win.  Sweep hidden width (the budget) x lambda, with a tanh reference per width.
An interior/high-lambda optimum that appears only at small width = the honest
existence proof that modulating the moment order buys something.
"""
import numpy as np
import torch
import torch.nn as nn

import reservoir as R
from reservoir.readout import standardize_fit, nrmse

from reservoir_lambda_local import masks, DelayField, LambdaAct

_S = np.sqrt(2.0)


class Net(nn.Module):
    def __init__(self, Hin, H, lam, numT=100, act="lam", gain0=1.0, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.lin = nn.Linear(Hin, H)
        self.bn = nn.BatchNorm1d(H, affine=True)
        # sharpness of the bump = effective gain gamma; the lobes are ~0.3 wide in
        # standardised units while 2Q(1-Q) is ~2 wide, so gain must reach ~5.
        # BN's affine weight IS gamma and is learnable; gain0 sets its init.
        with torch.no_grad():
            self.bn.weight.fill_(gain0)
        self.out = nn.Linear(H, 1)
        self.lam, self.numT, self.act = lam, numT, act
    def forward(self, x):
        p = self.bn(self.lin(x))
        z = torch.tanh(p) if self.act == "tanh" else LambdaAct.apply(p, self.lam, 0.0, self.numT)
        return self.out(z).squeeze(-1)


def train_eval(A, y, tr, te, H, lam, act="lam", numT=100, steps=3000, bs=256,
               lr=3e-3, gain0=1.0, seed=0):
    mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
    X = torch.tensor(As, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    net = Net(A.shape[1], H, lam, numT=numT, act=act, gain0=gain0, seed=seed)
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


def local_task(T, lag=2, centers=(-0.6, 0.0, 0.6), w=0.18, seed=0):
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    x = np.concatenate([np.zeros(lag), u[:T - lag]])
    y = sum(np.exp(-((x - c) / w) ** 2) for c in centers)
    return u, y - y.mean()


def main():
    T = 3000; u, y = local_task(T); tr, te = masks(T)
    A = DelayField(H=6).run(u)              # small field: no feature-selection trap
    lams = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    Hs = [3, 4, 6, 10]
    SEEDS = 2
    print("=== (b, fixed) 3-lobe local task, 6-tap field, DEPTH 1, 2 seeds, numT=100 ===")
    print("  A single hidden layer must build 3 lobes.  A monotone unit needs a PAIR")
    print("  per lobe (hard for SGD: tanh depth1 fails at ~0.97 even with H=64),")
    print("  a bump needs ONE unit.  So this is where reading the 2nd moment should pay.")
    for gain0 in (1.0, 4.0):
        print(f"\n--- BN gain init = {gain0} (bump sharpness; lobes need gamma ~5) ---")
        print("  H\\lam " + "  ".join(f"{l:5.2f}" for l in lams) + "  | tanh  | best")
        for H in Hs:
            vals = []
            for lam in lams:
                vals.append(np.mean([train_eval(A, y, tr, te, H, lam, gain0=gain0, seed=s)
                                     for s in range(SEEDS)]))
            tanh_e = np.mean([train_eval(A, y, tr, te, H, 0.0, act="tanh", gain0=gain0, seed=s)
                              for s in range(SEEDS)])
            i = int(np.argmin(vals))
            flag = ""
            if vals[i] < vals[0] * 0.95:
                flag = f"  <-- lam={lams[i]:.1f} beats lam0 by {100*(vals[0]-vals[i])/vals[0]:.0f}%"
            print(f"  {H:<3d} " + "  ".join(f"{v:5.2f}" for v in vals)
                  + f"  | {tanh_e:5.2f} | lam={lams[i]:.1f}{flag}", flush=True)


if __name__ == "__main__":
    main()
