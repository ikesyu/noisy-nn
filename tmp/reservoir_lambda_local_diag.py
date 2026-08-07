"""Diagnostic: is the 3-lobe local task learnable AT ALL, or is the harness broken?
Compare a plain tanh MLP (reference) against LambdaAct at lam=0 and lam=1,
with more capacity/steps.  If tanh also sits at NRMSE 1.0 the TASK is broken.
"""
import numpy as np
import torch
import torch.nn as nn

import reservoir as R
from reservoir.readout import standardize_fit, nrmse

from reservoir_lambda_local import (masks, DelayField, LambdaAct, local_task)


class RefNet(nn.Module):
    def __init__(self, Hin, H, act="tanh", seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.lin = nn.Linear(Hin, H); self.out = nn.Linear(H, 1); self.act = act
    def forward(self, x):
        z = torch.tanh(self.lin(x)) if self.act == "tanh" else torch.relu(self.lin(x))
        return self.out(z).squeeze(-1)


class LamNet(nn.Module):
    def __init__(self, Hin, H, lam, numT=100, bn=True, gain=1.0, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.lin = nn.Linear(Hin, H)
        self.bn = nn.BatchNorm1d(H, affine=True) if bn else None
        self.out = nn.Linear(H, 1)
        self.lam, self.numT, self.gain = lam, numT, gain
    def forward(self, x):
        p = self.lin(x)
        if self.bn is not None: p = self.bn(p)
        z = LambdaAct.apply(p * self.gain, self.lam, 0.0, self.numT)
        return self.out(z).squeeze(-1)


def run(net, A, y, tr, te, steps, bs=256, lr=3e-3, seed=0, wd=1e-4):
    mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
    X = torch.tensor(As, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        b = torch.tensor(rng.choice(tr_idx, size=bs, replace=False))
        net.train(); opt.zero_grad()
        loss = ((net(X[b]) - yt[b]) ** 2).mean()
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pr = [net(X[torch.tensor(te_idx[i:i+1000])]).numpy() for i in range(0, len(te_idx), 1000)]
    return nrmse(y[te_idx], np.concatenate(pr))


def main():
    T = 3000; u, y = local_task(T); tr, te = masks(T)
    A = DelayField(H=24).run(u)
    print(f"target stats: std={y.std():.3f}  frac|y|>0.1*std={np.mean(np.abs(y)>0.1*y.std()):.2f}")
    print("\n=== reference nets (deterministic activations) ===")
    for H in (8, 32):
        for act in ("tanh", "relu"):
            e = np.mean([run(RefNet(A.shape[1], H, act, seed=s), A, y, tr, te, 3000, seed=s)
                         for s in range(2)])
            print(f"  {act:5s} H={H:<3d} steps=3000 -> NRMSE {e:.3f}")
    print("\n=== LambdaAct, more capacity/steps, with & without BN, gain sweep ===")
    for lam in (0.0, 1.0):
        for H in (8, 32):
            for bn, gain in ((True, 1.0), (True, 3.0), (False, 1.0)):
                e = np.mean([run(LamNet(A.shape[1], H, lam, bn=bn, gain=gain, seed=s),
                                 A, y, tr, te, 3000, seed=s) for s in range(2)])
                tag = f"bn={bn} gain={gain}"
                print(f"  lam={lam:.1f} H={H:<3d} {tag:16s} -> NRMSE {e:.3f}", flush=True)


if __name__ == "__main__":
    main()
