"""§10.25: sub-threshold-constrained threshold activation -- keep the on-brand
"zero without noise" property while staying monotone (deep-stable) and SR-optimal.

Coupling h = h_0/sigma (like consolidation's rho dial sigma=rho*sigma0, h=h0/rho):
    z = Phi((d - h_0/sigma)/sigma) = Phi(d/sigma - h_0/sigma^2)
  sigma->0  => argument -> -inf  => z -> 0  (DETERMINISTIC silence, on-brand)
  monotone in d (a sigmoid)      => deep-composition stable (unlike the bump)
  a global sigma sweep should give an SR interior optimum.

Checks on 3-way sign parity (sign field):
  (A) does it SOLVE deep parity (depths 1-4) like the unconstrained threshold?
  (B) on-brand: mean hidden activity -> 0 as sigma -> 0.
  (C) SR: parity NRMSE vs sigma has an interior optimum.
"""
import numpy as np
import torch
import torch.nn as nn

import reservoir as R
from reservoir.readout import standardize_fit, nrmse

_S = np.sqrt(2.0)


def masks(T, wo=300, fr=0.7):
    idx = np.arange(wo, T); n = int(len(idx) * fr)
    tr = np.zeros(T, bool); tr[idx[:n]] = True
    te = np.zeros(T, bool); te[idx[n:]] = True
    return tr, te


class SignField:
    def __init__(self, H=32, gain=8.0): self.H, self.gain = H, gain
    def run(self, u):
        T = len(u); X = np.zeros((T, self.H))
        for k in range(self.H):
            X[k:, k] = np.tanh(self.gain * u[:T - k])
        return X


def subthresh(p, sigma, h0):
    """z = Phi(p/sigma - h0/sigma^2); p is the BN'd pre-activation (unit var)."""
    arg = p / sigma - h0 / (sigma * sigma)
    return 0.5 * (1 + torch.erf(arg / _S))


class SubThreshNet(nn.Module):
    def __init__(self, Hin, H, depth, sigma, h0, seed=0):
        super().__init__(); torch.manual_seed(seed)
        d = [Hin] + [H] * depth
        self.ls = nn.ModuleList([nn.Linear(d[i], d[i + 1]) for i in range(depth)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(d[i + 1], affine=True) for i in range(depth)])
        self.out = nn.Linear(d[-1], 1)
        self.sigma, self.h0 = sigma, h0
        self.last_act = None

    def forward(self, x):
        for L, bn in zip(self.ls, self.bns):
            x = subthresh(bn(L(x)), self.sigma, self.h0)
        self.last_act = x
        return self.out(x).squeeze(-1)


def train_eval(A, y, tr, te, depth, sigma, h0, H=64, epochs=2000, lr=3e-3, seed=0,
               return_act=False):
    mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
    X = torch.tensor(As, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    trm = torch.tensor(tr)
    net = SubThreshNet(A.shape[1], H, depth, sigma, h0, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = ((net(X)[trm] - yt[trm]) ** 2).mean()
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        e = nrmse(y[te], net(X).numpy()[te])
        act = float(net.last_act.mean()) if return_act else None
    return (e, act) if return_act else e


def parity_task(T, seed=0):
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    lag = lambda L: np.concatenate([np.zeros(L), u[:T - L]])
    y = np.sign(lag(2)) * np.sign(lag(6)) * np.sign(lag(10))
    return u, y - y.mean()


def main():
    T = 3000; u, y = parity_task(T); tr, te = masks(T)
    A = SignField(H=32, gain=8.0).run(u)
    h0 = 0.5

    print(f"=== (A) sub-threshold (h=h0/sigma, h0={h0}) solves deep parity? ===")
    for depth in (1, 2, 3, 4):
        es = [train_eval(A, y, tr, te, depth, sigma=0.8, h0=h0, seed=s) for s in range(3)]
        print(f"  depth {depth} (sigma=0.8): NRMSE={np.mean(es):.3f} (±{np.std(es):.2f})")

    print(f"\n=== (B) on-brand: mean hidden activity vs sigma (should ->0 as sigma->0) ===")
    print(f"    and (C) SR: parity NRMSE vs sigma (depth 3) ===")
    print(f"  {'sigma':>6} {'parity NRMSE':>13} {'mean activity':>14}")
    for sigma in (2.5, 1.5, 1.0, 0.8, 0.6, 0.45, 0.35):
        res = [train_eval(A, y, tr, te, 3, sigma=sigma, h0=h0, seed=s, return_act=True)
               for s in range(3)]
        e = np.mean([r[0] for r in res]); a = np.mean([r[1] for r in res])
        print(f"  {sigma:6.2f} {e:13.3f} {a:14.4f}")


if __name__ == "__main__":
    main()
