"""§10.27: does NOISE (sample crossing) rescue the deep-parity collapse of the
ANALYTIC crossing?  Direct analytic-vs-sample contrast on the SAME crossing
activation z = 2 Phi(d/sigma)(1 - Phi(d/sigma)) (the 2021 crossing itself).

analytic : deterministic z = 2 Phi(p)(1-Phi(p))            (mean field)
sample   : z = MC over numT of the two-threshold XOR crossing at +/- h, then
           mean over T-samples (real crossing); backward = analytic slope
           2(1-2Phi(p)) phi(p).  Same forward mean as analytic, but finite-numT
           stochasticity in every mini-batch.
Single condition difference = the noise. On 3-way sign parity, sweep depth 1-4.
If sample stays healthy where analytic collapses -> "noise rescues deep bump".
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


def _Phi(x): return 0.5 * (1 + torch.erf(x / _S))
def _phi(x): return torch.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)


class CrossingAnalytic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p):                                 # p = d/sigma (BN'd)
        P = _Phi(p)
        ctx.save_for_backward(P, _phi(p))
        return 2.0 * P * (1 - P)
    @staticmethod
    def backward(ctx, g):
        P, ph = ctx.saved_tensors
        return g * 2.0 * (1 - 2 * P) * ph


class CrossingSample(torch.autograd.Function):
    """MC two-threshold crossing: for numT iid noise samples n_i ~ N(0,1), count
    sign flips of 1[p + n_i > +h] and 1[p + n_i > -h] between consecutive samples
    (cyclic), average the two -> real crossing statistic. backward = analytic slope."""
    @staticmethod
    def forward(ctx, p, h, numT):
        n = torch.randn(numT, *p.shape)
        s = p.unsqueeze(0) + n
        b1 = (s > h).float(); b2 = (s > -h).float()
        x1 = (b1 - b1.roll(-1, 0)).abs().mean(0)
        x2 = (b2 - b2.roll(-1, 0)).abs().mean(0)
        P = _Phi(p)
        ctx.save_for_backward(P, _phi(p))
        return 0.5 * (x1 + x2)
    @staticmethod
    def backward(ctx, g):
        P, ph = ctx.saved_tensors
        return g * 2.0 * (1 - 2 * P) * ph, None, None


class Net(nn.Module):
    def __init__(self, Hin, H, depth, mode, numT=64, h=0.2, seed=0):
        super().__init__(); torch.manual_seed(seed)
        d = [Hin] + [H] * depth
        self.ls = nn.ModuleList([nn.Linear(d[i], d[i + 1]) for i in range(depth)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(d[i + 1], affine=True) for i in range(depth)])
        self.out = nn.Linear(d[-1], 1)
        self.mode, self.numT, self.h = mode, numT, h
    def _act(self, p):
        if self.mode == "analytic":
            return CrossingAnalytic.apply(p)
        return CrossingSample.apply(p, self.h, self.numT)
    def forward(self, x):
        for L, bn in zip(self.ls, self.bns):
            x = self._act(bn(L(x)))
        return self.out(x).squeeze(-1)


def train_eval(A, y, tr, te, depth, mode, numT=64, H=48, steps=900, bs=256,
               lr=3e-3, seed=0):
    mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
    X = torch.tensor(As, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    net = Net(A.shape[1], H, depth, mode, numT=numT, seed=seed)
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


def parity_task(T, seed=0):
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    lag = lambda L: np.concatenate([np.zeros(L), u[:T - L]])
    y = np.sign(lag(2)) * np.sign(lag(6)) * np.sign(lag(10))
    return u, y - y.mean()


def main():
    T = 3000; u, y = parity_task(T); tr, te = masks(T)
    A = SignField(H=32, gain=8.0).run(u)
    print("=== crossing activation: analytic vs sample, deep parity (3 seeds) ===")
    print(f"  {'depth':>6} {'analytic':>10} {'sample(numT64)':>15}")
    for depth in (1, 2, 3, 4):
        ea = [train_eval(A, y, tr, te, depth, "analytic", seed=s) for s in range(3)]
        es = [train_eval(A, y, tr, te, depth, "sample", numT=64, seed=s) for s in range(3)]
        print(f"  {depth:6d} {np.mean(ea):10.3f} {np.mean(es):15.3f}")


if __name__ == "__main__":
    main()
