"""§10.26: Delta_h sweep interpolates CROSSING(bump) <-> THRESHOLD(sigmoid).

Band activation (2021 KDE-slope form): z_band = P(h-dh < d+eta < h+dh)
  = Phi((d-h+dh)/sigma) - Phi((d-h-dh)/sigma), normalised to peak 1.
On-brand via h = h_0/sigma (sigma->0 => band at +inf => z->0).
  dh SMALL  -> narrow bump = CROSSING (non-monotone, local; deep-collapse?)
  dh LARGE  -> wide band -> the lower threshold dominates -> monotone SIGMOID
              (deep-stable). So dh is a single knob crossing<->threshold.

Sample version (numT), forward = MC band membership, backward = analytic band slope.
Test: 3-way sign parity, depths 1-4, sweep dh -> does deep collapse (small dh)
turn into deep stability (large dh)? Confirms the unification + the KDE-slope band.
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


class SampleBand(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, sigma, h0, dh, numT):
        h = h0 / sigma
        eta = torch.randn(numT, *d.shape) * sigma
        s = d.unsqueeze(0) + eta
        band = ((s > h - dh) & (s < h + dh)).float().mean(0)     # MC band membership
        norm = 2 * _Phi(torch.tensor(dh / sigma)) - 1            # analytic peak
        z = band / (norm + 1e-8)
        a1 = (d - h + dh) / sigma; a2 = (d - h - dh) / sigma
        slope = (_phi(a1) - _phi(a2)) / sigma / (norm + 1e-8)    # analytic band slope
        ctx.save_for_backward(slope)
        return z

    @staticmethod
    def backward(ctx, g):
        (slope,) = ctx.saved_tensors
        return g * slope, None, None, None, None


class Net(nn.Module):
    def __init__(self, Hin, H, depth, sigma, h0, dh, numT, seed=0):
        super().__init__(); torch.manual_seed(seed)
        d = [Hin] + [H] * depth
        self.ls = nn.ModuleList([nn.Linear(d[i], d[i + 1]) for i in range(depth)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(d[i + 1], affine=True) for i in range(depth)])
        self.out = nn.Linear(d[-1], 1)
        self.sigma, self.h0, self.dh, self.numT = sigma, h0, dh, numT
    def forward(self, x):
        for L, bn in zip(self.ls, self.bns):
            x = SampleBand.apply(bn(L(x)), self.sigma, self.h0, self.dh, self.numT)
        return self.out(x).squeeze(-1)


def train_eval(A, y, tr, te, depth, dh, sigma=0.8, h0=0.5, numT=64, H=48,
               steps=900, bs=256, lr=3e-3, seed=0):
    mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
    X = torch.tensor(As, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    net = Net(A.shape[1], H, depth, sigma, h0, dh, numT, seed=seed)
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
    dhs = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    print("=== parity NRMSE vs Delta_h (crossing<->threshold), sample numT=100 ===")
    print("  dh:    " + "  ".join(f"{d:>5}" for d in dhs) + "   (small=bump, large=sigmoid)")
    for depth in (1, 2, 3, 4):
        row = f"  depth{depth}:"
        for dh in dhs:
            es = [train_eval(A, y, tr, te, depth, dh, seed=s) for s in range(2)]
            row += f"  {np.mean(es):5.2f}"
        print(row)


if __name__ == "__main__":
    main()
