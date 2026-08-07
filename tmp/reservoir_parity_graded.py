"""Graded sharpening across depth (docs/idea_consolidation.md gauge path idea):
sharpening EVERY bump layer risks a sudden loss of nu/slope as it compounds
through depth (why deep bumpBN collapsed). Hypothesis: make INPUT layers BROAD
(gentle bump, large support) and progressively SHARPER toward the OUTPUT, so nu
and slope stay alive layer-by-layer and only the last layers carve sharply.

We implement per-layer target sharpness via a fixed gain schedule multiplying the
(batch-normalised) pre-activation before the bump: gain_l grows from input to
output. Compare on 3-way sign parity, depths 2-4:
  tanh              : monotone reference (stable in depth)
  bump-uniform      : same gain every layer (the collapsing case)
  bump-graded       : gain increases input->output (the hypothesis)
  bump-graded-learn : per-layer gain LEARNED but initialised graded
"""
import numpy as np
import torch
import torch.nn as nn

import reservoir as R
from reservoir.readout import standardize_fit, nrmse

_SQRT2 = np.sqrt(2.0)


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


def bump(x):
    P = 0.5 * (1 + torch.erf(x / _SQRT2))
    return 2.0 * P * (1 - P)


class BumpLayer(nn.Module):
    """BN(pre) * gain -> bump. gain fixed (schedule) or learned (init=g0)."""
    def __init__(self, din, dout, g0, learn_gain):
        super().__init__()
        self.lin = nn.Linear(din, dout)
        self.bn = nn.BatchNorm1d(dout, affine=False)
        self.bias = nn.Parameter(torch.zeros(dout))
        if learn_gain:
            self.gain = nn.Parameter(torch.full((dout,), float(g0)))
            self.fixed = None
        else:
            self.gain = None; self.fixed = g0

    def forward(self, x):
        pn = self.bn(self.lin(x))
        g = self.gain if self.gain is not None else self.fixed
        return bump(g * pn + self.bias)


class TanhNet(nn.Module):
    def __init__(self, Hin, H, depth, seed=0):
        super().__init__(); torch.manual_seed(seed)
        dims = [Hin] + [H] * depth
        self.ls = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(depth)])
        self.out = nn.Linear(dims[-1], 1)
    def forward(self, x):
        for L in self.ls: x = torch.tanh(L(x))
        return self.out(x).squeeze(-1)


class BumpNet(nn.Module):
    def __init__(self, Hin, H, depth, gains, learn, seed=0):
        super().__init__(); torch.manual_seed(seed)
        dims = [Hin] + [H] * depth
        self.ls = nn.ModuleList([BumpLayer(dims[i], dims[i + 1], gains[i], learn)
                                 for i in range(depth)])
        self.out = nn.Linear(dims[-1], 1)
    def forward(self, x):
        for L in self.ls: x = L(x)
        return self.out(x).squeeze(-1)


def train_eval(A, y, tr, te, make_net, epochs=2500, lr=3e-3, seed=0):
    mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
    X = torch.tensor(As, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    trm = torch.tensor(tr)
    net = make_net(A.shape[1], seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = ((net(X)[trm] - yt[trm]) ** 2).mean()
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return nrmse(y[te], net(X).numpy()[te])


def parity_task(T, seed=0):
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    lag = lambda L: np.concatenate([np.zeros(L), u[:T - L]])
    y = np.sign(lag(2)) * np.sign(lag(6)) * np.sign(lag(10))
    return u, y - y.mean()


def main():
    T = 3000; u, y = parity_task(T); tr, te = masks(T)
    A = SignField(H=32, gain=8.0).run(u); H = 48
    print("=== parity: graded sharpening across depth (sign field, 3 seeds) ===")
    for depth in (2, 3, 4):
        graded = list(np.linspace(0.7, 2.5, depth))     # broad -> sharp
        uniform = [1.5] * depth
        nets = {
            "tanh":        lambda Hin, s: TanhNet(Hin, H, depth, s),
            "bump-uniform": lambda Hin, s: BumpNet(Hin, H, depth, uniform, False, s),
            "bump-graded":  lambda Hin, s: BumpNet(Hin, H, depth, graded, False, s),
            "bump-graded-learn": lambda Hin, s: BumpNet(Hin, H, depth, graded, True, s),
        }
        row = f"  depth {depth} (gain {['%.1f'%g for g in graded]}):"
        print(row)
        for name, mk in nets.items():
            es = [train_eval(A, y, tr, te, mk, seed=s) for s in range(3)]
            print(f"      {name:20s}: NRMSE={np.mean(es):.3f} (±{np.std(es):.2f})")


if __name__ == "__main__":
    main()
