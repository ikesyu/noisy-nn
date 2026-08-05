"""Corrected gauge analysis (docs/idea_core.md: effective gain gamma=||a||/sigma is
the invariant; nu=E[z] and the local slope both PEAK on the bump FLANK and vanish
in the TAILS). Sharpening = raising gamma, but there is a CEILING at the flank:
past it, pre-activation swings into the tails and nu/slope DIE -- whether you
shrink sigma OR raise the gain (same gauge move). The earlier graded run set gains
to 2.5 = tails = vanishing (why it collapsed and did not reproduce bumpBN=0.552).

Part 1 (diagnostic): sweep the gain on BN'd pre-activation; measure nu (mean bump
activity) and mean |slope| -> locate the LIVE band.
Part 2 (corrected parity): keep gains in the LIVE band; graded [0.6->1.2] vs
uniform 1.0 vs learned, depths 2-4. Also show fixed gain 2.5 collapses (the bug).
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


def bump_slope(x):                              # d bump / dx
    P = 0.5 * (1 + torch.erf(x / _SQRT2))
    phi = torch.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
    return 2.0 * (1 - 2 * P) * phi


def diagnostic(A):
    """nu and mean|slope| of a bump layer vs gain (BN'd unit-variance pre-act)."""
    X = torch.tensor((A - A.mean(0)) / (A.std(0) + 1e-8), dtype=torch.float32)
    torch.manual_seed(0)
    lin = nn.Linear(A.shape[1], 48)
    p = lin(X)
    pn = (p - p.mean(0)) / (p.std(0) + 1e-8)     # BN to unit variance
    print("=== Part 1 diagnostic: nu and slope vs effective gain (sigma=1) ===")
    print(f"  {'gain(gamma)':>11} {'nu=E[z]':>9} {'mean|slope|':>12}   regime")
    for g in (0.3, 0.6, 1.0, 1.5, 2.5, 4.0):
        with torch.no_grad():
            nu = bump(g * pn).mean().item()
            sl = bump_slope(g * pn).abs().mean().item()
        reg = "PEAK-locked" if g <= 0.3 else ("live/flank" if g <= 1.2 else "TAILS/vanishing")
        print(f"  {g:11.1f} {nu:9.3f} {sl:12.4f}   {reg}")


class BumpLayer(nn.Module):
    def __init__(self, din, dout, g0, learn):
        super().__init__()
        self.lin = nn.Linear(din, dout)
        self.bn = nn.BatchNorm1d(dout, affine=False)
        self.bias = nn.Parameter(torch.zeros(dout))
        self.gain = nn.Parameter(torch.full((dout,), float(g0))) if learn else None
        self.fixed = None if learn else g0
    def forward(self, x):
        g = self.gain if self.gain is not None else self.fixed
        return bump(g * self.bn(self.lin(x)) + self.bias)


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


class TanhNet(nn.Module):
    def __init__(self, Hin, H, depth, seed=0):
        super().__init__(); torch.manual_seed(seed)
        dims = [Hin] + [H] * depth
        self.ls = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(depth)])
        self.out = nn.Linear(dims[-1], 1)
    def forward(self, x):
        for L in self.ls: x = torch.tanh(L(x))
        return self.out(x).squeeze(-1)


def train_eval(A, y, tr, te, make_net, epochs=2500, lr=3e-3, wd=1e-4, seed=0):
    mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
    X = torch.tensor(As, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    trm = torch.tensor(tr)
    net = make_net(A.shape[1], seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
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
    diagnostic(A)

    print("\n=== Part 2: parity with gains kept in the LIVE band (3 seeds) ===")
    for depth in (2, 3, 4):
        live_graded = list(np.linspace(0.6, 1.2, depth))
        nets = {
            "tanh": lambda Hin, s: TanhNet(Hin, H, depth, s),
            "bump fixed g=1.0": lambda Hin, s: BumpNet(Hin, H, depth, [1.0] * depth, False, s),
            "bump graded 0.6-1.2": lambda Hin, s: BumpNet(Hin, H, depth, live_graded, False, s),
            "bump learned(init 0.8)": lambda Hin, s: BumpNet(Hin, H, depth, [0.8] * depth, True, s),
            "bump fixed g=2.5 (tails)": lambda Hin, s: BumpNet(Hin, H, depth, [2.5] * depth, False, s),
        }
        print(f"  -- depth {depth} --")
        for name, mk in nets.items():
            es = [train_eval(A, y, tr, te, mk, seed=s) for s in range(3)]
            print(f"      {name:26s}: NRMSE={np.mean(es):.3f} (±{np.std(es):.2f})")


if __name__ == "__main__":
    main()
