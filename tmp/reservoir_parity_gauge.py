"""Gauge-aware bump for parity (docs/idea_consolidation.md §3-4): shrinking sigma
is the VANISHING PATH -- the bump becomes a narrow spike, so nu (crossing rate)
and slope die almost everywhere -> no gradient (why sigma=0.25 failed earlier).

The correct way to SHARPEN the response to the input is to raise the effective
GAIN (w/sigma) while KEEPING the operating point on the bump's high-slope FLANK
(nu in an intermediate band), so nu and slope stay alive. We implement this by
NORMALISING the pre-activation (batch-norm: centre + unit-variance) before the
bump and applying a LEARNED per-unit gain -- this pins units on the flank and
sharpens through gain, not through sigma->0. sigma is fixed at a live value.

Test on 3-way sign parity: does gauge-aware bump (BN + learned gain) match/beat
tanh, where plain bump lost (§10.20 update)?
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


def bump(x):                                    # sigma fixed = 1 (live width)
    P = 0.5 * (1 + torch.erf(x / _SQRT2))
    return 2.0 * P * (1 - P)


class Layer(nn.Module):
    """act in {tanh, bump, bumpBN}. bumpBN: BN(pre) * learned-gain -> bump, which
    keeps nu/slope alive (flank) and sharpens via gain (the gauge-aware path)."""
    def __init__(self, din, dout, act, tr_mask):
        super().__init__()
        self.lin = nn.Linear(din, dout); self.act = act
        self.tr = tr_mask
        if act == "bumpBN":
            self.bn = nn.BatchNorm1d(dout, affine=False)
            self.gain = nn.Parameter(torch.ones(dout))
            self.bias = nn.Parameter(torch.zeros(dout))

    def forward(self, x):
        p = self.lin(x)
        if self.act == "tanh":
            return torch.tanh(p)
        if self.act == "bump":
            return bump(p)
        # bumpBN: normalise pre-activation (train stats), sharpen via learned gain
        pn = self.bn(p)
        return bump(self.gain * pn + self.bias)


class DeepNet(nn.Module):
    def __init__(self, Hin, H, depth, act, tr_mask, seed=0):
        super().__init__(); torch.manual_seed(seed)
        dims = [Hin] + [H] * depth
        self.layers = nn.ModuleList([Layer(dims[i], dims[i + 1], act, tr_mask)
                                     for i in range(depth)])
        self.out = nn.Linear(dims[-1], 1)

    def forward(self, x):
        for L in self.layers:
            x = L(x)
        return self.out(x).squeeze(-1)


def train_eval(A, y, tr, te, depth, act, H=48, epochs=2500, lr=3e-3, seed=0):
    mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
    X = torch.tensor(As, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    trm = torch.tensor(tr)
    net = DeepNet(A.shape[1], H, depth, act, trm, seed=seed)
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
    A = SignField(H=32, gain=8.0).run(u)
    print("=== parity: gauge-aware bump (BN+gain) vs plain bump vs tanh ===")
    print("  (sign field, 3 seeds; bumpBN keeps nu/slope alive, sharpens via gain)")
    for depth in (1, 2, 3):
        row = f"  depth {depth}:"
        for act in ("tanh", "bump", "bumpBN"):
            es = [train_eval(A, y, tr, te, depth, act, seed=s) for s in range(3)]
            row += f"  {act}={np.mean(es):.3f}(±{np.std(es):.2f})"
        print(row)


if __name__ == "__main__":
    main()
