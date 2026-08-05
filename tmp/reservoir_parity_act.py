"""Does the crossing BUMP nonlinearity solve parity where a monotone (tanh)
nonlinearity fails? (docs/idea_reservoir.md 論点1 / §10.20 update.)

The NNN crossing phi(x)=2Phi(x/sigma)(1-Phi(x/sigma)) is a width-tunable BUMP
(local, non-monotone, RBF-like). Parity/XOR needs to carve space locally, so a
bump should beat a monotone ridge (tanh). §10.20 found bump vs monotone ~= 0 on
NARMA (a smooth task); the prediction is that on PARITY the gap opens.

Same sign field, same depth, same budget; swap ONLY the hidden activation and
sweep the bump width sigma. PASS if bump < tanh on parity and sharper bump (small
sigma) improves it -> "the crossing nonlinearity is the key to parity".
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


def bump(x, sigma):
    P = 0.5 * (1 + torch.erf(x / (sigma * _SQRT2)))
    return 2.0 * P * (1 - P)


class DeepNet(nn.Module):
    """depth hidden layers; act in {bump, tanh, relu}; bump width = sigma."""
    def __init__(self, Hin, H=48, depth=2, act="bump", sigma=1.0, seed=0):
        super().__init__(); torch.manual_seed(seed)
        dims = [Hin] + [H] * depth
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(depth)])
        self.out = nn.Linear(dims[-1], 1)
        self.act, self.sigma = act, sigma

    def _f(self, x):
        if self.act == "bump": return bump(x, self.sigma)
        if self.act == "tanh": return torch.tanh(x)
        return torch.relu(x)

    def forward(self, x):
        for lin in self.lins:
            x = self._f(lin(x))
        return self.out(x).squeeze(-1)


def train_eval(A, y, tr, te, depth, act, sigma=1.0, H=48, epochs=2500, lr=3e-3, seed=0):
    mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
    X = torch.tensor(As, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    trm = torch.tensor(tr)
    net = DeepNet(A.shape[1], H=H, depth=depth, act=act, sigma=sigma, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((net(X)[trm] - yt[trm]) ** 2).mean()
        loss.backward(); opt.step()
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

    print("=== activation comparison on parity (sign field, depth 2, 3 seeds) ===")
    for act, sig in (("tanh", None), ("relu", None),
                     ("bump", 1.0), ("bump", 0.5), ("bump", 0.25)):
        es = [train_eval(A, y, tr, te, 2, act, sigma=(sig or 1.0), seed=s) for s in range(3)]
        tag = f"{act}" + (f"(sigma={sig})" if sig else "")
        print(f"  {tag:14s}: NRMSE={np.mean(es):.3f} (+/-{np.std(es):.3f})")

    print("\n=== best bump-sigma across depth (parity) ===")
    for depth in (1, 2, 3):
        row = f"  depth {depth}:"
        for act, sig in (("tanh", 1.0), ("bump", 0.5), ("bump", 0.25)):
            es = [train_eval(A, y, tr, te, depth, act, sigma=sig, seed=s) for s in range(3)]
            row += f"  {act}{('' if act=='tanh' else f'({sig})')}={np.mean(es):.3f}"
        print(row)


if __name__ == "__main__":
    main()
