"""論点1 (docs/idea_reservoir.md): on a FIXED noise field, does a LEARNED DEEP
crossing map beat a single crossing layer and a linear read-out reservoir?

Contrast with standard RC (fixed reservoir + LINEAR read-out). Our position:
fixed field + forward-only-LEARNED nonlinear map + linear read-out. Because the
field is fixed, the crossing map is FEEDFORWARD (no time recurrence), so autograd
on it EQUALS the forward-only credit (no BPTT-through-time; §7.1/§10.2 cosine 1.0).
So we train deep crossing maps by autograd here -- it is the forward-only-
equivalent -- and sweep depth.

Models (all on the same fixed LDN field states A(t)):
  depth 0 : ridge read-out of A            (= standard-RC-style linear read-out)
  depth 1 : A -> [crossing hidden] -> linear
  depth 2 : A -> [cross] -> [cross] -> linear
  depth 3 : ...
Tasks: NARMA-20; a COMPOSED nonlinear target; a delayed sign-parity proxy.
"""
import numpy as np
import torch
import torch.nn as nn

import reservoir as R
from reservoir.readout import ridge_fit, standardize_fit, nrmse

_SQRT2 = np.sqrt(2.0)


def masks(T, wo=300, fr=0.7):
    idx = np.arange(wo, T); n = int(len(idx) * fr)
    tr = np.zeros(T, bool); tr[idx[:n]] = True
    te = np.zeros(T, bool); te[idx[n:]] = True
    return tr, te


class CrossingAnalytic(torch.autograd.Function):
    """Deterministic crossing E[z]=2P(1-P), P=Phi(x); grad 2(1-2P)phi(x)."""
    @staticmethod
    def forward(ctx, x):
        P = 0.5 * (1 + torch.erf(x / _SQRT2))
        phi = torch.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
        ctx.save_for_backward(P, phi)
        return 2.0 * P * (1 - P)

    @staticmethod
    def backward(ctx, g):
        P, phi = ctx.saved_tensors
        return g * 2.0 * (1 - 2 * P) * phi


class DeepCrossing(nn.Module):
    def __init__(self, Hin, H=48, depth=1, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        dims = [Hin] + [H] * depth
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(depth)])
        self.out = nn.Linear(dims[-1], 1)

    def forward(self, x):
        for lin in self.lins:
            x = CrossingAnalytic.apply(lin(x))
        return self.out(x).squeeze(-1)


def train_eval(A, y, tr, te, depth, H=48, epochs=2500, lr=3e-3, seed=0):
    mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
    X = torch.tensor(As, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    trm = torch.tensor(tr); tem = torch.tensor(te)
    net = DeepCrossing(A.shape[1], H=H, depth=depth, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    ymu = yt[trm].mean()
    for _ in range(epochs):
        opt.zero_grad()
        pred = net(X)
        loss = ((pred[trm] - yt[trm]) ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(X).numpy()
    return nrmse(y[te], pred[te])


def run(name, u, y, seeds=3, epochs=2500):
    T = len(u); tr, te = masks(T)
    A = R.LDNField(H=48, theta=60.0).run(u)
    # depth 0 = linear ridge read-out of the field (standard-RC-style)
    mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
    Xb = np.concatenate([As, np.ones((T, 1))], 1)
    W = ridge_fit(Xb[tr], y[tr], 1e-2)
    e0 = nrmse(y[te], (Xb @ W)[te])
    print(f"\n=== {name} ===")
    print(f"  depth 0 (linear read-out RC): NRMSE={e0:.3f}")
    for depth in (1, 2, 3):
        es = [train_eval(A, y, tr, te, depth, epochs=epochs, seed=s) for s in range(seeds)]
        print(f"  depth {depth} (learned crossing): NRMSE={np.mean(es):.3f} (+/-{np.std(es):.3f})")


def composed_task(T, seed=0):
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    lag = lambda L: np.concatenate([np.zeros(L), u[:T - L]])
    y = np.tanh(3 * ((lag(2) + lag(8)) ** 2 - lag(5) ** 2))
    return u, y - y.mean()


def parity_task(T, seed=0):
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    lag = lambda L: np.concatenate([np.zeros(L), u[:T - L]])
    y = np.sign(lag(2)) * np.sign(lag(6)) * np.sign(lag(10))
    return u, y - y.mean()


def main():
    T = 3000
    print("論点1: does LEARNED crossing depth help on a fixed LDN field?")
    run("NARMA-20", *R.narma_x(T, 20, seed=0))
    run("composed  y=tanh((u2+u8)^2 - u5^2)", *composed_task(T))
    run("3-way sign parity  y=sign(u2)sign(u6)sign(u10)", *parity_task(T))


if __name__ == "__main__":
    main()
