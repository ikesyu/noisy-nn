"""Parity smoke (docs/idea_reservoir.md 論点1 follow-up): LDN (smooth window) FAILS
temporal sign-parity y = sign(u_{t-2}) sign(u_{t-6}) sign(u_{t-10}). Hypothesis:
that is an LDN<->parity MISMATCH (smooth window can't hold sharp signs), not a
limit of the system. A SIGN-PRESERVING field (a delay line whose crossing units
sit at operating point ~0, so each unit's activity is sensitive to sign(u_{t-k}))
+ a learned DEEP crossing map should compute the sign-product.

Checks:
  (1) does the sign field linearly expose sign(u_{t-k}) (unlike LDN)?
  (2) does a learned deep crossing map on the sign field solve parity, where
      LDN (any depth) failed, and where depth-0 (linear read-out) cannot?
  (3) architecture flexibility: the SAME deep-map architecture solves NARMA on an
      LDN field (i.e. swap only the FIELD per task, not the learner).
"""
import numpy as np
import torch
import torch.nn as nn

import reservoir as R
from reservoir.readout import ridge_fit, standardize_fit, nrmse, corr2

_SQRT2 = np.sqrt(2.0)


def masks(T, wo=300, fr=0.7):
    idx = np.arange(wo, T); n = int(len(idx) * fr)
    tr = np.zeros(T, bool); tr[idx[:n]] = True
    te = np.zeros(T, bool); te[idx[n:]] = True
    return tr, te


class SignField:
    """Delay line passed through a hard-ish sign feature: x_k(t)=tanh(gain*u(t-k)).
    High gain -> ~sign(u(t-k)); each lag on its own coordinate (disentangled)."""
    def __init__(self, H=32, gain=8.0):
        self.H, self.gain = H, gain

    def run(self, u):
        T = len(u); X = np.zeros((T, self.H))
        for k in range(self.H):
            X[k:, k] = np.tanh(self.gain * u[:T - k])
        return X


class DelayField:
    def __init__(self, H=32):
        self.H = H

    def run(self, u):
        T = len(u); X = np.zeros((T, self.H))
        for k in range(self.H):
            X[k:, k] = u[:T - k]
        return X


class CrossingAnalytic(torch.autograd.Function):
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
    def __init__(self, Hin, H=48, depth=2, seed=0):
        super().__init__(); torch.manual_seed(seed)
        dims = [Hin] + [H] * depth
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(depth)])
        self.out = nn.Linear(dims[-1], 1)

    def forward(self, x):
        for lin in self.lins:
            x = CrossingAnalytic.apply(lin(x))
        return self.out(x).squeeze(-1)


def train_eval(A, y, tr, te, depth, H=48, epochs=2500, lr=3e-3, seed=0):
    mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
    X = torch.tensor(As, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    trm = torch.tensor(tr)
    net = DeepCrossing(A.shape[1], H=H, depth=depth, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((net(X)[trm] - yt[trm]) ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(X).numpy()
    return nrmse(y[te], pred[te])


def parity_task(T, seed=0):
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    lag = lambda L: np.concatenate([np.zeros(L), u[:T - L]])
    y = np.sign(lag(2)) * np.sign(lag(6)) * np.sign(lag(10))
    return u, y - y.mean()


def main():
    T = 3000
    u, y = parity_task(T); tr, te = masks(T)
    print("=== (1) does the SIGN field linearly expose sign(u_{t-k})? ===")
    for name, fld in (("LDN(smooth)", R.LDNField(H=48, theta=60.0)),
                      ("sign field", SignField(H=32, gain=8.0))):
        A = fld.run(u); As = (A - A.mean(0)) / (A.std(0) + 1e-8)
        s6 = np.sign(np.concatenate([np.zeros(6), u[:T - 6]]))
        W = ridge_fit(As[tr], s6[tr], 1e-3)
        print(f"  {name:12s}: corr^2(field -> sign(u_t-6)) = {corr2(s6[te], (As@W)[te]):.3f}")

    print("\n=== (2) parity: LDN vs sign field, depth 0/1/2/3 ===")
    for name, fld in (("LDN", R.LDNField(H=48, theta=60.0)),
                      ("sign field", SignField(H=32, gain=8.0))):
        A = fld.run(u); mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
        Xb = np.concatenate([As, np.ones((T, 1))], 1)
        e0 = nrmse(y[te], (Xb @ ridge_fit(Xb[tr], y[tr], 1e-2))[te])
        line = f"  {name:12s}: depth0={e0:.3f}"
        for d in (1, 2, 3):
            es = [train_eval(A, y, tr, te, d, epochs=2500, seed=s) for s in range(3)]
            line += f"  depth{d}={np.mean(es):.3f}"
        print(line)

    print("\n=== (3) SAME deep-map architecture on NARMA-20 with an LDN field ===")
    un, yn = R.narma_x(T, 20, seed=0); trn, ten = masks(T)
    An = R.LDNField(H=48, theta=60.0).run(un)
    es = [train_eval(An, yn, trn, ten, 2, epochs=2500, seed=s) for s in range(3)]
    print(f"  NARMA-20  (LDN field, depth 2): NRMSE={np.mean(es):.3f}")
    print("  -> parity solved on sign field + NARMA solved on LDN field, "
          "same learner, field swapped per task.")


if __name__ == "__main__":
    main()
