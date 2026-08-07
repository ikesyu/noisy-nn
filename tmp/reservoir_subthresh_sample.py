"""§10.25 (sample version, numT=100): sub-threshold threshold activation with the
h = h_0/sigma coupling, evaluated by REAL Monte-Carlo (finite numT), to test SR.

Forward: z = mean over numT of 1[d + eta > h], eta ~ N(0, sigma^2), h = h_0/sigma.
   (sample crossing barrier: sub-threshold signal cannot fire without noise.)
Backward: analytic slope of E[z] = Phi((d-h)/sigma) w.r.t. d (surrogate; the
   sample forward keeps the true finite-sample statistic, per §13.1's scheme).
Because h = h_0/sigma, sigma->0 => h->inf => z->0 deterministically (on-brand).

Checks on 3-way sign parity (sign field), numT=100:
  (A) deep parity (depths 1-4) at a working sigma;
  (B) on-brand: mean activity -> 0 as sigma -> 0;
  (C) SR: parity NRMSE vs sigma -- interior optimum expected (unlike analytic).
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


class SampleThreshold(torch.autograd.Function):
    """forward: MC estimate of P(d+eta>h), eta~N(0,sigma^2); backward: analytic
    d(E[z])/dd = phi((d-h)/sigma)/sigma (surrogate slope)."""
    @staticmethod
    def forward(ctx, d, sigma, h0, numT):
        h = h0 / sigma
        eta = torch.randn(numT, *d.shape, device=d.device) * sigma
        z = ((d.unsqueeze(0) + eta) > h).float().mean(0)          # [*d.shape]
        arg = (d - h) / sigma
        slope = torch.exp(-0.5 * arg * arg) / np.sqrt(2 * np.pi) / sigma
        ctx.save_for_backward(slope)
        return z

    @staticmethod
    def backward(ctx, g):
        (slope,) = ctx.saved_tensors
        return g * slope, None, None, None


class Net(nn.Module):
    def __init__(self, Hin, H, depth, sigma, h0, numT, seed=0):
        super().__init__(); torch.manual_seed(seed)
        d = [Hin] + [H] * depth
        self.ls = nn.ModuleList([nn.Linear(d[i], d[i + 1]) for i in range(depth)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(d[i + 1], affine=True) for i in range(depth)])
        self.out = nn.Linear(d[-1], 1)
        self.sigma, self.h0, self.numT = sigma, h0, numT
        self.last_act = None

    def forward(self, x):
        for L, bn in zip(self.ls, self.bns):
            x = SampleThreshold.apply(bn(L(x)), self.sigma, self.h0, self.numT)
        self.last_act = x
        return self.out(x).squeeze(-1)


def train_eval(A, y, tr, te, depth, sigma, h0, numT=100, H=64, steps=1500, bs=512,
               lr=3e-3, seed=0, return_act=False):
    """Mini-batch training (sample numT noise only for the batch, so numT=100 is
    affordable). Test evaluated full-batch in chunks."""
    mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
    X = torch.tensor(As, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    net = Net(A.shape[1], H, depth, sigma, h0, numT, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        b = torch.tensor(rng.choice(tr_idx, size=bs, replace=False))
        net.train(); opt.zero_grad()
        loss = ((net(X[b]) - yt[b]) ** 2).mean()
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        preds, acts = [], []
        for i in range(0, len(te_idx), 1000):
            j = te_idx[i:i + 1000]
            p = net(X[torch.tensor(j)]); preds.append(p.numpy())
            if return_act:
                acts.append(float(net.last_act.mean()))
        pred = np.concatenate(preds)
        e = nrmse(y[te_idx], pred)
        act = float(np.mean(acts)) if return_act else None
    return (e, act) if return_act else e


def parity_task(T, seed=0):
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    lag = lambda L: np.concatenate([np.zeros(L), u[:T - L]])
    y = np.sign(lag(2)) * np.sign(lag(6)) * np.sign(lag(10))
    return u, y - y.mean()


def main():
    T = 3000; u, y = parity_task(T); tr, te = masks(T)
    A = SignField(H=32, gain=8.0).run(u)
    h0, numT = 0.5, 100

    print(f"=== SAMPLE version (numT={numT}, h=h0/sigma, h0={h0}) ===")
    print("(A) deep parity at sigma=0.8:")
    for depth in (1, 2, 3, 4):
        es = [train_eval(A, y, tr, te, depth, 0.8, h0, numT=numT, seed=s) for s in range(3)]
        print(f"    depth {depth}: NRMSE={np.mean(es):.3f} (±{np.std(es):.2f})")

    print("\n(B) on-brand + (C) SR: parity NRMSE and mean activity vs sigma (depth 3):")
    print(f"  {'sigma':>6} {'parity NRMSE':>13} {'mean activity':>14}")
    for sigma in (2.5, 1.5, 1.0, 0.8, 0.6, 0.45, 0.35):
        res = [train_eval(A, y, tr, te, 3, sigma, h0, numT=numT, seed=s, return_act=True)
               for s in range(3)]
        e = np.mean([r[0] for r in res]); a = np.mean([r[1] for r in res])
        print(f"  {sigma:6.2f} {e:13.3f} {a:14.4f}")


if __name__ == "__main__":
    main()
