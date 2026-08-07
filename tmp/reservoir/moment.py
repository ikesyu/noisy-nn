"""Shared harness for the MOMENT-ORDER activation comparisons (docs §10.24-10.36, §13.1).

Single home for the primitives that the reservoir_lambda_* / reservoir_gamma_* /
reservoir_absum_* / reservoir_narma_* / reservoir_noisematch / reservoir_numT
scripts had been re-defining or chain-importing from one another:

    masks        train/test split with washout
    DelayField   faithful linear delay line (lagged inputs as columns)
    SignField    tanh(gain*u) delay field (parity setting of §10.24/§10.27)
    parity_task  3-way sign parity target
    local_task   3-lobe local (multimodal) target of ONE lagged input
    LambdaAct    z = mean_t[(1-lam) b_t + lam |b_{t+1}-b_t|]  (sample forward,
                 analytic backward) -- lam=0 threshold/1st moment, lam=1
                 crossing/2nd moment
    AnaThr       analytic (mean-field) threshold  Phi(p)
    AnaCross     analytic (mean-field) crossing   2 Phi(p)(1-Phi(p))

The experiment-specific Nets / train_eval variants (learnable-gain BN, frozen
gamma, per-layer gamma schedule, learnable per-unit lambda, ...) stay in their
scripts: each one IS the experiment.  Definitions here are verbatim moves; the
scripts re-export what they used to define, so recorded runs reproduce exactly.
"""
import numpy as np
import torch

_S = np.sqrt(2.0)


def _Phi(x): return 0.5 * (1 + torch.erf(x / _S))
def _phi(x): return torch.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)


def masks(T, wo=300, fr=0.7):
    idx = np.arange(wo, T); n = int(len(idx) * fr)
    tr = np.zeros(T, bool); tr[idx[:n]] = True
    te = np.zeros(T, bool); te[idx[n:]] = True
    return tr, te


class DelayField:
    """faithful linear delay line: the field carries the lagged inputs."""
    def __init__(self, H=24): self.H = H
    def run(self, u):
        T = len(u); X = np.zeros((T, self.H))
        for k in range(self.H):
            X[k:, k] = u[:T - k]
        return X


class SignField:
    def __init__(self, H=32, gain=8.0): self.H, self.gain = H, gain
    def run(self, u):
        T = len(u); X = np.zeros((T, self.H))
        for k in range(self.H):
            X[k:, k] = np.tanh(self.gain * u[:T - k])
        return X


def parity_task(T, seed=0):
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    lag = lambda L: np.concatenate([np.zeros(L), u[:T - L]])
    y = np.sign(lag(2)) * np.sign(lag(6)) * np.sign(lag(10))
    return u, y - y.mean()


def local_task(T, lag=5, centers=(-0.6, 0.0, 0.6), w=0.18, seed=0):
    """multimodal (3-lobe) function of ONE lagged input: intrinsically local."""
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    x = np.concatenate([np.zeros(lag), u[:T - lag]])
    y = sum(np.exp(-((x - c) / w) ** 2) for c in centers)
    return u, y - y.mean()


class LambdaAct(torch.autograd.Function):
    """z = mean_t[(1-lam) b_t + lam |b_{t+1}-b_t|],  b_t = 1[p + n_t > h]."""
    @staticmethod
    def forward(ctx, p, lam, h, numT):
        n = torch.randn(numT, *p.shape)
        b = ((p.unsqueeze(0) + n) > h).float()
        rate = b.mean(0)                                   # 1st moment  -> Q(s)
        flip = (b - b.roll(-1, 0)).abs().mean(0)           # 2nd moment  -> 2Q(1-Q)
        P = _Phi(p - h)
        ctx.save_for_backward(P, _phi(p - h))
        ctx.lam = lam
        return (1.0 - lam) * rate + lam * flip

    @staticmethod
    def backward(ctx, g):
        P, ph = ctx.saved_tensors
        lam = ctx.lam
        slope = ((1.0 - lam) + lam * 2.0 * (1.0 - 2.0 * P)) * ph
        return g * slope, None, None, None


class AnaThr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p): ctx.save_for_backward(_phi(p)); return _Phi(p)
    @staticmethod
    def backward(ctx, g): (d,) = ctx.saved_tensors; return g * d


class AnaCross(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p):
        P = _Phi(p); ctx.save_for_backward(P, _phi(p)); return 2 * P * (1 - P)
    @staticmethod
    def backward(ctx, g):
        P, d = ctx.saved_tensors; return g * 2 * (1 - 2 * P) * d
