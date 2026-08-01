"""Stochastic-resonance (SR) diagnostic for the (B)-mix noise-modulated map
(docs/idea_reservoir.md §13.1).

The question: in (B)-mix, is the additive noise a functional RESOURCE, or just a
smooth way to implement a monotone nonlinearity? The test ports the classic SR
curve (recipe_sr.md) into the reservoir: with the SAME trained weights (d, M, c),
sweep the global noise strength s and score BOTH crossing responses:

    analytic (mean-field): z_k = 2 Phi(d_k/sigma_k) (1 - Phi(d_k/sigma_k))   -- h=0
    sample   (mechanism):  finite-numT Monte-Carlo crossing with threshold h>0

sigma_k(t) is the trained field-driven noise scale, scaled globally by s (the
neuromodulator-concentration axis). The threshold h is FIXED (not scaled), so at
small s the signal falls below h and cannot cross -> the SR barrier that the
mean-field lacks. Pass condition (§13.1): the sample curve has an interior optimum
(reverse-U) reproduced across seeds, while the analytic curve's optimum is gone /
much shallower (mean-field can rescale a vanishing feature with the read-out).
"""
import numpy as np
from scipy.special import ndtr

from .readout import ridge_fit, standardize_fit, nrmse, corr2

_SQRT2PI = np.sqrt(2.0 * np.pi)


def _softplus(z):
    return np.where(z > 30, z, np.log1p(np.exp(np.clip(z, -30, 30))))


def sigma_base(m):
    """Trained field-driven noise scale sigma_base[T, Ho] of a mixed (B) map.

    Uses the map's own standardised field self.A and learned M, c, floor, so the
    sweep shares the map's exact weights (the §13.1 same-weights requirement)."""
    if not getattr(m, "mix", False):
        raise ValueError("sr.sigma_base expects a mix=True NoiseModulatedMap")
    pre = m.A @ m.M.T + m.c
    return m.floor + _softplus(pre)


def analytic_z(d, sigma):
    """Mean-field crossing z[T, Ho] = 2 Phi(d/sigma)(1 - Phi(d/sigma))."""
    P = ndtr(d[None, :] / sigma)
    return 2.0 * P * (1.0 - P)


def sample_z(d, sigma, numT, h, seed, chunk=256):
    """Monte-Carlo crossing z[T, Ho] with numT noise samples and threshold h.

    Per (t, k): draw numT iid samples d_k + eta, eta ~ N(0, sigma_k(t)^2),
    binarise at +/-h, cyclic-XOR consecutive samples (a 'crossing'), average the
    two thresholds. -> 2 Phi(d/sigma)(1-Phi(d/sigma)) as numT->inf, h->0."""
    rng = np.random.default_rng(seed)
    T, Ho = sigma.shape
    z = np.empty((T, Ho))
    for i in range(0, T, chunk):
        s = sigma[i:i + chunk, :, None]                    # [c, Ho, 1]
        eta = rng.standard_normal((s.shape[0], Ho, numT)) * s
        x = d[None, :, None] + eta
        b1 = (x > h).astype(np.float64)
        b2 = (x > -h).astype(np.float64)
        x1 = np.abs(b1 - np.roll(b1, -1, axis=2)).mean(2)  # cyclic XOR along numT
        x2 = np.abs(b2 - np.roll(b2, -1, axis=2)).mean(2)
        z[i:i + chunk] = 0.5 * (x1 + x2)
    return z


def _standardise_design(z, tr):
    X = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    mu, sd = standardize_fit(X[tr])
    return (X - mu) / sd


def task_nrmse_feats(z, y, tr, te, alpha=1e-2):
    """Ridge read-out (re-solved on the swept features) -> test NRMSE."""
    Xs = _standardise_design(z, tr)
    W = ridge_fit(Xs[tr], y[tr], alpha)
    yh = Xs @ W
    return nrmse(y[te], yh[te])


def memory_capacity_feats(z, u, tr, te, kmax=60, alpha=1e-2):
    """Total memory capacity sum_k corr^2(u(t-k), reconstruction) from features.
    narma_x's drive u is iid uniform, so it doubles as the MC probe."""
    Xs = _standardise_design(z, tr)
    total = 0.0
    for k in range(1, kmax + 1):
        tgt = np.zeros_like(u)
        tgt[k:] = u[:-k]
        W = ridge_fit(Xs[tr], tgt[tr], alpha)
        yh = Xs @ W
        total += corr2(tgt[te], yh[te])
    return total
