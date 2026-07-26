"""The NNN crossing map — the NONLINEARITY of the separated architecture,
learned FORWARD-ONLY (no BPTT).

A fixed noise field supplies memory as exogenous states a(t) [T, Hf]; the map
places a hidden crossing layer on them,

    z_j(t) = 2 Phi(pre_j/s) (1 - Phi(pre_j/s)),   pre = W1 a + b,

read out linearly. Because the field is fixed, z(t) has NO temporal recurrence in
theta=(W1,b), so dL/dtheta is a static per-timestep sum — no time unrolling, no
BPTT (this is the whole point, docs/idea_reservoir.md §7.1). The credit uses the
crossing's analytic slope and the read-out weights; a weight-decayed Adam with a
ridge read-out re-solved each step. (A genuine forward-noise / cov_jac estimator
gives the same gradient to cosine 1.0; see reservoir_covjac.py in tmp/.)
"""
import numpy as np
from scipy.special import ndtr

from .readout import ridge_fit, standardize_fit, nrmse

_SQRT2PI = np.sqrt(2.0 * np.pi)


def _norm_pdf(u):
    return np.exp(-0.5 * u * u) / _SQRT2PI


class LearnedCrossingMap:
    """Forward-only hidden crossing map on a fixed field. Ho hidden units."""

    def __init__(self, A, y, tr, Ho, s=1.0, alpha=1e-2, lr=0.02, wd=1e-2, seed=0):
        mu, sd = standardize_fit(A[tr])
        self.A = (A - mu) / sd                  # standardise the fixed field once
        self.y, self.tr, self.Ho, self.s = y, tr, Ho, s
        self.alpha, self.lr, self.wd = alpha, lr, wd
        rng = np.random.default_rng(seed)
        Hf = A.shape[1]
        self.W1 = rng.standard_normal((Ho, Hf)) / np.sqrt(Hf)
        self.b = rng.uniform(-1, 1, size=Ho)
        self._m1 = np.zeros_like(self.W1); self._v1 = np.zeros_like(self.W1)
        self._mb = np.zeros(Ho); self._vb = np.zeros(Ho); self._t = 0

    def _feat(self):
        pre = self.A @ self.W1.T + self.b
        arg = pre / self.s
        P = ndtr(arg)
        return 2 * P * (1 - P), 2 * (1 - 2 * P) * _norm_pdf(arg) / self.s

    def _readout(self, z):
        X = np.concatenate([z, np.ones((len(z), 1))], axis=1)
        mu, sd = standardize_fit(X[self.tr])
        W = ridge_fit(((X - mu) / sd)[self.tr], self.y[self.tr], self.alpha)
        return W, mu, sd, X

    def predict_all(self, W1=None, b=None):
        """Prediction over all timesteps (optionally with a fixed random map)."""
        w1 = self.W1 if W1 is None else W1
        bb = self.b if b is None else b
        arg = (self.A @ w1.T + bb) / self.s
        z = 2 * ndtr(arg) * (1 - ndtr(arg))
        W, mu, sd, X = self._readout(z)
        return (X - mu) / sd @ W

    def step(self):
        self._t += 1
        z, slope = self._feat()
        W, mu, sd, X = self._readout(z)
        e = np.where(self.tr, (X - mu) / sd @ W - self.y, 0.0)
        n = self.tr.sum()
        Wz = W[:self.Ho] / sd[0, :self.Ho]
        g_pre = (2.0 / n) * e[:, None] * Wz[None, :] * slope
        gW1 = g_pre.T @ self.A + self.wd * self.W1
        gb = g_pre.sum(axis=0)
        c = min(1.0, 1.0 / (np.sqrt((gW1 ** 2).sum() + (gb ** 2).sum()) + 1e-12))
        for p, g, m, v in ((self.W1, gW1 * c, self._m1, self._v1),
                           (self.b, gb * c, self._mb, self._vb)):
            m[:] = 0.9 * m + 0.1 * g
            v[:] = 0.999 * v + 0.001 * g * g
            p -= self.lr * (m / (1 - 0.9 ** self._t)) / (np.sqrt(v / (1 - 0.999 ** self._t)) + 1e-8)

    def train(self, epochs):
        for _ in range(epochs):
            self.step()
        return self

    def eval(self, te, epochs):
        self.train(epochs)
        return nrmse(self.y[te], self.predict_all()[te])


class NoiseModulatedMap:
    """Model (B) — the FAITHFUL noise-modulated NNN (docs §10.18-10.19).

    The noise field is the NNN's ADDITIVE per-unit NOISE (not the input): unit k
    has a FIXED operating point d_k, and its noise scale is driven by the field,
        z_k(t) = 2 Phi(d_k/sigma_k(t)) (1 - Phi(d_k/sigma_k(t))),
        sigma_k(t) = floor + softplus(pre_k(t)),
    read out linearly. Two couplings of the field to the noise (docs §10.19):
      mix=False (diagonal): pre_k = g_k A_k(t) + c_k  — unit k reads field coord k.
          z is monotone in a single coordinate, so the read-out is a GAM of the
          field coords: it beats ESN only when the field's COORDINATES pre-mix
          lags (LDN) — a single-lag delay field fails.
      mix=True (mixed):     pre_k = (M A(t))_k + c_k  — unit k reads a LEARNED
          linear combination of all field coords. z_k is then a monotone
          (sigmoidal) function of a projection, i.e. a standard hidden unit, so
          the lag-mixing can happen in the NOISE MAP: even a single-lag field
          works. The residual gap to (A) is the monotone-vs-bump difference (A's
          crossing is non-monotone in its projection); see docs §10.19.

    d_k, the noise map (g,c or M,c) and the read-out are learned forward-only."""

    def __init__(self, A, y, tr, Ho, floor=0.25, mix=False,
                 alpha=1e-2, lr=0.03, wd=1e-3, seed=0):
        mu, sd = standardize_fit(A[tr])
        self.A = (A - mu) / sd
        self.Hf, self.Ho, self.mix = A.shape[1], Ho, mix
        self.y, self.tr, self.floor = y, tr, floor
        self.alpha, self.lr, self.wd = alpha, lr, wd
        rng = np.random.default_rng(seed)
        self.d = rng.uniform(-2, 2, Ho)                 # fixed-input operating points
        self.c = np.zeros(Ho)                           # per-unit noise offset
        if mix:
            self.M = rng.standard_normal((Ho, self.Hf)) / np.sqrt(self.Hf)
            self._st = {"d": self._z2(Ho), "M": self._z2((Ho, self.Hf)), "c": self._z2(Ho)}
        else:
            self.idx = np.arange(Ho) % self.Hf          # unit k reads field coord k
            self.g = np.ones(Ho)
            self._st = {"d": self._z2(Ho), "g": self._z2(Ho), "c": self._z2(Ho)}
        self._t = 0

    @staticmethod
    def _z2(shape):
        return [np.zeros(shape), np.zeros(shape)]

    @staticmethod
    def _softplus(z):
        return np.where(z > 30, z, np.log1p(np.exp(np.clip(z, -30, 30))))

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _pre(self):
        if self.mix:
            return self.A @ self.M.T + self.c
        return self.g * self.A[:, self.idx] + self.c

    def _feat(self):
        pre = self._pre()
        sig = self.floor + self._softplus(pre)
        arg = self.d / sig
        P = ndtr(arg)
        return 2 * P * (1 - P), sig, 2 * (1 - 2 * P) * _norm_pdf(arg), pre

    def _readout(self, z):
        X = np.concatenate([z, np.ones((len(z), 1))], axis=1)
        mu, sd = standardize_fit(X[self.tr])
        W = ridge_fit(((X - mu) / sd)[self.tr], self.y[self.tr], self.alpha)
        return W, mu, sd, X

    def predict_all(self):
        z, _, _, _ = self._feat()
        W, mu, sd, X = self._readout(z)
        return (X - mu) / sd @ W

    def _adam(self, name, p, g):
        m, v = self._st[name]
        m[:] = 0.9 * m + 0.1 * g; v[:] = 0.999 * v + 0.001 * g * g
        p -= self.lr * (m / (1 - 0.9 ** self._t)) / (np.sqrt(v / (1 - 0.999 ** self._t)) + 1e-8)

    def step(self):
        self._t += 1
        z, sig, dz_darg, pre = self._feat()
        W, mu, sd, X = self._readout(z)
        e = np.where(self.tr, (X - mu) / sd @ W - self.y, 0.0)
        n = self.tr.sum()
        base = (2.0 / n) * e[:, None] * (W[:self.Ho] / sd[0, :self.Ho])[None, :] * dz_darg
        gd = (base / sig).sum(0) + self.wd * self.d                    # arg = d/sigma
        dLdpre = base * (-self.d / sig ** 2) * self._sigmoid(pre)      # -> d(loss)/d(pre)
        gc = dLdpre.sum(0)
        grads = [("d", self.d, gd), ("c", self.c, gc)]
        if self.mix:
            grads.append(("M", self.M, dLdpre.T @ self.A + self.wd * self.M))
        else:
            grads.append(("g", self.g, (dLdpre * self.A[:, self.idx]).sum(0) + self.wd * self.g))
        for name, p, gr in grads:
            self._adam(name, p, gr * min(1.0, 5.0 / (np.linalg.norm(gr) + 1e-12)))

    def eval(self, te, epochs):
        for _ in range(epochs):
            self.step()
        return nrmse(self.y[te], self.predict_all()[te])
