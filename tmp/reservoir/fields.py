"""Noise-field designs — the MEMORY substrate of the separated architecture.

Ours = a noise field (memory) + an NNN crossing map (nonlinearity, learned
forward-only). This module holds the field designs compared in
docs/idea_reservoir.md §10.11-10.15. All fields are LINEAR (memory only; the
nonlinearity lives in the NNN map) and driven by the scalar input u only (fair
vs ESN). run(u) -> states X[T, H].

Dissipative fields (decay to zero without input, giving ESP / fading memory) with
a biological reading:
    CascadeField    signal propagation + decay (feedforward / synfire chain)
    DampedOrthField exponential decay of a rotation (neuromodulator clearance)
    LDNField        Legendre window memory == hippocampal/EC TIME CELLS (LMU)
    DiffusionField  leaky diffusion (volume transmission) -- thin memory
DelayLineField (explicit shift register) is the lossless reference; a random
LinearReservoir is the poor-design baseline.
"""
import numpy as np
from scipy.linalg import expm
from scipy.special import eval_legendre


class LinearReservoir:
    """Random sparse linear reservoir (the poor-design baseline: entangled)."""

    def __init__(self, H=48, spectral_radius=0.9, leak=0.1, w_in_scale=1.0,
                 density=0.1, seed=0):
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((H, H)) * (rng.random((H, H)) < density)
        W *= spectral_radius / (np.max(np.abs(np.linalg.eigvals(W))) + 1e-12)
        self.W, self.leak, self.H = W, leak, H
        self.w_in = w_in_scale * rng.choice([-1.0, 1.0], size=H)

    def run(self, u):
        x = np.zeros(self.H); X = np.empty((len(u), self.H))
        for t in range(len(u)):
            x = (1 - self.leak) * x + self.leak * (self.W @ x + self.w_in * u[t])
            X[t] = x
        return X


class DelayLineField:
    """Shift register x[t, i] = u[t-1-i]: explicit, disentangled memory of H lags."""

    def __init__(self, H=48, **_):
        self.H = H

    def run(self, u):
        T = len(u); X = np.zeros((T, self.H))
        for i in range(self.H):
            X[i + 1:, i] = u[:T - i - 1]
        return X


class CascadeField:
    """Leaky delay chain x_i(t)=a·x_{i-1}(t-1), x_0=u(t) -> a^i u(t-i).
    Dissipative (decays to zero); biology = signal propagation with decay."""

    def __init__(self, H=48, a=0.92, **_):
        self.H, self.a = H, a

    def run(self, u):
        x = np.zeros(self.H); X = np.empty((len(u), self.H))
        for t in range(len(u)):
            xn = np.empty(self.H); xn[0] = u[t]; xn[1:] = self.a * x[:-1]
            x = xn; X[t] = x
        return X


class DampedOrthField:
    """Orthogonal (rotational) field with rho<1: implicit but ENTANGLED memory,
    exponential decay. Biology = neuromodulator exponential clearance."""

    def __init__(self, H=48, rho=0.97, w_in_scale=1.0, seed=0, **_):
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((H, H)))
        self.W = rho * Q
        self.w_in = w_in_scale * rng.standard_normal(H)
        self.H = H

    def run(self, u):
        x = np.zeros(self.H); X = np.empty((len(u), self.H))
        for t in range(len(u)):
            x = self.W @ x + self.w_in * u[t]; X[t] = x
        return X


class LDNField:
    """Legendre Delay Network (Voelker 2019): optimal orthogonal window memory.
    Implicit yet DISENTANGLED long memory; dissipates over the window theta.
    Biology = TIME CELLS / working memory (LMU). ZOH-discretised for stability."""

    def __init__(self, H=48, theta=60.0, dt=1.0, **_):
        d = H
        A = np.zeros((d, d)); B = np.zeros(d)
        for i in range(d):
            B[i] = (2 * i + 1) * ((-1) ** i)
            for j in range(d):
                A[i, j] = (2 * i + 1) * (-1.0 if i < j else (-1.0) ** (i - j + 1))
        self.Ad = expm((dt / theta) * A)
        self.Bd = np.linalg.solve(A, (self.Ad - np.eye(d)) @ B)
        self.H, self.theta = d, theta

    def run(self, u):
        m = np.zeros(self.H); X = np.empty((len(u), self.H))
        for t in range(len(u)):
            m = self.Ad @ m + self.Bd * u[t]; X[t] = m
        return X

    def decode_weights(self, r):
        """Read-out weights that reconstruct the input at fractional delay r in
        [0,1] (r*theta steps ago) — a 'delay cell'. Used for the time-cell view."""
        return np.array([eval_legendre(i, 2 * r - 1) for i in range(self.H)])


class DiffusionField:
    """Leaky diffusion on a line (§5.2): dissipative but thin memory.
    Biology = volume transmission of a diffusing neuromodulator."""

    def __init__(self, H=48, D=0.6, gamma=0.02, dt=0.5, seed=0, **_):
        L = np.zeros((H, H))
        for i in range(H):
            deg = 0
            for j in (i - 1, i + 1):
                if 0 <= j < H:
                    L[i, j] = -1; deg += 1
            L[i, i] = deg
        self.M = np.eye(H) - dt * (D * L + gamma * np.eye(H))
        self.H = H
        self.w_in = np.random.default_rng(seed).choice([-1.0, 1.0], size=H)

    def run(self, u):
        x = np.zeros(self.H); X = np.empty((len(u), self.H))
        for t in range(len(u)):
            x = self.M @ x + self.w_in * u[t]; X[t] = x
        return X


def pulse_decay(field, T=200, pulse_at=1):
    """||field(t)|| after a single input pulse (normalised) — shows dissipation."""
    u = np.zeros(T); u[pulse_at] = 1.0
    norm = np.linalg.norm(field.run(u), axis=1)
    return norm / (norm.max() + 1e-12)
