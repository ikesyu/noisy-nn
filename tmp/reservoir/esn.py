"""Standard leaky-integrator ESN — the baseline that couples memory and
nonlinearity in ONE tanh reservoir (read-out only)."""
import numpy as np


class LeakyESN:
    def __init__(self, H=48, spectral_radius=0.95, leak=0.3, w_in_scale=1.0,
                 density=0.1, seed=0):
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((H, H)) * (rng.random((H, H)) < density)
        self.W = W * (spectral_radius / (np.max(np.abs(np.linalg.eigvals(W))) + 1e-12))
        self.w_in = w_in_scale * rng.choice([-1.0, 1.0], size=H)
        self.leak, self.H = leak, H

    def run(self, u):
        x = np.zeros(self.H); X = np.empty((len(u), self.H))
        for t in range(len(u)):
            x = (1 - self.leak) * x + self.leak * np.tanh(self.W @ x + self.w_in * u[t])
            X[t] = x
        return X
