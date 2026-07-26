"""Benchmark tasks.

narma_x: a NARMA family whose memory ORDER x is the only thing that changes. The
memory term is mean-normalised so the dynamics stays bounded and non-degenerate
across x (the classic sum-form NARMA saturates / destabilises at large x). It is
a nonlinear recurrence of order x (depends on y's own past and on u(t-x)u(t-1)),
so it needs BOTH long memory and nonlinearity.
"""
import numpy as np


def narma_x(T, x, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=T); y = np.zeros(T)
    for t in range(x, T):
        y[t] = np.tanh(0.3 * y[t - 1] + 0.5 * y[t - 1] * np.mean(y[t - x:t])
                       + 1.5 * u[t - x] * u[t - 1] + 0.1)
    return u, y


def mc_input(T, seed=0, low=-1.0, high=1.0):
    """i.i.d. uniform drive for the memory-capacity measurement."""
    return np.random.default_rng(seed).uniform(low, high, size=T)
