"""Prior-art baselines for the fairness comparison (docs/idea_reservoir.md §13.3).

NG-RC (Gauthier et al., Nat. Commun. 2021): a delay-line linear memory plus
EXPLICIT degree-2 polynomial features, read out by ridge -- NO trained features.
Structurally the closest prior art to (B)-mix + delay field, but its coupling is
ADDITIVE / input-path (the lagged inputs and their products enter the read-out
directly), whereas (B) couples the field MULTIPLICATIVELY as each unit's noise
scale. LMU (Voelker 2019) is covered by the (A) LearnedCrossingMap on an LDN
field (LDN memory + a learned nonlinear map + linear read-out, forward-only) --
also additive/input-path.

`budget_*` give the trainable-parameter count and total state/feature dimension
so the methods can be placed on a common "error vs budget" axis.
"""
import numpy as np


class NGRC:
    """Next-Generation Reservoir Computing (Gauthier et al. 2021).

    features(u) -> O[T, F] = [1 | lagged inputs | unique degree-2 products].
    delay = number of lags (incl. lag 0); degree 2 adds all u_i u_j (i<=j)."""

    def __init__(self, delay=20, degree=2, stride=1):
        if degree not in (1, 2):
            raise ValueError("degree must be 1 or 2")
        self.delay, self.degree, self.stride = delay, degree, stride

    def features(self, u):
        T, d, s = len(u), self.delay, self.stride
        lin = np.zeros((T, d))
        for i in range(d):
            L = i * s                                    # lag i*stride
            if L == 0:
                lin[:, i] = u
            else:
                lin[L:, i] = u[:T - L]                   # u(t - i*stride)
        parts = [np.ones((T, 1)), lin]
        if self.degree == 2:
            iu = np.triu_indices(d)                      # i<=j incl. squares
            parts.append(lin[:, iu[0]] * lin[:, iu[1]])
        return np.concatenate(parts, axis=1)

    def feature_dim(self):
        d = self.delay
        return 1 + d + (d * (d + 1) // 2 if self.degree == 2 else 0)


def budget_ngrc(delay, degree=2):
    F = NGRC(delay, degree).feature_dim()
    return {"trainable": F + 1, "state": F, "feature": F, "label": f"d={delay}"}


def budget_esn(H):
    return {"trainable": H + 1, "state": H, "feature": H, "label": f"H={H}"}


def budget_lmu(Ho, Hf):
    # W1 (Ho x Hf) + b (Ho) + read-out (Ho + 1)
    return {"trainable": Ho * Hf + 2 * Ho + 1, "state": Hf + Ho,
            "feature": Ho, "label": f"Ho={Ho}"}


def budget_ours(Ho, Hf):
    # M (Ho x Hf) + c (Ho) + d (Ho) + read-out (Ho + 1)
    return {"trainable": Ho * Hf + 3 * Ho + 1, "state": Hf + Ho,
            "feature": Ho, "label": f"Ho={Ho}"}
