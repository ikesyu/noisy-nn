"""Credit estimators built from the forward noise of an NNN.

These recover, from forward statistics alone, quantities that backpropagation
takes from a backward pass. `covariance_credit` regresses the loss on a unit's
own fluctuation; `cov_weight` recovers a forward weight matrix from the
covariance between a layer's activity and the next layer's pre-activation (a
weight mirror), which removes the need for a transposed-weight read. `ManualOpt`
applies the resulting gradients and keeps its state per weight, so it adds no
weight transport of its own.
"""
import math

import torch

EPS = 1e-6


def covariance_credit(z_l, L, credit, eps: float = EPS):
    """Credit g_z ~= dL/dz of each unit: the loss regressed on its fluctuation.

    `z_l` is [N, T, H] and `L` is the per-sample loss [N, T].

    credit == "pooled"   : one coefficient per unit, centred over N and T.
    credit == "per_input": centred over T only, so the between-input variation of
        the loss cannot confound the regression. This is the far less biased,
        input-dependent estimate of the local gradient.

    Returns (g_broadcast, g_unit): `g_broadcast` broadcasts against [N, T, H],
    and `g_unit` is a [H] per-unit summary.
    """
    if credit == "pooled":
        cz = z_l - z_l.mean(dim=(0, 1), keepdim=True)
        cL = (L - L.mean()).unsqueeze(-1)                       # [N, T, 1]
        cov = (cL * cz).mean(dim=(0, 1))                        # [H]
        var = (cz ** 2).mean(dim=(0, 1))                        # [H]
        g_z = cov / (var + eps)                                 # [H]
        return g_z.view(1, 1, -1), g_z
    cz = z_l - z_l.mean(dim=1, keepdim=True)                    # [N, T, H]
    cL = (L - L.mean(dim=1, keepdim=True)).unsqueeze(-1)        # [N, T, 1]
    cov = (cL * cz).mean(dim=1)                                 # [N, H]
    var = (cz ** 2).mean(dim=1)                                 # [N, H]
    g_z = cov / (var + eps)                                     # [N, H]
    return g_z.unsqueeze(1), g_z.mean(dim=0)


def cov_weight(d_next, z_prev, pool: bool = False, eps: float = EPS):
    """Weight mirror: recover the forward W of d_next = W z_prev + b.

    The pre-activation is linear in the previous activity, so at a fixed input
    the single-variable regression coefficient of d_next_j on z_prev_i equals
    W_ji whenever the z_prev_i fluctuate independently, which the per-unit
    crossing noise ensures. The estimate is therefore taken within each input,
    over the T samples, and averaged over inputs:

        W_hat[j, i] = mean_n Cov_t(d_next_j, z_prev_i) / Var_t(z_prev_i)

    Correlate with the continuous pre-activation `d_next`, never with a binary
    activation, whose pathwise response is degenerate. `pool` instead sums
    numerator and denominator over inputs before dividing once: W is the same
    for every input, unlike a gradient, so this spends all N*T samples on one
    estimate and has far lower variance.

    `d_next` is [N, T, Ho] and `z_prev` is [N, T, Hi]; returns [Ho, Hi].
    """
    cd = d_next - d_next.mean(dim=1, keepdim=True)                # [N, T, Ho]
    cz = z_prev - z_prev.mean(dim=1, keepdim=True)                # [N, T, Hi]
    cov = torch.einsum("nto,nti->noi", cd, cz) / d_next.shape[1]  # [N, Ho, Hi]
    var = (cz ** 2).mean(dim=1)                                   # [N, Hi]
    if pool:
        return cov.sum(dim=0) / (var.sum(dim=0).unsqueeze(0) + eps)
    W = cov / (var.unsqueeze(1) + eps)                            # [N, Ho, Hi]
    return W.mean(dim=0)


class ManualOpt:
    """In-place SGD / Adam for manually computed gradients.

    Adam changes only how the step is taken, not the credit behind it, and its
    moments are closed over each weight, so no weight transport appears.
    """

    def __init__(self, kind: str, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.kind = kind
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m, self.v, self.step = {}, {}, {}

    def update(self, key: str, param, grad, lr: float):
        """Apply the step in place and return the decrement applied, so a caller
        can integrate the same known increment into its weight mirrors
        (Kolen-Pollack tracking)."""
        if self.kind == "sgd":
            step = lr * grad
            param.data -= step
            return step
        if key not in self.m:
            self.m[key] = torch.zeros_like(grad)
            self.v[key] = torch.zeros_like(grad)
            self.step[key] = 0
        self.step[key] += 1
        m, v = self.m[key], self.v[key]
        m.mul_(self.b1).add_(grad, alpha=1.0 - self.b1)
        v.mul_(self.b2).addcmul_(grad, grad, value=1.0 - self.b2)
        m_hat = m / (1.0 - self.b1 ** self.step[key])
        v_hat = v / (1.0 - self.b2 ** self.step[key])
        step = lr * m_hat / (v_hat.sqrt() + self.eps)
        param.data -= step
        return step


def lr_at(lr0: float, epoch: int, epochs: int, decay: str) -> float:
    """Learning-rate schedule, shrinking the end-stage stochastic noise ball.

    "cosine", "exp" (down to 1% of `lr0`), or constant for any other value.
    """
    if decay == "cosine":
        return lr0 * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, epochs)))
    if decay == "exp":
        return lr0 * (0.01 ** (epoch / max(1, epochs)))
    return lr0
