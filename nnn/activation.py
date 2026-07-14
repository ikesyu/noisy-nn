"""Crossing activation functions of the Noise-modulated Neural Network (NNN).

All functions are stateless `torch.autograd.Function`s. Noise injection and
statistics are implemented as separate layers (see `nnn.layer`).

Reference: Ikemoto, Neurocomputing 2021.
"""
import torch

from . import noise


def _cyclic_xor(binary: torch.Tensor) -> torch.Tensor:
    """Cyclic XOR along dim 1: |b[t+1] - b[t]| with wrap-around at the end.

    Works for both [N, T] and [N, T, D] tensors of {0, 1} values.
    """
    xor = torch.abs(binary[:, 1:] - binary[:, :-1])
    last = torch.abs(binary[:, -1] - binary[:, 0])
    return torch.cat([xor, last.unsqueeze(1)], dim=1)


class Crossing(torch.autograd.Function):
    """Crossing activation (realtime version) for inputs of shape [N, T].

    Forward: binarize the input at thresholds +h and -h, XOR each along the
    T dimension, and average the two results.
    Backward: kernel density estimate of the local derivative, using a uniform
    kernel of width 2h — one sample per element.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, h: float = 0.05) -> torch.Tensor:
        h = abs(h)
        xor1 = _cyclic_xor((input > h).float())
        xor2 = _cyclic_xor((input > -h).float())
        ctx.save_for_backward(xor1, xor2)
        ctx.h = h
        return (xor2 + xor1) / 2

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        xor1, xor2 = ctx.saved_tensors
        coeff = (xor2 - xor1) / (2 * ctx.h)
        return coeff * grad_output, None


class CrossingSample(torch.autograd.Function):
    """Crossing activation (sampling version) for inputs of shape [N, T, D].

    Forward is identical to `Crossing` applied along the T dimension.
    Backward averages the kernel density estimate over the T samples and
    repeats it along T, giving one low-variance slope per unit.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, h: float = 0.05) -> torch.Tensor:
        h = abs(h)
        xor1 = _cyclic_xor((input > h).float())
        xor2 = _cyclic_xor((input > -h).float())
        ctx.save_for_backward(xor1, xor2)
        ctx.h = h
        return (xor2 + xor1) / 2

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        xor1, xor2 = ctx.saved_tensors
        T = xor1.size(1)
        coeff = torch.mean(xor2 - xor1, dim=1) / (2 * ctx.h)
        coeff = coeff.unsqueeze(1).repeat(1, T, 1)
        return coeff * grad_output, None


class CrossingStatistic(torch.autograd.Function):
    """Crossing activation (statistic version) for inputs of shape [N, D].

    Forward: expand the input `numT` times along a simulated T dimension, add
    noise, apply the crossing, and return the mean over T.
    Backward: kernel density estimate with a uniform kernel of width 2h.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, h: float = 0.05, numT: int = 100,
                noise_like: callable = noise.gaussian_noise_like(0.0, 1.0)) -> torch.Tensor:
        if h <= 0:
            raise ValueError("Threshold `h` must be positive.")
        if not callable(noise_like):
            raise TypeError("`noise_like` must be a callable function.")

        input = input.unsqueeze(1).repeat(1, numT, 1)
        input = input + noise_like(input)
        xor1 = torch.mean(_cyclic_xor((input > h).float()), dim=1)
        xor2 = torch.mean(_cyclic_xor((input > -h).float()), dim=1)
        ctx.save_for_backward(xor1, xor2)
        ctx.h = h
        return (xor2 + xor1) / 2.0

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        xor1, xor2 = ctx.saved_tensors
        coeff = (xor2 - xor1) / (2 * ctx.h)
        return coeff * grad_output, None, None, None


class CrossingAnalytic(torch.autograd.Function):
    """Crossing activation (analytic version) for inputs of shape [N, D].

    Uses the PDF and CDF of the noise distribution to compute the expected
    response and its derivative in closed form:

        E[z] = 2 P (1 - P),    dE[z]/dd = 2 (1 - 2 P) p,    P = CDF(d).
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor,
                noisePDF: callable = noise.gaussian_pdf_torch(0, 1.0),
                noiseCDF: callable = noise.gaussian_cdf_torch(0, 1.0)) -> torch.Tensor:
        if not callable(noisePDF) or not callable(noiseCDF):
            raise TypeError("`noisePDF` and `noiseCDF` must be callable functions.")

        P = noiseCDF(input)
        p = noisePDF(input)
        ctx.save_for_backward(P, p)
        return 2.0 * P * (1 - P)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        P, p = ctx.saved_tensors
        coeff = (1 - 2 * P) * p
        return coeff * grad_output, None, None


class ParabolicCrossingAnalytic(torch.autograd.Function):
    """Analytic crossing response for bounded uniform noise.

    For eta ~ Uniform(center - radius, center + radius):

        y = 0.5 * [1 - ((x - center) / radius)^2]_+.

    The radius acts as a recruitment/noise-strength parameter; radius = 0
    makes both the output and the derivative exactly zero.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, center=0.0, radius=1.0,
                epsilon: float = 1e-10) -> torch.Tensor:
        c = torch.as_tensor(center, device=input.device, dtype=input.dtype)
        r = torch.as_tensor(radius, device=input.device, dtype=input.dtype)
        c, r = torch.broadcast_tensors(c, r)
        r = torch.clamp(r, min=0.0)
        r_safe = torch.clamp(r, min=epsilon)

        u = (input - c) / r_safe
        active = (torch.abs(u) < 1.0) & (r > epsilon)
        output = 0.5 * (1.0 - u * u)
        output = torch.where(active, output, torch.zeros_like(input))

        ctx.save_for_backward(input, c, r, active)
        ctx.epsilon = epsilon
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, c, r, active = ctx.saved_tensors
        r_safe = torch.clamp(r, min=ctx.epsilon)
        coeff = -(input - c) / (r_safe * r_safe)
        coeff = torch.where(active, coeff, torch.zeros_like(input))
        return coeff * grad_output, None, None, None


class HatApproxCrossingAnalytic(torch.autograd.Function):
    """Piecewise-linear (hat) approximation of the parabolic response.

    Normalized mode:  y = 0.5 * [1 - |x - center| / radius]_+
    Coupled mode:     y = [radius - |x - center|]_+

    The normalized mode matches the peak scale of the parabolic response.
    The coupled mode needs no division and also vanishes as the recruitment
    radius goes to zero, which suits hardware implementations.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, center=0.0, radius=1.0,
                normalized: bool = True, epsilon: float = 1e-10) -> torch.Tensor:
        c = torch.as_tensor(center, device=input.device, dtype=input.dtype)
        r = torch.as_tensor(radius, device=input.device, dtype=input.dtype)
        c, r = torch.broadcast_tensors(c, r)
        r = torch.clamp(r, min=0.0)
        r_safe = torch.clamp(r, min=epsilon)

        diff = input - c
        abs_diff = torch.abs(diff)
        active = (abs_diff < r) & (r > epsilon)

        if normalized:
            output = 0.5 * (1.0 - abs_diff / r_safe)
        else:
            output = r - abs_diff
        output = torch.where(active, output, torch.zeros_like(input))

        ctx.save_for_backward(diff, r, active)
        ctx.normalized = normalized
        ctx.epsilon = epsilon
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        diff, r, active = ctx.saved_tensors
        sign = torch.sign(diff)
        if ctx.normalized:
            r_safe = torch.clamp(r, min=ctx.epsilon)
            coeff = -0.5 * sign / r_safe
        else:
            coeff = -sign
        coeff = torch.where(active, coeff, torch.zeros_like(diff))
        return coeff * grad_output, None, None, None, None


# =========================================================
# Simple check: run `python -m nnn.activation` from the repository root.
#
# Note: CrossingStatistic approaches CrossingAnalytic as numT is increased.
# The forward matches well, but the backward keeps a visible gap due to the
# kernel density estimation error of the uniform kernel.
# =========================================================
if __name__ == "__main__":
    print("Crossing")
    N, T = 2, 15
    h = 0.5
    input = torch.randn(N, T, requires_grad=True)
    output = Crossing.apply(input, h)
    print("Forward: ", output)
    output.backward(torch.ones_like(output))
    print("Backward: ", input.grad)

    print("CrossingSample")
    N, T, D = 2, 5, 3
    h = 0.5
    input = torch.randn(N, T, D, requires_grad=True)
    output = CrossingSample.apply(input, h)
    print("Forward: ", output)
    output.backward(torch.ones_like(output))
    print("Backward: ", input.grad)

    print("CrossingStatistic")
    input = torch.randn(N, D, requires_grad=True)
    print("Input:", input)
    output = CrossingStatistic.apply(input, 0.05, 10000,
                                     noise.gaussian_noise_like(0.0, 1.0))
    print("Forward: ", output)
    output.backward(torch.ones_like(output))
    print("Backward: ", input.grad)

    print("CrossingAnalytic")
    print("Input:", input)
    output = CrossingAnalytic.apply(input, noise.gaussian_pdf_torch(0.0, 1.0),
                                    noise.gaussian_cdf_torch(0.0, 1.0))
    print("Forward: ", output)
    output.backward(torch.ones_like(output))
    print("Backward: ", input.grad)
