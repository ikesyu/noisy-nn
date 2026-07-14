"""Noise generators and distribution functions (PDF / CDF) for the NNN.

Each `*_noise_like` factory returns a closure `gen(x)` that samples noise of
the same shape as `x`. Each `*_pdf_torch` / `*_cdf_torch` factory returns a
closure evaluating the distribution element-wise; a tensor-valued scale
parameter gives per-element distributions via broadcasting.
"""
import math
from typing import Callable, Union

import torch


def gaussian_noise_like(mu: torch.Tensor, sigma: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns a sampler for Gaussian noise N(mu, sigma^2)."""
    def gen(input: torch.Tensor) -> torch.Tensor:
        return (torch.randn_like(input) * sigma) + mu

    return gen


def stable_noise_like(alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor,
                      mean_zero: bool = True, epsilon: float = 1e-8) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns a sampler for the alpha-stable distribution S^0(alpha, beta, gamma, delta).

    Sampling uses the Chambers-Mallows-Stuck (CMS) method in the S^0
    parameterization. `alpha` in (0, 2], `beta` in [-1, 1], `gamma` > 0 are
    broadcastable tensors; `epsilon` groups alpha ≈ 1 into the Cauchy-type
    branch for numerical stability.

    The location delta is chosen per element as follows:
      - 1 < alpha <= 2: the mean exists; if `mean_zero`, delta is set to
        -beta * gamma * tan(pi * alpha / 2) so the mean is zero
        (alpha = 2 gives delta = 0).
      - alpha ≈ 1 (|alpha - 1| <= epsilon): Cauchy-type CMS formula; the mean
        does not exist, so delta = 0 (zero-centered).
      - alpha < 1: neither mean nor variance exists; delta = 0, and beta is
        forced to 0 (symmetric) so the mode/quantiles stay centered.
    """

    def _sample_like(shape, device, dtype):
        a = alpha.to(device=device, dtype=dtype).expand(shape)
        b = beta.to(device=device, dtype=dtype).expand(shape)
        g = gamma.to(device=device, dtype=dtype).expand(shape)
        # alpha < 1: force beta = 0 (element-wise) to keep the center at zero.
        mask_lt1 = a < (1.0 - epsilon)
        b = torch.where(mask_lt1, torch.zeros_like(b), b)

        # Random sources: U ~ Uniform(-pi/2, pi/2), W ~ Exp(1)
        U = (torch.rand(shape, device=device, dtype=dtype) - 0.5) * math.pi
        W = torch.distributions.Exponential(
            rate=torch.tensor(1.0, device=device, dtype=dtype)).sample(shape)

        X = torch.empty(shape, device=device, dtype=dtype)
        m1 = (torch.abs(a - 1.0) <= epsilon)    # alpha ≈ 1
        m2 = ~m1                                # alpha != 1

        # --- alpha != 1: general CMS formula for S^0 ---
        if m2.any():
            aa = a[m2]; bb = b[m2]; gg = g[m2]; u = U[m2]; w = W[m2]

            tan_term = torch.tan(math.pi * aa / 2)
            phi = (1.0 / aa) * torch.atan(bb * tan_term)
            S = (1 + (bb ** 2) * (tan_term ** 2)) ** (1.0 / (2.0 * aa))

            num = torch.sin(aa * (u + phi))
            den = (torch.cos(u)) ** (1.0 / aa)
            frac = (torch.cos(u - aa * (u + phi)) / w) ** ((1.0 - aa) / aa)

            Y = S * num / den * frac            # standardized S^0(alpha, beta, 1, 0)

            if mean_zero:
                # delta = -b g tan(pi a / 2) only where 1 < alpha < 2
                # (alpha = 2 gives tan(pi) = 0, so delta = 0 there anyway).
                delta = torch.zeros_like(Y)
                m_mean = (aa > 1.0) & (aa < 2.0)
                if m_mean.any():
                    delta[m_mean] = -bb[m_mean] * gg[m_mean] * torch.tan(math.pi * aa[m_mean] / 2.0)
                X[m2] = gg * Y + delta
            else:
                X[m2] = gg * Y

        # --- alpha ≈ 1: Cauchy-type CMS formula for S^0, delta = 0 ---
        if m1.any():
            bb = b[m1]; gg = g[m1]; u = U[m1]; w = W[m1]
            Y = (2.0 / math.pi) * (
                torch.tan(u) - bb * torch.log((math.pi / 2) * w * torch.cos(u) / ((math.pi / 2) + bb * u))
            )
            X[m1] = gg * Y

        return X

    def gen(x: torch.Tensor) -> torch.Tensor:
        return _sample_like(x.shape, x.device, x.dtype)

    return gen


def gaussian_pdf_torch(mu: float, sigma, epsilon: float = 1e-10) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns the PDF of N(mu, sigma^2).

    `sigma` may be a scalar or a tensor (element-wise, broadcastable).
    Where sigma <= epsilon the PDF is defined as zero.
    """
    def pdf(x) -> torch.Tensor:
        if torch.is_tensor(sigma):
            mask = sigma > epsilon
            return torch.where(
                mask,
                1 / (torch.sqrt(torch.tensor(2.0) * torch.pi) * sigma) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2),
                torch.tensor(0.0, device=x.device, dtype=x.dtype)
            )
        if sigma <= epsilon:
            return torch.zeros_like(x)
        return (1 / (torch.sqrt(torch.tensor(2.0) * torch.pi) * sigma) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2))

    return pdf


def gaussian_cdf_torch(mu: float, sigma: Union[float, torch.Tensor], epsilon: float = 1e-10) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns the CDF of N(mu, sigma^2).

    `sigma` may be a scalar or a tensor (element-wise, broadcastable).
    Where sigma <= epsilon the CDF is defined as zero.
    """
    def cdf(x: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(sigma):
            mask = sigma > epsilon
            return torch.where(
                mask,
                0.5 * (1 + torch.erf((x - mu) / (sigma * torch.sqrt(torch.tensor(2.0))))),
                torch.tensor(0.0, device=x.device, dtype=x.dtype)
            )
        if sigma <= epsilon:
            return torch.zeros_like(x)
        return 0.5 * (1 + torch.erf((x - mu) / (sigma * torch.sqrt(torch.tensor(2.0)))))

    return cdf


def stable_charfunc_S0(t: torch.Tensor, a: torch.Tensor, b: torch.Tensor, g: torch.Tensor,
                       d: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Characteristic function of the alpha-stable distribution (S^0 form).

        phi(t) = exp( i d t - |g t|^a * [1 - i b sign(t) omega(t, a)] )
        omega(t, a) = tan(pi a / 2)      if a != 1
                    = -(2/pi) log|t|     if a ≈ 1

    All parameters are broadcast to a common shape; returns a complex tensor.
    """
    tB, aB, bB, gB, dB = torch.broadcast_tensors(t, a, b, g, d)

    # omega(t, a), switching on a ≈ 1 and avoiding log(0)
    is_a1 = (torch.abs(aB - 1.0) <= epsilon)
    omega_a_ne_1 = torch.tan(math.pi * aB / 2.0)
    tiny = torch.finfo(tB.dtype).tiny
    omega_a_eq_1 = -(2.0 / math.pi) * torch.log(torch.clamp(torch.abs(tB), min=tiny))
    omega = torch.where(is_a1, omega_a_eq_1, omega_a_ne_1)

    gt_abs_a = (torch.abs(gB * tB)) ** aB
    sign_t = torch.sign(tB)

    real_part = -gt_abs_a
    imag_part = dB * tB + gt_abs_a * (bB * sign_t * omega)
    return torch.exp(real_part) * (torch.cos(imag_part) + 1j * torch.sin(imag_part))


def _trapz_last_dim(y: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Trapezoidal rule over the last dimension of `y` with constant spacing `dt`."""
    if y.size(-1) == 1:
        return y[..., 0] * dt
    y0 = y[..., 0]
    yN = y[..., -1]
    mid = y[..., 1:-1].sum(dim=-1)
    return dt * (0.5 * (y0 + yN) + mid)


def _prepare_params(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                    gamma: torch.Tensor, mean_zero: bool = True, epsilon: float = 1e-8):
    """Broadcasts alpha, beta, gamma to x.shape and computes delta.

    If `mean_zero` and 1 < alpha <= 2: delta = -beta * gamma * tan(pi * alpha / 2)
    (zero mean); otherwise delta = 0.
    """
    a = alpha.to(device=x.device, dtype=x.dtype).expand(x.shape)
    b = beta.to(device=x.device, dtype=x.dtype).expand(x.shape)
    g = gamma.to(device=x.device, dtype=x.dtype).expand(x.shape)

    d = torch.zeros_like(a)
    if mean_zero:
        m_mean = (a > 1.0 + epsilon)
        if m_mean.any():
            d[m_mean] = -b[m_mean] * g[m_mean] * torch.tan(math.pi * a[m_mean] / 2.0)

    return a, b, g, d


def stable_pdf_torch(alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor,
                     mean_zero: bool = True, n_grid: int = 4096, t_max: float = 50.0,
                     epsilon: float = 1e-8) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns the PDF of S^0(alpha, beta, gamma, delta) via Fourier inversion:

        f_X(x) = (1/pi) ∫_0^∞ Re[ e^{-i t x} phi(t) ] dt

    Increase `n_grid` and/or `t_max` for higher accuracy (slower); heavy tails
    or alpha ≈ 1 may require a larger `t_max`.
    """
    alpha_ = alpha
    beta_ = beta
    gamma_ = gamma

    def pdf(x: torch.Tensor) -> torch.Tensor:
        a, b, g, d = _prepare_params(x, alpha_, beta_, gamma_, mean_zero, epsilon)

        # quadrature grid t in (0, t_max]
        N = int(n_grid)
        t = torch.linspace(0.0, float(t_max), steps=N + 1, device=x.device, dtype=x.dtype)[1:]
        dt = t[1] - t[0] if t.numel() > 1 else torch.tensor(float(t_max), device=x.device, dtype=x.dtype)

        # broadcast to x.shape + (N,)
        t_view = t.view(*([1] * x.ndim), -1)
        x_view = x.unsqueeze(-1)
        aB = a.unsqueeze(-1); bB = b.unsqueeze(-1)
        gB = g.unsqueeze(-1); dB = d.unsqueeze(-1)

        phi = stable_charfunc_S0(t_view, aB, bB, gB, dB, epsilon=epsilon)
        phase = torch.cos(-t_view * x_view) + 1j * torch.sin(-t_view * x_view)

        integrand = torch.real(phase * phi)
        I = _trapz_last_dim(integrand, dt)
        return (1.0 / math.pi) * I

    return pdf


def stable_cdf_torch(alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor,
                     mean_zero: bool = True, n_grid: int = 4096, t_max: float = 50.0,
                     epsilon: float = 1e-8) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns the CDF of S^0(alpha, beta, gamma, delta) via Fourier inversion:

        F_X(x) = 1/2 - (1/pi) ∫_0^∞ Im[ e^{-i t x} phi(t) / t ] dt
    """
    alpha_ = alpha
    beta_ = beta
    gamma_ = gamma

    def cdf(x: torch.Tensor) -> torch.Tensor:
        a, b, g, d = _prepare_params(x, alpha_, beta_, gamma_, mean_zero, epsilon)

        # quadrature grid t in (0, t_max]
        N = int(n_grid)
        t = torch.linspace(0.0, float(t_max), steps=N + 1, device=x.device, dtype=x.dtype)[1:]
        dt = t[1] - t[0] if t.numel() > 1 else torch.tensor(float(t_max), device=x.device, dtype=x.dtype)

        # broadcast to x.shape + (N,)
        t_view = t.view(*([1] * x.ndim), -1)
        x_view = x.unsqueeze(-1)
        aB = a.unsqueeze(-1); bB = b.unsqueeze(-1)
        gB = g.unsqueeze(-1); dB = d.unsqueeze(-1)

        phi = stable_charfunc_S0(t_view, aB, bB, gB, dB, epsilon=epsilon)
        phase = torch.cos(-t_view * x_view) + 1j * torch.sin(-t_view * x_view)

        integrand = torch.imag(phase * phi) / t_view
        I = _trapz_last_dim(integrand, dt)
        return 0.5 - (1.0 / math.pi) * I

    return cdf


def gen_stdvec(dim: int, start: int, end: int, on_std: float = 0.5, off_std: float = 0.0) -> torch.Tensor:
    """Returns a [dim] std vector: `on_std` on [start:end], `off_std` elsewhere.

    Multiplying an [N, D] tensor by this [D] vector broadcasts along N, so it
    can switch per-unit noise on and off.
    """
    stdvec = torch.full((dim,), off_std)
    stdvec[start:end] = on_std
    return stdvec


def interpolate_stdvecs(stdvec1: torch.Tensor, stdvec2: torch.Tensor, rate: float = 0.5) -> torch.Tensor:
    """Linearly interpolates two std vectors: rate = 0 gives stdvec1, 1 gives stdvec2."""
    return (1.0 - rate) * stdvec1 + rate * stdvec2


def uniform_noise_like(center: torch.Tensor, radius: torch.Tensor,
                       epsilon: float = 1e-10) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns a sampler for bounded uniform noise Uniform(center - radius, center + radius).

    This noise family gives the exact parabolic analytic crossing response

        E[z] = 0.5 * [1 - ((d - center) / radius)^2]_+.

    Where radius <= epsilon the noise is deterministically zero.
    """
    def gen(input: torch.Tensor) -> torch.Tensor:
        c = torch.as_tensor(center, device=input.device, dtype=input.dtype)
        r = torch.as_tensor(radius, device=input.device, dtype=input.dtype)
        c, r = torch.broadcast_tensors(c, r)
        r_safe = torch.clamp(r, min=0.0)
        u = torch.rand_like(input) * 2.0 - 1.0
        sampled = c + r_safe * u
        return torch.where(r_safe > epsilon, sampled, torch.zeros_like(input))
    return gen


def uniform_pdf_torch(center: float, radius: Union[float, torch.Tensor],
                      epsilon: float = 1e-10) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns the PDF of Uniform(center - radius, center + radius)."""
    def pdf(x: torch.Tensor) -> torch.Tensor:
        c = torch.as_tensor(center, device=x.device, dtype=x.dtype)
        r = torch.as_tensor(radius, device=x.device, dtype=x.dtype)
        r = torch.clamp(r, min=0.0)
        inside = (torch.abs(x - c) < r) & (r > epsilon)
        val = 1.0 / (2.0 * torch.clamp(r, min=epsilon))
        return torch.where(inside, val, torch.zeros_like(x))
    return pdf


def uniform_cdf_torch(center: float, radius: Union[float, torch.Tensor],
                      epsilon: float = 1e-10) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns the CDF of Uniform(center - radius, center + radius)."""
    def cdf(x: torch.Tensor) -> torch.Tensor:
        c = torch.as_tensor(center, device=x.device, dtype=x.dtype)
        r = torch.as_tensor(radius, device=x.device, dtype=x.dtype)
        r_safe = torch.clamp(r, min=epsilon)
        raw = (x - c + r_safe) / (2.0 * r_safe)
        out = torch.clamp(raw, 0.0, 1.0)
        return torch.where(r > epsilon, out, torch.zeros_like(x))
    return cdf


def recruitment_to_radius(recruitment: Union[float, torch.Tensor], max_radius: float = 1.0,
                          min_radius: float = 0.0) -> Union[float, torch.Tensor]:
    """Maps a recruitment/noise-field intensity in [0, 1] to an activation radius.

    Zero means the unit is not recruited (inactive); larger values expand the
    local active region up to `max_radius`.
    """
    if torch.is_tensor(recruitment):
        return min_radius + (max_radius - min_radius) * torch.clamp(recruitment, 0.0, 1.0)
    return min_radius + (max_radius - min_radius) * max(0.0, min(1.0, float(recruitment)))


# =========================================================
# Simple check: run `python -m nnn.noise` from the repository root.
# =========================================================
if __name__ == "__main__":
    stdvec1 = gen_stdvec(10, 2, 5)
    print(stdvec1)
    stdvec2 = gen_stdvec(10, 4, 9)
    print(stdvec2)
    stdvec = interpolate_stdvecs(stdvec1, stdvec2)
    print(stdvec)

    alpha = torch.Tensor([[0.5, 1.0], [1.5, 2.0]])
    beta = torch.Tensor([[0, 0], [0, 0]])
    gamma = torch.Tensor([[1.0, 1.0], [1.0, 1.0]])
    x = torch.Tensor([[0, 0], [0, 0]])

    noise_gen = stable_noise_like(alpha, beta, gamma)
    print(noise_gen(x))
    pdf = stable_pdf_torch(alpha, beta, gamma)
    print(pdf(x))
    cdf = stable_cdf_torch(alpha, beta, gamma)
    print(cdf(x))
