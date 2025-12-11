import torch
import math
from typing import Callable, Union


def gaussian_noise_like(mu: torch.Tensor, sigma: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """ Creates a function that generates Gaussian noise with a specified mean and standard deviation.

    The returned function takes an input tensor and generates Gaussian noise of the same shape.

    Args:
        mu (torch.Tensor): The mean of the Gaussian noise.
        sigma (torch.Tensor): The standard deviation of the Gaussian noise.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: A function that takes a tensor as input 
        and returns a tensor of the same shape with Gaussian noise applied.
    """
    def gen(input: torch.Tensor) -> torch.Tensor:
        return (torch.randn_like(input) * sigma) + mu
    
    return gen


def stable_noise_like(alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, mean_zero: bool = True, epsilon: float = 1e-8) -> Callable[[torch.Tensor], torch.Tensor]:
    """ Creates a function that generates noise that complies Alpha-Stable distribution S^0(alpha, beta, gamma, -beta gamma tan((pi alpha)/2​)).
    
    The returned function takes an input tensor and generates noise of the same shape.
    
    Args:
        alpha (torch.Tensor): (0, 2]
        beta (torch.Tensor): [-1, 1]
        gamma (torch.Tensor): (0, infty)
        epsilon (float): Approx [1e-8, 1e-6] (smaller: finer/unstable, larger: rougher/stable)
    
    Returns:
        Callable[[torch.Tensor], torch.Tensor]:

    - S^0 parameterization is employed
    - alpha, beta, gamma: broadcastable tensors
   
    (A) alpha < 1:
    - The expected value (mean) does not exist.
    - In this implementation, the center of the distribution is set to 0 (delta=0).
    - When asymmetric (beta≠0), even with zero-centered data, the quantiles and mode shift. 
      Therefore, to prioritize consistency in practical applications, we fix beta=0 (symmetric).

    (B) alpha ≈ 1（|alpha-1|<=epsilon）:
    - Sampling is performed using the special formula for α=1 (the Cauchy-type formula in CMS).
    - Since the mean is undefined, delta=0 (zero-centered symmetric/asymmetric).
    - To ensure numerical stability, we group the neighborhood using epsilon and include it in this branch.

    (C) 1 < alpha <= 2:
    - An expected value (mean) exists.
    - delta = -beta * gamma * tan(pi*alpha/2)
      （if α=2, tan(pi)=0 leads delta=0）
    - Sample using the CMS standard formula, then apply scale+shift at the end.

    Note) When alpha == 2, it follows a normal distribution and beta is meaningless.
          When alpha < 2, the mean exists but the variance may diverge.
          When alpha == 1, it follows a Cauchy distribution; the mean does not exist, but there is a sort of central tendency.
          When alpha < 1, it becomes a fat-tailed distribution where both the mean and variance diverge. It can be zero.
    """

    def _sample_like(shape, device, dtype):
        # Broadcast params to the target shape
        a = alpha.to(device=device, dtype=dtype).expand(shape)
        b = beta.to(device=device, dtype=dtype).expand(shape)
        g = gamma.to(device=device, dtype=dtype).expand(shape)
        # if it is near to α<1, fix β to 0 (element-wise).
        mask_lt1 = a < (1.0 - epsilon)
        b = torch.where(mask_lt1, torch.zeros_like(b), b)

        # Random sources
        # U ~ Uniform(-pi/2, pi/2), W ~ Exp(1)
        U = (torch.rand(shape, device=device, dtype=dtype) - 0.5) * math.pi
        W = torch.distributions.Exponential(rate=torch.tensor(1.0, device=device, dtype=dtype)).sample(shape)

        # Output
        X = torch.empty(shape, device=device, dtype=dtype)

        # Masks
        m1 = (torch.abs(a - 1.0) <= epsilon)  # alpha ≈ 1
        m2 = ~m1                                # alpha != 1

        # --- alpha != 1 (general CMS for S^0) ---
        if m2.any():
            aa = a[m2]; bb = b[m2]; gg = g[m2]; u = U[m2]; w = W[m2]

            # Precompute constants that depend on alpha (elementwise!)
            tan_term = torch.tan(math.pi * aa / 2)
            phi = (1.0/aa) * torch.atan(bb * tan_term) # phi(a,b)
            S = (1 + (bb**2) * (tan_term**2))**(1.0/(2.0*aa))

            num = torch.sin(aa * (u + phi))
            den = (torch.cos(u))**(1.0/aa)
            frac = (torch.cos(u - aa*(u + phi)) / w)**((1.0 - aa)/aa)

            Y = S * num / den * frac  # standardized S^0(alpha,beta,1,0)

            if mean_zero:
                # mean-zero delta (only meaningful for 1<alpha<=2)
                # implement piecewise for numerical sense:
                # if aa==2 -> tan(pi)=0, delta=0; if aa in (1,2) -> usual formula
                delta = torch.zeros_like(Y)
                # mask where 1<alpha<2
                m_mean = (aa > 1.0) & (aa < 2.0)
                if m_mean.any():
                    delta[m_mean] = -bb[m_mean] * gg[m_mean] * torch.tan(math.pi * aa[m_mean] / 2.0)
                # aa==2 → delta=0 already
                X[m2] = gg * Y + delta
            else:
                X[m2] = gg * Y  # delta = 0. the location is not shifted.
            
        # --- alpha == 1 (Cauchy-type, S^0 CMS form) ---
        if m1.any():
            bb = b[m1]; gg = g[m1]; u = U[m1]; w = W[m1]
            # CMS for alpha=1 in S^0
            # Note: mean does not exist; use delta=0 (zero-centered symmetry) if needed
            # Y ~ S^0(1,beta,1,0):
            Y = (2.0/math.pi) * (
                torch.tan((math.pi/2) * (torch.ones_like(bb) + (2.0/math.pi)*bb*u))  # equivalent safe form
            )
            # A numerically stable canonical form is:
            # Y = (2/pi) * ( (pi/2 + bb*u).tan() - bb * torch.log((pi/2)*w*torch.cos(u)/(pi/2 + bb*u)) )
            # Using the "classic" CMS alpha=1 form:
            Y = (2.0/math.pi) * (
                torch.tan(u) - bb * torch.log((math.pi/2)*w*torch.cos(u)/( (math.pi/2) + bb*u ))
            )

            X[m1] = gg * Y  # delta=0

        return X

    def gen(x: torch.Tensor) -> torch.Tensor:
        return _sample_like(x.shape, x.device, x.dtype)

    return gen


def gaussian_pdf_torch(mu: float, sigma, epsilon: float = 1e-10) -> Callable[[torch.Tensor], torch.Tensor]:
    """ Returns the probability density function (PDF) of Gaussian distribution.

    This function returns a closure that computes the Gaussian PDF for a given input `x`.
    It supports both scalar and tensor inputs for `sigma`, handling cases where `sigma`
    is below a specified threshold (`epsilon`) to avoid division by zero.

    Args:
        mu (float): The mean of the Gaussian distribution.
        sigma (float or torch.Tensor): The standard deviation(s) of the Gaussian distribution.
            If `sigma` is a tensor, the computation is element-wise.
        epsilon (float, optional): A small threshold to avoid division by zero.
            If `sigma` is below this value, the function returns zero. Default is `1e-10`.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: A function that computes the Gaussian PDF
        for a given input tensor `x`.

    Example:
        >>> pdf = gaussian_pdf_torch(mu=0.0, sigma=1.0)
        >>> x = torch.tensor([0.0, 1.0, 2.0])
        >>> pdf(x)
        tensor([0.3989, 0.2419, 0.0539])

        >>> sigma_tensor = torch.tensor([1.0, 0.5, 0.1])
        >>> pdf = gaussian_pdf_torch(mu=0.0, sigma=sigma_tensor)
        >>> x = torch.tensor([0.0, 1.0, 2.0])
        >>> pdf(x)
        tensor([0.3989, 0.1079, 0.0000])
    """
    def pdf(x) -> torch.Tensor:
        # Check if sigma is a tensor
        if torch.is_tensor(sigma):
            # Apply thresholding for numerical stability
            mask = sigma > epsilon
            result = torch.where(
                mask,
                1 / (torch.sqrt(torch.tensor(2.0) * torch.pi) * sigma) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2),
                torch.tensor(0.0, device=x.device, dtype=x.dtype)
            )
            return result
        else:
            # Handle scalar sigma case
            if sigma <= epsilon:
                return torch.zeros_like(x)
            else:
                return (1 / (torch.sqrt(torch.tensor(2.0) * torch.pi) * sigma) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2))

    return pdf


def gaussian_cdf_torch(mu: float, sigma: Union[float, torch.Tensor], epsilon: float = 1e-10) -> Callable[[torch.Tensor], torch.Tensor]:
    """ Returns the cumulative density function (CDF) of Gaussian distribution.

    This function returns another function (`cdf`), which computes the cumulative density function for a given input tensor `x`.
    If `sigma` is a tensor, the function evaluates the CDF element-wise while ensuring numerical stability by applying a threshold (`epsilon`).
    If `sigma` is a scalar and below `epsilon`, the output is set to zero.

    Args:
        mu (float): The mean of the Gaussian distribution.
        sigma (Union[float, torch.Tensor]): The standard deviation of the Gaussian distribution, which can be a scalar or a tensor.
        epsilon (float, optional): A small threshold to prevent division by zero. Defaults to `1e-10`.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: A function that computes the CDF of a Gaussian distribution for a given input tensor.

    Example:
        >>> mu, sigma = 0.0, 1.0
        >>> cdf = gaussian_cdf_torch(mu, sigma)
        >>> x = torch.tensor([0.0, 1.0, 2.0])
        >>> cdf(x)
        tensor([0.5000, 0.8413, 0.9772])
    """
    def cdf(x: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(sigma):
            mask = sigma > epsilon
            result = torch.where(
                mask,
                0.5 * (1 + torch.erf((x - mu) / (sigma * torch.sqrt(torch.tensor(2.0))))),
                torch.tensor(0.0, device=x.device, dtype=x.dtype)
            )
            return result
        else:
            if sigma <= epsilon:
                return torch.zeros_like(x)
            else:
                return 0.5 * (1 + torch.erf((x - mu) / (sigma * torch.sqrt(torch.tensor(2.0)))))

    return cdf


def stable_charfunc_S0(t: torch.Tensor, a: torch.Tensor, b: torch.Tensor, g: torch.Tensor, d: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """ Characteristic function of the alpha-stable distribution
    S^0 parameterization characteristic function φ(t), fully broadcasted.
      φ(t) = exp( i d t - |g t|^a * [ 1 - i b sign(t) ω(t,a) ] )
    where
      ω(t,a) = tan(π a / 2)              if a != 1
             = -(2/π) log|t|             if a == 1

    Args:
        a: alpha
        b: beta
        g: gamma
        d: delta
        epsilon:
        *) All inputs can be broadcastable; this function broadcasts them to a common shape.

    Returns: 
        Complex tensor with the broadcasted shape.
    """

    # 1) broadcast all parameters
    tB, aB, bB, gB, dB = torch.broadcast_tensors(t, a, b, g, d)

    # 2) safely create ω(t,a) by `where`.
    is_a1 = (torch.abs(aB - 1.0) <= epsilon)

    # tan(π a / 2)（for a≠1）
    omega_a_ne_1 = torch.tan(math.pi * aB / 2.0)

    # -(2/π) log|t|（for a≈1）… avoid log(0)
    tiny = torch.finfo(tB.dtype).tiny
    omega_a_eq_1 = -(2.0 / math.pi) * torch.log(torch.clamp(torch.abs(tB), min=tiny))

    omega = torch.where(is_a1, omega_a_eq_1, omega_a_ne_1)

    # 3) compute φ(t)
    gt_abs_a = (torch.abs(gB * tB))**aB
    sign_t   = torch.sign(tB)

    real_part = -gt_abs_a
    imag_part = dB * tB + gt_abs_a * (bB * sign_t * omega)

    # e^(real + i imag)
    return torch.exp(real_part) * (torch.cos(imag_part) + 1j * torch.sin(imag_part))


def _trapz_last_dim(y: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """ Applies trapezoidal rule over the last dimension of y, with constant spacing dt (scalar tensor).

    Args:
        y: Integrand
        dt: Time step for numerical integration
    
    Returns:
        Integrated function
    """
    if y.size(-1) == 1:
        return y[..., 0] * dt
    y0 = y[..., 0]
    yN = y[..., -1]
    mid = y[..., 1:-1].sum(dim=-1)
    return dt * (0.5 * (y0 + yN) + mid)


def _prepare_params(x: torch.Tensor, alpha: torch.Tensor, beta:  torch.Tensor, gamma: torch.Tensor, mean_zero: bool = True, epsilon: float = 1e-8):
    """ Broadcasts α,β,γ to x.shape and compute δ.

    Args:
        x: A tensor which has reference shape.
        alpha:
        beta:
        gamma:
        mean_zero: Set True to adjust delta to keep the average to zero
        epsilon:

    Returns:
        Broadcasted alpha, beta, gamma, and delta

    If 1<α<=2: δ = -β γ tan(π α / 2), else δ = 0.
    """
    a = alpha.to(device=x.device, dtype=x.dtype).expand(x.shape)
    b = beta.to(device=x.device, dtype=x.dtype).expand(x.shape)
    g = gamma.to(device=x.device, dtype=x.dtype).expand(x.shape)

    # decide the delta to configure the mean/center at zero.
    d = torch.zeros_like(a)
    if mean_zero:
        m_mean = (a > 1.0 + epsilon)
        if m_mean.any():
            d[m_mean] = -b[m_mean] * g[m_mean] * torch.tan(math.pi * a[m_mean] / 2.0)
        # if alpha is similar to 2, tan(pi)=0 holds and let delta zero.
    
    return a, b, g, d


def stable_pdf_torch(alpha: torch.Tensor, beta:  torch.Tensor, gamma: torch.Tensor, mean_zero: bool = True, n_grid: int = 4096, t_max: float = 50.0, epsilon: float = 1e-8) -> Callable[[torch.Tensor], torch.Tensor]:
    """ Creates a PDF function for S^0(alpha, beta, gamma, balanced delta) via Fourier inversion
      f_X(x) = (1/π) ∫_0^∞ Re[ e^{-i t x} φ(t) ] dt
    Returns a function pdf(x) -> tensor same shape as x.
    
    Args:
        alpha:
        beta:
        gamma:
        n_grid:
        t_max:
        epsilon:

    Returns:
        Probabilistic Distribution Function of Stable Distribution
    
    Tips:
      - Increase n_grid and/or t_max for higher accuracy (slower).
      - Heavy tails or α≈1 may require larger t_max.
    """
    # cache params (kept as-is; moved/broadcast in call)
    alpha_ = alpha
    beta_  = beta
    gamma_ = gamma

    def pdf(x: torch.Tensor) -> torch.Tensor:
        a, b, g, d = _prepare_params(x, alpha_, beta_, gamma_, mean_zero, epsilon)

        # Quadrature grid t ∈ (0, t_max]
        N = int(n_grid)
        t = torch.linspace(0.0, float(t_max), steps=N+1, device=x.device, dtype=x.dtype)[1:]  # (N,)
        dt = t[1] - t[0] if t.numel() > 1 else torch.tensor(float(t_max), device=x.device, dtype=x.dtype)

        # broadcast to x.shape + (N,)
        t_view = t.view(*([1]*x.ndim), -1)
        x_view = x.unsqueeze(-1)
        aB = a.unsqueeze(-1); bB = b.unsqueeze(-1)
        gB = g.unsqueeze(-1); dB = d.unsqueeze(-1)

        phi = stable_charfunc_S0(t_view, aB, bB, gB, dB, epsilon=epsilon)   # complex
        phase = torch.cos(-t_view * x_view) + 1j * torch.sin(-t_view * x_view)

        integrand = torch.real(phase * phi)   # x.shape + (N,)
        I = _trapz_last_dim(integrand, dt)    # x.shape
        return (1.0 / math.pi) * I

    return pdf


def stable_cdf_torch(alpha: torch.Tensor, beta:  torch.Tensor, gamma: torch.Tensor, mean_zero: bool = True, n_grid: int = 4096, t_max: float = 50.0, epsilon: float = 1e-8) -> Callable[[torch.Tensor], torch.Tensor]:
    """ Creates a CDF function for S^0(alpha, beta, gamma, balanced delta) via Fourier inversion:
      F_X(x) = 1/2 - (1/π) ∫_0^∞ Im[ e^{-i t x} φ(t) / t ] dt
    Returns a function cdf(x) -> tensor same shape as x.
    
    Args:
        alpha:
        beta:
        gamma:
        n_grid:
        t_max:
        epsilon:

    Returns:
        Cumulative Distribution Function of Stable Distribution
    
    """
    alpha_ = alpha
    beta_  = beta
    gamma_ = gamma

    def cdf(x: torch.Tensor) -> torch.Tensor:
        a, b, g, d = _prepare_params(x, alpha_, beta_, gamma_, mean_zero, epsilon)

        # Quadrature grid t ∈ (0, t_max]
        N = int(n_grid)
        t = torch.linspace(0.0, float(t_max), steps=N+1, device=x.device, dtype=x.dtype)[1:]  # (N,)
        dt = t[1] - t[0] if t.numel() > 1 else torch.tensor(float(t_max), device=x.device, dtype=x.dtype)

        # broadcast to x.shape + (N,)
        t_view = t.view(*([1]*x.ndim), -1)
        x_view = x.unsqueeze(-1)
        aB = a.unsqueeze(-1); bB = b.unsqueeze(-1)
        gB = g.unsqueeze(-1); dB = d.unsqueeze(-1)

        phi = stable_charfunc_S0(t_view, aB, bB, gB, dB, epsilon=epsilon)   # complex
        phase = torch.cos(-t_view * x_view) + 1j * torch.sin(-t_view * x_view)

        # Im[ phase * phi / t ]
        integrand = torch.imag(phase * phi) / t_view
        I = _trapz_last_dim(integrand, dt)
        return 0.5 - (1.0 / math.pi) * I

    return cdf


def gen_stdvec(dim: int, start: int, end: int, on_std: float = 0.5, off_std: float = 0.0) -> torch.Tensor:
    """Generates a tensor specifying the standard deviation of Gaussian noise.

    This function creates a tensor of size `(dim,)` initialized with `off_std`, 
    and sets the values in the range `[start:end]` to `on_std`. 
    In PyTorch, due to broadcasting, multiplying an `[N, D]` tensor by a `[D]` 
    tensor results in an element-wise multiplication that is applied along the `N` direction.

    Args:
        dim (int): The size of the output tensor.
        start (int): The starting index (inclusive) of the region where `on_std` is applied.
        end (int): The ending index (exclusive) of the region where `on_std` is applied.
        on_std (float, optional): The standard deviation value for the active range. Default is `0.5`.
        off_std (float, optional): The standard deviation value for the inactive range. Default is `0.0`.

    Returns:
        torch.Tensor: A 1D tensor of shape `(dim,)` specifying the standard deviation for Gaussian noise.

    """
    stdvec = torch.full((dim,), off_std)  # Initialize with off_std
    stdvec[start:end] = on_std  # Set the specified range to on_std
    return stdvec
    

def interpolate_stdvecs(stdvec1: torch.Tensor, stdvec2: torch.Tensor, rate: float = 0.5) -> torch.Tensor:
    """Interpolates between two standard deviation vectors.

    This function performs linear interpolation between two tensors 
    representing standard deviation vectors.

    Args:
        stdvec1 (torch.Tensor): The first standard deviation vector.
        stdvec2 (torch.Tensor): The second standard deviation vector.
        rate (float, optional): The interpolation rate. A value of 0.0 
            returns `stdvec1`, a value of 1.0 returns `stdvec2`, and values 
            in between yield a linear combination. Default is 0.5.

    Returns:
        torch.Tensor: The interpolated standard deviation vector.
    """
    stdvec = (1.0 - rate) * stdvec1 + rate * stdvec2
    return stdvec


# =========================================================
# Simple check
# Run `python -m nnn.noise` from outside the nnn directory.
# =========================================================
if __name__ == "__main__":
    stdvec1 = gen_stdvec(10, 2, 5)
    print(stdvec1)
    stdvec2 = gen_stdvec(10, 4, 9)
    print(stdvec2)
    stdvec = interpolate_stdvecs(stdvec1, stdvec2)
    print(stdvec)
    
    alpha = torch.Tensor([
        [0.5, 1.0],
        [1.5, 2.0]
    ])
    beta = torch.Tensor([
        [0, 0],
        [0, 0]
    ])
    gamma = torch.Tensor([
        [1.0, 1.0],
        [1.0, 1.0]
    ])

    x = torch.Tensor([
        [0,0],
        [0,0]
    ])

    noise_gen = stable_noise_like(alpha, beta, gamma)
    y = noise_gen(x)
    print(y)

    pdf = stable_pdf_torch(alpha, beta, gamma)
    y = pdf(x)
    print(y)

    cdf = stable_cdf_torch(alpha, beta, gamma)
    y = cdf(x)
    print(y)
    