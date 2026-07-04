import math

import numpy as np
import torch


class Structure:

    def __init__(self, noise_structure, activation_ratio=0.5):
        self.noise_structure = noise_structure
        self.activation_ratio = activation_ratio
        self.real_structure = [1] + [np.prod(noise_structure)]*2+[1]

        # [8*8*8]  # [8, 8, 8]


def gaussian_fill(shape, center, sigma, amplitude=1.0, normalize=False):
    # print(f"{shape=} {center=}")
    """
    shape: [n0, n1, ..., nD]) with an N-D Gaussian:
      G(x) = amplitude * exp(-0.5 * sum_d ((x_d - center[d]) / sigma[d])**2)

    - center: list/tuple/1D tensor of length out.ndim in [0,1]
    - sigma:  scalar or sequence of length len(shape)
    - normalize=True will rescale so out.sum()==1 (discrete normalization)
    """
    D = len(shape)
    out = torch.empty(*shape, dtype=torch.float32)
    assert len(center) == D, f"center must have one entry per dimension {shape=} {center=}"

    device, dtype = out.device, out.dtype

    center = torch.as_tensor(center, dtype=dtype, device=device)
    sigma = torch.as_tensor(sigma, dtype=dtype, device=device)
    if sigma.numel() == 1:
        sigma = sigma.expand(D)

    out.fill_(amplitude)
    # Use separability: product of 1D Gaussians equals N-D Gaussian with diagonal covariance
    for d, n in enumerate(out.shape):
        coords = torch.linspace(0, 1, n, device=device, dtype=dtype)
        g1 = torch.exp(-0.5 *
                       ((coords - center[d]) / sigma[d]) ** 2)
        shape = [1] * D
        shape[d] = n
        out.mul_(g1.view(shape))
    if normalize:
        s = out.sum()
        if s > 0:
            out.div_(s)
    return out


def compute_sigma(a, p, d):
    """
    Compute sigma for a d-dimensional Gaussian with peak=1,
    such that the volume of the ball where amplitude > a is p.

    Parameters:
    d (int): Number of dimensions
    a (float): Amplitude threshold (0 < a < 1)
    p (float): Desired volume

    Returns:
    float: The value of sigma
    """
    if not (0 < a < 1):
        raise ValueError("a must be between 0 and 1")
    if p <= 0:
        raise ValueError("p must be positive")
    if d <= 0:
        raise ValueError("d must be positive")

    # Volume of unit ball in d dimensions
    C_d = math.pi**(d/2) / math.gamma(d/2 + 1)

    # Radius r such that volume of ball is p
    r = (p / C_d)**(1/d)

    # From r^2 = -2 * sigma^2 * ln(a)
    sigma_squared = - (r**2) / (2 * math.log(a))

    return math.sqrt(sigma_squared)


def noise_pattern(structure, center):
    cut_at = 0.1
    sigma = compute_sigma(
        cut_at, structure.activation_ratio, len(structure.noise_structure))
    # print(f"d={len(noise_structure)} sigma={sigma}")
    scaled_center = structure.activation_ratio / \
        2+(1-structure.activation_ratio)*np.array(center)
    stdvecs = gaussian_fill(shape=structure.noise_structure,
                            center=scaled_center,
                            sigma=[sigma for n in structure.noise_structure],
                            amplitude=1,
                            normalize=False
                            ).reshape(-1)
    stdvecs = torch.where(stdvecs >= cut_at, stdvecs, 0)
    return [stdvecs]*2


def noise_pattern_table(structure, shape):
    noise_patterns = np.empty(shape, dtype=object)
    # print(f"noise pattern_table {shape=}")
    for idx in np.ndindex(shape):
        noise_patterns[idx] = noise_pattern(
            structure, fractional_index(idx, shape))
    return noise_patterns


def fractional_index(idx, shape):
    fidx = tuple(i / (n - 1) if n > 1 else 0.0 for i, n in zip(idx, shape))
    return fidx
