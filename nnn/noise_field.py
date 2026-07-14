"""Noise-field utilities: spatial Gaussian patterns of per-unit noise strength.

A noise field assigns each hidden unit a noise intensity; units with zero
intensity are detached from both inference and learning. Here the field is a
Gaussian bump placed on a virtual grid of units (`Structure.noise_structure`).
"""
import math

import numpy as np
import torch


class Structure:
    """Virtual unit grid of a noise field.

    `noise_structure` is the grid shape (e.g. [8, 8]); the actual network has
    `prod(noise_structure)` units per hidden layer (`real_structure`).
    `activation_ratio` is the target volume of the recruited region.
    """

    def __init__(self, noise_structure, activation_ratio=0.5):
        self.noise_structure = noise_structure
        self.activation_ratio = activation_ratio
        self.real_structure = [1] + [np.prod(noise_structure)] * 2 + [1]


def gaussian_fill(shape, center, sigma, amplitude=1.0, normalize=False):
    """Fills a tensor of `shape` with an N-D Gaussian bump.

        G(x) = amplitude * exp(-0.5 * sum_d ((x_d - center[d]) / sigma[d])^2)

    Coordinates run over [0, 1] per dimension. `center` has one entry per
    dimension; `sigma` is a scalar or one entry per dimension.
    `normalize=True` rescales so the tensor sums to 1.
    """
    D = len(shape)
    out = torch.empty(*shape, dtype=torch.float32)
    assert len(center) == D, f"center must have one entry per dimension {shape=} {center=}"

    device, dtype = out.device, out.dtype
    center = torch.as_tensor(center, dtype=dtype, device=device)
    sigma = torch.as_tensor(sigma, dtype=dtype, device=device)
    if sigma.numel() == 1:
        sigma = sigma.expand(D)

    # separability: a product of 1-D Gaussians equals an N-D Gaussian
    # with diagonal covariance
    out.fill_(amplitude)
    for d, n in enumerate(out.shape):
        coords = torch.linspace(0, 1, n, device=device, dtype=dtype)
        g1 = torch.exp(-0.5 * ((coords - center[d]) / sigma[d]) ** 2)
        shape = [1] * D
        shape[d] = n
        out.mul_(g1.view(shape))
    if normalize:
        s = out.sum()
        if s > 0:
            out.div_(s)
    return out


def compute_sigma(a, p, d):
    """Returns the sigma of a d-dimensional Gaussian with peak 1 such that the
    ball where the amplitude exceeds `a` has volume `p`."""
    if not (0 < a < 1):
        raise ValueError("a must be between 0 and 1")
    if p <= 0:
        raise ValueError("p must be positive")
    if d <= 0:
        raise ValueError("d must be positive")

    C_d = math.pi ** (d / 2) / math.gamma(d / 2 + 1)    # volume of the unit ball
    r = (p / C_d) ** (1 / d)                            # ball of volume p
    sigma_squared = -(r ** 2) / (2 * math.log(a))       # from r^2 = -2 sigma^2 ln(a)
    return math.sqrt(sigma_squared)


def noise_pattern(structure: Structure, center):
    """Returns per-layer std vectors for a Gaussian bump centred at `center`.

    `center` is a fractional position in [0, 1] per grid dimension. The bump
    is cut off below 0.1, and its sigma is chosen so the recruited region has
    volume `structure.activation_ratio`. Returns one flattened [prod(grid)]
    vector per hidden layer (two identical layers here).
    """
    cut_at = 0.1
    sigma = compute_sigma(
        cut_at, structure.activation_ratio, len(structure.noise_structure))
    scaled_center = structure.activation_ratio / \
        2 + (1 - structure.activation_ratio) * np.array(center)
    stdvecs = gaussian_fill(shape=structure.noise_structure,
                            center=scaled_center,
                            sigma=[sigma for n in structure.noise_structure],
                            amplitude=1,
                            normalize=False
                            ).reshape(-1)
    stdvecs = torch.where(stdvecs >= cut_at, stdvecs, 0)
    return [stdvecs] * 2


def noise_pattern_table(structure: Structure, shape):
    """Returns a `shape`-shaped object array of noise patterns whose bump
    centres are spread evenly over the grid."""
    noise_patterns = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        noise_patterns[idx] = noise_pattern(
            structure, fractional_index(idx, shape))
    return noise_patterns


def fractional_index(idx, shape):
    """Maps an integer index to fractional coordinates in [0, 1] per dimension."""
    return tuple(i / (n - 1) if n > 1 else 0.0 for i, n in zip(idx, shape))
