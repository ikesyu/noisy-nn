"""Layers of the NNN: noise injection, sampling, readout, and crossing activations.

Each `*CrossingLayer` combines a noise source with the corresponding crossing
activation from `nnn.activation`. Tensor shapes follow the convention
[N, D] (per input) or [N, T, D] (T stochastic samples per input).
"""
import torch
import torch.nn as nn

from . import noise
from . import activation


class GaussianNoiseLayer(nn.Module):
    """Adds Gaussian noise N(mean, std^2) to the input.

    `mean` / `std` given to `forward` override and replace the stored values.
    """

    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor, mean: float = None, std: float = None) -> torch.Tensor:
        if mean is not None:
            self.mean = mean
        if std is not None:
            self.std = std
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise


class SampleLayer(nn.Module):
    """Expands [N, D] to [N, T, D] by repeating the input T (= numT) times."""

    def __init__(self, numT=10):
        super().__init__()
        self.numT = numT

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1).repeat(1, self.numT, 1)


class EnsembleMeanLayer(nn.Module):
    """Reduces [N, T, D] to [N, D] by averaging over the T dimension."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=1)


class FilterLayer(nn.Module):
    """Circular moving average of window `w` along the T dimension of [N, T].

    Implemented as a fixed (non-trainable) uniform Conv1d kernel, so the
    backward pass is the same moving average.
    """

    def __init__(self, w: int):
        super().__init__()
        if not isinstance(w, int) or w <= 0:
            raise ValueError("w must be a positive int.")
        self.w = w

        self.conv = nn.Conv1d(1, 1, w, stride=1, padding=w // 2,
                              bias=False, padding_mode="circular")
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / w)
        self.conv.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("Input must be [N, T].")
        # add a dummy channel for Conv1d and remove it afterwards
        y = self.conv(x.unsqueeze(1)).squeeze(1)
        if self.w % 2 == 0:
            y = y[:, :x.size(1)]    # even w: drop the extra trailing element
        return y


class GaussianCrossingLayer(nn.Module):
    """Gaussian noise + `Crossing` (realtime version, [N, T]).

    An `std` given to `forward` overrides and replaces the stored value.
    """

    def __init__(self, std=0.5, h=0.2):
        super().__init__()
        self.h = h
        self.std = std
        self.noise_layer = GaussianNoiseLayer(std=self.std)

    def forward(self, x: torch.Tensor, std: float = None) -> torch.Tensor:
        if std is not None:
            self.std = std
        x = self.noise_layer(x, std=self.std)
        return activation.Crossing.apply(x, self.h)


class GaussianCrossingSampleLayer(nn.Module):
    """Gaussian noise + `CrossingSample` (sampling version, [N, T, D]).

    An `std` given to `forward` overrides and replaces the stored value.
    """

    def __init__(self, std=0.5, h=0.2):
        super().__init__()
        self.h = h
        self.std = std
        self.noise_layer = GaussianNoiseLayer(std=self.std)

    def forward(self, x: torch.Tensor, std: float = None) -> torch.Tensor:
        if std is not None:
            self.std = std
        x = self.noise_layer(x, std=self.std)
        return activation.CrossingSample.apply(x, self.h)


class GaussianCrossingStatisticLayer(nn.Module):
    """Gaussian noise + `CrossingStatistic` (statistic version, [N, D]).

    Draws `numT` noise samples internally and returns the sample mean of the
    crossing response.
    """

    def __init__(self, std=0.5, h=0.2, numT=100):
        super().__init__()
        self.h = h
        self.std = std
        self.numT = numT

    def forward(self, x: torch.Tensor, std: float = None) -> torch.Tensor:
        if std is not None:
            self.std = std
        noise_values = noise.gaussian_noise_like(0.0, self.std)
        return activation.CrossingStatistic.apply(x, self.h, self.numT, noise_values)


class GaussianCrossingAnalyticLayer(nn.Module):
    """`CrossingAnalytic` with the Gaussian PDF/CDF ([N, D]).

    Computes the expected crossing response in closed form; no sampling.
    """

    def __init__(self, std=0.5):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor, std: float = None) -> torch.Tensor:
        if std is not None:
            self.std = std
        pdf = noise.gaussian_pdf_torch(0.0, self.std)
        cdf = noise.gaussian_cdf_torch(0.0, self.std)
        return activation.CrossingAnalytic.apply(x, pdf, cdf)


class ParabolicCrossingAnalyticLayer(nn.Module):
    """`ParabolicCrossingAnalytic` (analytic response of bounded uniform noise).

    `recruitment` is a noise-field intensity in [0, 1], mapped to
    radius = max_radius * recruitment; recruitment = 0 makes the unit
    completely inactive, matching the role of zero noise in the NNN.
    An explicit `radius` given to `forward` overrides the recruitment.
    """

    def __init__(self, recruitment=1.0, center=0.0, max_radius=1.0, epsilon=1e-10):
        super().__init__()
        self.recruitment = recruitment
        self.center = center
        self.max_radius = max_radius
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, recruitment=None, radius=None, center=None) -> torch.Tensor:
        if recruitment is not None:
            self.recruitment = recruitment
        if center is not None:
            self.center = center
        if radius is None:
            radius = noise.recruitment_to_radius(self.recruitment, self.max_radius)
        return activation.ParabolicCrossingAnalytic.apply(x, self.center, radius, self.epsilon)


class HatApproxCrossingAnalyticLayer(nn.Module):
    """`HatApproxCrossingAnalytic` (piecewise-linear hardware-oriented response).

    `recruitment` is mapped to radius = max_radius * recruitment, so the same
    noise-field vector can be used as a neuron-recruitment vector. See the
    activation for the normalized/coupled modes.
    """

    def __init__(self, recruitment=1.0, center=0.0, max_radius=1.0, normalized=True, epsilon=1e-10):
        super().__init__()
        self.recruitment = recruitment
        self.center = center
        self.max_radius = max_radius
        self.normalized = normalized
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, recruitment=None, radius=None, center=None) -> torch.Tensor:
        if recruitment is not None:
            self.recruitment = recruitment
        if center is not None:
            self.center = center
        if radius is None:
            radius = noise.recruitment_to_radius(self.recruitment, self.max_radius)
        return activation.HatApproxCrossingAnalytic.apply(x, self.center, radius, self.normalized, self.epsilon)


class UniformCrossingSampleLayer(nn.Module):
    """Bounded uniform noise + `CrossingSample` ([N, T, D]).

    Noise is drawn from Uniform(center - radius, center + radius); the
    expectation over samples recovers the exact parabolic analytic response,
    so this layer is the Monte-Carlo counterpart of
    `ParabolicCrossingAnalyticLayer`. A `radius` given to `forward` applies to
    that call only.
    """

    def __init__(self, radius: float = 1.0, center: float = 0.0, h: float = 0.1):
        super().__init__()
        self.radius = radius
        self.center = center
        self.h = h

    def forward(self, x: torch.Tensor, radius: float = None) -> torch.Tensor:
        r = radius if radius is not None else self.radius
        noise_gen = noise.uniform_noise_like(center=self.center, radius=r)
        return activation.CrossingSample.apply(x + noise_gen(x), self.h)


# =========================================================
# Simple check: run `python -m nnn.layer` from the repository root.
# =========================================================
if __name__ == "__main__":
    x = torch.rand(3, 5, 4)     # [N, T, D]
    w = torch.rand(3, 10)       # [N, T]
    print("Input tensor shape:", x.shape)

    noise_layer = GaussianNoiseLayer(std=10.0)
    ensemble_layer = EnsembleMeanLayer()
    sample_layer = SampleLayer(numT=3)
    filter_layer = FilterLayer(w=2)

    noisy = noise_layer(x)
    output = ensemble_layer(noisy)
    filt = filter_layer(w)
    sampled = sample_layer(output)

    print("Output tensor shape:", output.shape)
    print("Input tensor:\n", x)
    print("Noisy input tensor:\n", noisy)
    print("Output tensor:\n", output)
    print("Before Filter:\n", w)
    print("Filtered output tensor:\n", filt)
    print("Sampled tensor:\n", sampled)
