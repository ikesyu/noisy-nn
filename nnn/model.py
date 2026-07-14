"""Simple feedforward NNN models.

Every model is a stack of linear layers with a crossing activation between
them; the variants differ only in how the crossing response is computed:

    SimpleNNN                   realtime sampling on [N, T] streams
    SimpleNNNSample             T stochastic samples per input, ensemble-mean readout
    SimpleNNNStatistic          per-layer statistical estimation (internal sampling)
    SimpleNNNAnalytic           closed-form expected response (Gaussian noise)
    SimpleNNNParabolicAnalytic  closed-form expected response (bounded uniform noise)
    SimpleNNNHatApproxAnalytic  piecewise-linear hardware-oriented approximation
    SimpleNNNUniformSample      Monte-Carlo counterpart of the parabolic model

`structure` lists the layer sizes [D_in, H_1, ..., H_k, D_out]; the crossing
is applied to every layer except the last (linear readout).
"""
import torch
import torch.nn as nn
from typing import Optional, Sequence

from . import layer


def _build_fcs(structure: Sequence[int], output_bias: bool) -> nn.ModuleList:
    """Linear layers for `structure`; only the output layer's bias follows `output_bias`."""
    out_index = len(structure) - 2
    return nn.ModuleList([
        nn.Linear(pre, post, bias=(output_bias if index == out_index else True))
        for index, (pre, post) in enumerate(zip(structure, structure[1:]))
    ])


def _check_per_layer(name: str, values, n_hidden: int) -> None:
    """Raises if a per-hidden-layer list has the wrong length."""
    if values is not None and len(values) != n_hidden:
        raise ValueError(f"The length of `{name}` must match len(structure)-2.")


class SimpleNNNBase(nn.Module):
    """Sample-level NNN hidden stack without a final linear readout.

    Input: x [N, D_in].  Output: z_samples [N, T, D_hidden_last].
    """

    def __init__(self, structure: Sequence[int] = (1, 64, 64), std: float = 0.6,
                 h: float = 0.15, t: int = 64):
        super().__init__()
        if len(structure) < 2:
            raise ValueError("structure must contain input dim and at least one hidden dim.")

        self.structure = list(structure)
        self.std = std
        self.h = h
        self.t = t

        self.fcs = nn.ModuleList([
            nn.Linear(pre, post, bias=True)
            for pre, post in zip(self.structure[:-1], self.structure[1:])
        ])
        self.gaussian_crossing = nn.ModuleList([
            layer.GaussianCrossingSampleLayer(std=self.std, h=self.h)
            for _ in self.fcs
        ])
        self.sampled_layer = layer.SampleLayer(numT=self.t)

    @property
    def out_features(self) -> int:
        return self.structure[-1]

    def forward(self, x: torch.Tensor, stds: Optional[Sequence[float]] = None) -> torch.Tensor:
        if stds is None:
            stds = [self.std] * len(self.fcs)
        if len(stds) != len(self.fcs):
            raise ValueError("len(stds) must match the number of hidden layers.")

        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i == 0:
                x = self.sampled_layer(x)   # [N, H] -> [N, T, H]
            x = self.gaussian_crossing[i](x, stds[i])
        return x


class SimpleNNN(nn.Module):
    """NNN with the realtime crossing on [N, T] streams.

    Gaussian noise and `Crossing` are applied after every hidden layer; the
    output layer is linear. `stds` in `forward` sets a per-layer noise std.
    """

    def __init__(self, structure=(1, 25, 25, 1), std=0.5, h=0.2, w=20, output_bias=False):
        if len(structure) < 3:
            raise ValueError("The structure list must have at least 3 elements.")

        super().__init__()
        self.structure = list(structure)
        self.std = std
        self.h = h
        self.w = w
        self.output_bias = output_bias
        self.filter_layer = layer.FilterLayer(w=self.w)

        self.fcs = _build_fcs(self.structure, output_bias)
        self.gaussian_crossing = nn.ModuleList([
            layer.GaussianCrossingLayer(self.std, self.h)
            for _ in range(len(self.structure) - 2)
        ])

    def forward(self, x: torch.Tensor, stds: list = None) -> torch.Tensor:
        n_hidden = len(self.structure) - 2
        _check_per_layer("stds", stds, n_hidden)
        if stds is None:
            stds = [self.std] * n_hidden

        for i in range(len(self.structure) - 1):
            x = self.fcs[i](x)
            if i < n_hidden:
                x = self.gaussian_crossing[i](x, stds[i])
        return x


class SimpleNNNSample(nn.Module):
    """NNN with the sampling crossing: T stochastic samples per input.

    The input is expanded to [N, T, D] after the first linear layer, Gaussian
    noise and `CrossingSample` are applied at every hidden layer, and the
    linear readout is averaged over T (ensemble mean).
    """

    def __init__(self, structure=(1, 25, 25, 1), std=0.5, h=0.2, t=10, output_bias=False):
        if len(structure) < 3:
            raise ValueError("The structure list must have at least 3 elements.")

        super().__init__()
        self.structure = list(structure)
        self.std = std
        self.h = h
        self.t = t
        self.output_bias = output_bias
        self.sampled_layer = layer.SampleLayer(numT=self.t)
        self.ensemble_layer = layer.EnsembleMeanLayer()

        self.fcs = _build_fcs(self.structure, output_bias)
        self.gaussian_crossing = nn.ModuleList([
            layer.GaussianCrossingSampleLayer(self.std, self.h)
            for _ in range(len(self.structure) - 2)
        ])

    def forward(self, x: torch.Tensor, stds: list = None) -> torch.Tensor:
        n_hidden = len(self.structure) - 2
        _check_per_layer("stds", stds, n_hidden)
        if stds is None:
            stds = [self.std] * n_hidden

        for i in range(len(self.structure) - 1):
            x = self.fcs[i](x)
            if i == 0:
                x = self.sampled_layer(x)
            if i < n_hidden:
                x = self.gaussian_crossing[i](x, stds[i])
            if i == n_hidden:
                x = self.ensemble_layer(x)
        return x


class SimpleNNNStatistic(nn.Module):
    """NNN with the statistic crossing: per-layer internal sampling on [N, D].

    Each hidden layer estimates the expected crossing response from `t`
    internal noise samples; no explicit T dimension appears outside the layer.
    """

    def __init__(self, structure=(1, 25, 25, 1), std=0.5, h=0.2, t=10, output_bias=False):
        if len(structure) < 3:
            raise ValueError("The structure list must have at least 3 elements.")

        super().__init__()
        self.structure = list(structure)
        self.std = std
        self.h = h
        self.t = t
        self.output_bias = output_bias

        self.fcs = _build_fcs(self.structure, output_bias)
        self.gaussian_crossing = nn.ModuleList([
            layer.GaussianCrossingStatisticLayer(self.std, self.h, self.t)
            for _ in range(len(self.structure) - 2)
        ])

    def forward(self, x: torch.Tensor, stds: list = None) -> torch.Tensor:
        n_hidden = len(self.structure) - 2
        _check_per_layer("stds", stds, n_hidden)
        if stds is None:
            stds = [self.std] * n_hidden

        for i in range(len(self.structure) - 1):
            x = self.fcs[i](x)
            if i < n_hidden:
                x = self.gaussian_crossing[i](x, stds[i])
        return x


class SimpleNNNAnalytic(nn.Module):
    """NNN with the analytic crossing: closed-form expected response, no sampling.

    Uses the Gaussian PDF/CDF to compute the expected crossing response
    deterministically at every hidden layer.
    """

    def __init__(self, structure=(1, 25, 25, 1), std=0.5, output_bias=False):
        if len(structure) < 3:
            raise ValueError("The structure list must have at least 3 elements.")

        super().__init__()
        self.structure = list(structure)
        self.std = std
        self.output_bias = output_bias

        self.fcs = _build_fcs(self.structure, output_bias)
        self.gaussian_crossing = nn.ModuleList([
            layer.GaussianCrossingAnalyticLayer(self.std)
            for _ in range(len(self.structure) - 2)
        ])

    def forward(self, x: torch.Tensor, stds: list = None) -> torch.Tensor:
        for i in range(len(self.structure) - 1):
            x = self.fcs[i](x)
            if i < len(self.structure) - 2:
                x = self.gaussian_crossing[i](x, stds[i] if stds else self.std)
        return x


class SimpleNNNParabolicAnalytic(nn.Module):
    """NNN with the parabolic analytic crossing (bounded uniform noise).

    Replaces the Gaussian analytic response with the exact response induced by
    bounded uniform noise: z = 0.5 * [1 - ((d - c) / r)^2]_+. The
    `recruitment(s)` argument plays the same role as the noise-strength field
    in the original NNN: recruitment = 0 detaches the unit.
    """

    def __init__(self, structure=(1, 25, 25, 1), recruitment=1.0, center=0.0,
                 max_radius=1.0, output_bias=False):
        if len(structure) < 3:
            raise ValueError("The structure list must have at least 3 elements.")

        super().__init__()
        self.structure = list(structure)
        self.recruitment = recruitment
        self.center = center
        self.max_radius = max_radius
        self.output_bias = output_bias

        self.fcs = _build_fcs(self.structure, output_bias)
        self.activations = nn.ModuleList([
            layer.ParabolicCrossingAnalyticLayer(
                recruitment=self.recruitment,
                center=self.center,
                max_radius=self.max_radius,
            )
            for _ in range(len(self.structure) - 2)
        ])

    def forward(self, x: torch.Tensor, recruitments: list = None, stds: list = None,
                radii: list = None, centers: list = None) -> torch.Tensor:
        """`stds` is accepted as an alias of `recruitments` for compatibility
        with existing NNN scripts; an explicit `radii` overrides recruitment."""
        if recruitments is None and stds is not None:
            recruitments = stds
        n_hidden = len(self.structure) - 2
        _check_per_layer("recruitments", recruitments, n_hidden)
        _check_per_layer("radii", radii, n_hidden)
        _check_per_layer("centers", centers, n_hidden)

        for i in range(len(self.structure) - 1):
            x = self.fcs[i](x)
            if i < n_hidden:
                rec = recruitments[i] if recruitments is not None else self.recruitment
                rad = radii[i] if radii is not None else None
                cen = centers[i] if centers is not None else self.center
                x = self.activations[i](x, recruitment=rec, radius=rad, center=cen)
        return x


class SimpleNNNHatApproxAnalytic(nn.Module):
    """NNN with the hat-shaped hardware-oriented crossing approximation.

    Normalized mode: z = 0.5 * [1 - |d - c| / r]_+.
    Coupled mode:    z = [r - |d - c|]_+.
    The `recruitment(s)` argument is mapped to the radius and therefore acts
    as a neuron-recruitment/noise-field intensity.
    """

    def __init__(self, structure=(1, 25, 25, 1), recruitment=1.0, center=0.0,
                 max_radius=1.0, normalized=True, output_bias=False):
        if len(structure) < 3:
            raise ValueError("The structure list must have at least 3 elements.")

        super().__init__()
        self.structure = list(structure)
        self.recruitment = recruitment
        self.center = center
        self.max_radius = max_radius
        self.normalized = normalized
        self.output_bias = output_bias

        self.fcs = _build_fcs(self.structure, output_bias)
        self.activations = nn.ModuleList([
            layer.HatApproxCrossingAnalyticLayer(
                recruitment=self.recruitment,
                center=self.center,
                max_radius=self.max_radius,
                normalized=self.normalized,
            )
            for _ in range(len(self.structure) - 2)
        ])

    def forward(self, x: torch.Tensor, recruitments: list = None, stds: list = None,
                radii: list = None, centers: list = None) -> torch.Tensor:
        """`stds` is accepted as an alias of `recruitments` for compatibility
        with existing NNN scripts; an explicit `radii` overrides recruitment."""
        if recruitments is None and stds is not None:
            recruitments = stds
        n_hidden = len(self.structure) - 2
        _check_per_layer("recruitments", recruitments, n_hidden)
        _check_per_layer("radii", radii, n_hidden)
        _check_per_layer("centers", centers, n_hidden)

        for i in range(len(self.structure) - 1):
            x = self.fcs[i](x)
            if i < n_hidden:
                rec = recruitments[i] if recruitments is not None else self.recruitment
                rad = radii[i] if radii is not None else None
                cen = centers[i] if centers is not None else self.center
                x = self.activations[i](x, recruitment=rec, radius=rad, center=cen)
        return x


class SimpleNNNUniformSample(nn.Module):
    """NNN with the sampling crossing under bounded uniform noise.

    Monte-Carlo counterpart of `SimpleNNNParabolicAnalytic`: uniform noise
    Uniform(center - radius, center + radius) and `CrossingSample` at every
    hidden layer, with an ensemble-mean linear readout over `t` samples.
    """

    def __init__(self, structure=(1, 25, 25, 1), radius=1.0, center=0.0,
                 h=0.1, t=64, output_bias=False):
        if len(structure) < 3:
            raise ValueError("The structure list must have at least 3 elements.")

        super().__init__()
        self.structure = list(structure)
        self.radius = radius
        self.center = center
        self.h = h
        self.t = t
        self.output_bias = output_bias

        self.fcs = _build_fcs(self.structure, output_bias)
        self.uniform_crossing = nn.ModuleList([
            layer.UniformCrossingSampleLayer(radius=radius, center=center, h=h)
            for _ in range(len(self.structure) - 2)
        ])
        self.sampled_layer = layer.SampleLayer(numT=t)
        self.ensemble_layer = layer.EnsembleMeanLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_hidden = len(self.structure) - 2
        for i in range(len(self.structure) - 1):
            x = self.fcs[i](x)
            if i == 0:
                x = self.sampled_layer(x)
            if i < n_hidden:
                x = self.uniform_crossing[i](x)
            if i == n_hidden:
                x = self.ensemble_layer(x)
        return x


# =========================================================
# Simple check: run `python -m nnn.model` from the repository root.
# =========================================================
if __name__ == "__main__":
    import numpy as np
    import torch.optim as optim

    x = np.linspace(-2 * np.pi, 2 * np.pi, 10000).reshape(-1, 1)
    y = np.sin(x)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    net = SimpleNNN()
    # net = SimpleNNNSample()
    # net = SimpleNNNStatistic()
    # net = SimpleNNNAnalytic()
    # net = SimpleNNNParabolicAnalytic()
    # net = SimpleNNNHatApproxAnalytic()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    epochs = 1000
    print("Training starts")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = net(x_tensor)
        loss = criterion(output, y_tensor)
        print("\r" + f"loss: {loss}", end="")
        loss.backward()
        optimizer.step()
    print(".")
    print("Training ends")
