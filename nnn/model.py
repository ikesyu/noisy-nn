import torch
import torch.nn as nn
from typing import Sequence
from typing import Optional

from . import noise
from . import activation
from . import layer


class SimpleNNNBase(nn.Module):
    """Sample-level NNN hidden base without the final linear output layer.
    Input:
        x: [N, D_in]
    Output:
        z_samples: [N, T, D_hidden_last]
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
                # [N, H] -> [N, T, H]
                x = self.sampled_layer(x)
            x = self.gaussian_crossing[i](x, stds[i])
        return x


class SimpleNNN(nn.Module):
    """A simple neural network using sample-based and realtime computations.

    This network processes inputs using multiple layers, applying Gaussian noise 
    and the Crossing activation function at each layer, and then 
    averaging the outputs at the final layer.

    Args:
        structure (list, optional): List of layer sizes, where `len(structure) >= 3`. Defaults to `[1,50,50,1]`.
        std (float, optional): Standard deviation of the Gaussian noise. Defaults to `0.5`.
        h (float, optional): Threshold parameter for the Crossing activation function. Defaults to `0.2`.
        w (int, optional): Width of moving average. Defaults to `10`.
        output_bias (bool, optional): Whether the output layer has a bias term. Defaults to `False`.

    Raises:
        ValueError: If `structure` has fewer than 3 layers.
    """

    def __init__(self, structure=[1,25,25,1], std=0.5, h=0.2, w=20, output_bias=False):
        if len(structure) < 3:
            raise ValueError("The structure list must have at least 3 elements.")

        super(SimpleNNN, self).__init__()
        self.fcs = nn.ModuleList()
        self.gaussian_crossing = nn.ModuleList()
        self.structure = structure
        self.std = std
        self.h = h
        self.w = w
        self.output_bias = output_bias
        self.filter_layer = layer.FilterLayer(w=self.w)

        for index, (pre, post) in enumerate(zip(self.structure, self.structure[1:])):
            self.fcs.append(nn.Linear(pre, post, bias=(self.output_bias if index == len(self.structure)-2 else True)))

        for i in range(len(self.structure)-2):
            self.gaussian_crossing.append(layer.GaussianCrossingLayer(self.std, self.h))

    def forward(self, x: torch.Tensor, stds: list = None) -> torch.Tensor:
        """Computes the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape `[N, D]`.
            stds (list, optional): List of standard deviations for each layer. 
                If `None`, the default `std` is used for all layers.

        Returns:
            torch.Tensor: Output tensor of shape `[N, D]`.

        Raises:
            ValueError: If `stds` is provided but its length does not match `len(structure)-2`.
        """
        if stds is not None and len(stds) != (len(self.structure) - 2):
            raise ValueError("The length of `stds` must match `len(structure)-2`.")
        if stds is None:
            stds = [self.std] * (len(self.structure) - 2)

        for i in range(len(self.structure) - 1):
            x = self.fcs[i](x)
            if i < len(self.structure) - 2:
                x = self.gaussian_crossing[i](x, stds[i])
            
        return x


class SimpleNNNSample(nn.Module):
    """A simple neural network using sample-based computations.

    This network processes inputs using multiple layers, applying Gaussian noise 
    and the Crossing activation function (sample version)at each layer, and then 
    averaging the outputs at the final layer.

    Args:
        structure (list, optional): List of layer sizes, where `len(structure) >= 3`. Defaults to `[1,25,25,1]`.
        std (float, optional): Standard deviation of the Gaussian noise. Defaults to `0.5`.
        h (float, optional): Threshold parameter for the Crossing activation function. Defaults to `0.2`.
        t (int, optional): Number of samples per input. Defaults to `10`.
        output_bias (bool, optional): Whether the output layer has a bias term. Defaults to `False`.

    Raises:
        ValueError: If `structure` has fewer than 3 layers.
    """

    def __init__(self, structure=[1,25,25,1], std=0.5, h=0.2, t=10, output_bias=False):
        if len(structure) < 3:
            raise ValueError("The structure list must have at least 3 elements.")

        super(SimpleNNNSample, self).__init__()
        self.fcs = nn.ModuleList()
        self.gaussian_crossing = nn.ModuleList()
        self.structure = structure
        self.std = std
        self.h = h
        self.t = t
        self.output_bias = output_bias
        self.sampled_layer = layer.SampleLayer(numT=self.t)
        self.ensemble_layer = layer.EnsembleMeanLayer()

        for index, (pre, post) in enumerate(zip(self.structure, self.structure[1:])):
            self.fcs.append(nn.Linear(pre, post, bias=(self.output_bias if index == len(self.structure)-2 else True)))

        for i in range(len(self.structure)-2):
            self.gaussian_crossing.append(layer.GaussianCrossingSampleLayer(self.std, self.h))

    def forward(self, x: torch.Tensor, stds: list = None) -> torch.Tensor:
        """Computes the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape `[N, D]`.
            stds (list, optional): List of standard deviations for each layer. 
                If `None`, the default `std` is used for all layers.

        Returns:
            torch.Tensor: Output tensor of shape `[N, D]`.

        Raises:
            ValueError: If `stds` is provided but its length does not match `len(structure)-2`.
        """
        if stds is not None and len(stds) != (len(self.structure) - 2):
            raise ValueError("The length of `stds` must match `len(structure)-2`.")
        if stds is None:
            stds = [self.std] * (len(self.structure) - 2)

        for i in range(len(self.structure) - 1):
            x = self.fcs[i](x)
            if i == 0:
                x = self.sampled_layer(x)
            if i < len(self.structure) - 2:
                x = self.gaussian_crossing[i](x, stds[i])
            if i == len(self.structure) - 2:
                x = self.ensemble_layer(x)
        return x


class SimpleNNNStatistic(nn.Module):
    """A simple neural network using statistical computation per layer.

    This network applies the Crossing activation function using statistical estimation, 
    instead of direct sampling, at each layer.

    Args:
        structure (list, optional): List of layer sizes, where `len(structure) >= 3`. Defaults to `[1,25,25,1]`.
        std (float, optional): Standard deviation of the Gaussian noise. Defaults to `0.5`.
        h (float, optional): Threshold parameter for the Crossing activation function. Defaults to `0.2`.
        t (int, optional): Number of samples for statistical estimation. Defaults to `10`.
        output_bias (bool, optional): Whether the output layer has a bias term. Defaults to `False`.

    Raises:
        ValueError: If `structure` has fewer than 3 layers.
    """

    def __init__(self, structure=[1,25,25,1], std=0.5, h=0.2, t=10, output_bias=False):
        if len(structure) < 3:
            raise ValueError("The structure list must have at least 3 elements.")

        super(SimpleNNNStatistic, self).__init__()
        self.fcs = nn.ModuleList()
        self.gaussian_crossing = nn.ModuleList()
        self.structure = structure
        self.std = std
        self.h = h
        self.t = t
        self.output_bias = output_bias

        for index, (pre, post) in enumerate(zip(self.structure, self.structure[1:])):
            self.fcs.append(nn.Linear(pre, post, bias=(self.output_bias if index == len(self.structure)-2 else True)))

        for i in range(len(self.structure) - 2):
            self.gaussian_crossing.append(layer.GaussianCrossingStatisticLayer(self.std, self.h, self.t))

    def forward(self, x: torch.Tensor, stds: list = None) -> torch.Tensor:
        """Computes the forward pass using statistical estimation.

        Args:
            x (torch.Tensor): Input tensor of shape `[N, D]`.
            stds (list, optional): List of standard deviations for each layer. 
                If `None`, the default `std` is used for all layers.

        Returns:
            torch.Tensor: Output tensor of shape `[N, D]`.

        Raises:
            ValueError: If `stds` is provided but its length does not match `len(structure)-2`.
        """
        if stds is not None and len(stds) != (len(self.structure) - 2):
            raise ValueError("The length of `stds` must match `len(structure)-2`.")
        if stds is None:
            stds = [self.std] * (len(self.structure) - 2)

        for i in range(len(self.structure) - 1):
            x = self.fcs[i](x)
            if i < len(self.structure) - 2:
                x = self.gaussian_crossing[i](x, stds[i])
        return x


class SimpleNNNAnalytic(nn.Module):
    """A simple neural network using an analytic representation of noise.

    Instead of sampling, this network utilizes the probability density function (PDF) 
    and cumulative density function (CDF) of Gaussian noise to compute the expected 
    output of the Crossing activation function.

    Args:
        structure (list, optional): List of layer sizes, where `len(structure) >= 3`. Defaults to `[1,25,25,1]`.
        std (float, optional): Standard deviation of the Gaussian noise. Defaults to `0.5`.
        output_bias (bool, optional): Whether the output layer has a bias term. Defaults to `False`.

    Raises:
        ValueError: If `structure` has fewer than 3 layers.
    """

    def __init__(self, structure=[1,25,25,1], std=0.5, output_bias=False):
        if len(structure) < 3:
            raise ValueError("The structure list must have at least 3 elements.")

        super(SimpleNNNAnalytic, self).__init__()
        self.fcs = nn.ModuleList()
        self.gaussian_crossing = nn.ModuleList()
        self.structure = structure
        self.std = std
        self.output_bias = output_bias

        for index, (pre, post) in enumerate(zip(self.structure, self.structure[1:])):
            self.fcs.append(nn.Linear(pre, post, bias=(self.output_bias if index == len(self.structure)-2 else True)))

        for i in range(len(self.structure) - 2):
            self.gaussian_crossing.append(layer.GaussianCrossingAnalyticLayer(self.std))

    def forward(self, x: torch.Tensor, stds: list = None) -> torch.Tensor:
        """Computes the forward pass using the analytic noise model.

        Args:
            x (torch.Tensor): Input tensor of shape `[N, D]`.
            stds (list, optional): List of standard deviations for each layer. 

        Returns:
            torch.Tensor: Output tensor of shape `[N, D]`.
        """
        for i in range(len(self.structure) - 1):
            x = self.fcs[i](x)
            if i < len(self.structure) - 2:
                x = self.gaussian_crossing[i](x, stds[i] if stds else self.std)
        return x


# =========================================================
# Compact-support NNN variants
# =========================================================
class SimpleNNNParabolicAnalytic(nn.Module):
    """Simple NNN using the parabolic analytical crossing response.

    This model extends SimpleNNNAnalytic by replacing the Gaussian analytical
    response with the exact analytical response induced by bounded uniform noise:

        z = 0.5 * [1 - ((d - c) / r)^2]_+.

    The `recruitment`/`recruitments` argument plays the same conceptual role as
    the noise-strength field in the original NNN: recruitment=0 detaches the unit,
    and larger values expand the active region.
    """

    def __init__(self, structure=[1, 25, 25, 1], recruitment=1.0, center=0.0,
                 max_radius=1.0, output_bias=False):
        if len(structure) < 3:
            raise ValueError("The structure list must have at least 3 elements.")

        super(SimpleNNNParabolicAnalytic, self).__init__()
        self.fcs = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.structure = structure
        self.recruitment = recruitment
        self.center = center
        self.max_radius = max_radius
        self.output_bias = output_bias

        for index, (pre, post) in enumerate(zip(self.structure, self.structure[1:])):
            self.fcs.append(nn.Linear(pre, post, bias=(self.output_bias if index == len(self.structure)-2 else True)))

        for _ in range(len(self.structure) - 2):
            self.activations.append(layer.ParabolicCrossingAnalyticLayer(
                recruitment=self.recruitment,
                center=self.center,
                max_radius=self.max_radius,
            ))

    def forward(self, x: torch.Tensor, recruitments: list = None, stds: list = None,
                radii: list = None, centers: list = None) -> torch.Tensor:
        """Computes the forward pass.

        Args:
            x: Input tensor of shape [N, D].
            recruitments: Optional list of recruitment/noise-field tensors, one per hidden layer.
            stds: Alias of recruitments for compatibility with existing NNN scripts.
            radii: Optional list of explicit radii. If given, it overrides recruitment.
            centers: Optional list of center values/tensors, one per hidden layer.
        """
        if recruitments is None and stds is not None:
            recruitments = stds
        n_hidden = len(self.structure) - 2
        if recruitments is not None and len(recruitments) != n_hidden:
            raise ValueError("The length of `recruitments` must match len(structure)-2.")
        if radii is not None and len(radii) != n_hidden:
            raise ValueError("The length of `radii` must match len(structure)-2.")
        if centers is not None and len(centers) != n_hidden:
            raise ValueError("The length of `centers` must match len(structure)-2.")

        for i in range(len(self.structure) - 1):
            x = self.fcs[i](x)
            if i < n_hidden:
                rec = recruitments[i] if recruitments is not None else self.recruitment
                rad = radii[i] if radii is not None else None
                cen = centers[i] if centers is not None else self.center
                x = self.activations[i](x, recruitment=rec, radius=rad, center=cen)
        return x


class SimpleNNNHatApproxAnalytic(nn.Module):
    """Simple NNN using a hat-shaped hardware-oriented approximation.

    Normalized mode:
        z = 0.5 * [1 - |d - c| / r]_+.

    Coupled mode:
        z = [r - |d - c|]_+.

    The `recruitment`/`recruitments` argument is mapped to radius and therefore
    behaves as a neuron-recruitment/noise-field intensity parameter.
    """

    def __init__(self, structure=[1, 25, 25, 1], recruitment=1.0, center=0.0,
                 max_radius=1.0, normalized=True, output_bias=False):
        if len(structure) < 3:
            raise ValueError("The structure list must have at least 3 elements.")

        super(SimpleNNNHatApproxAnalytic, self).__init__()
        self.fcs = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.structure = structure
        self.recruitment = recruitment
        self.center = center
        self.max_radius = max_radius
        self.normalized = normalized
        self.output_bias = output_bias

        for index, (pre, post) in enumerate(zip(self.structure, self.structure[1:])):
            self.fcs.append(nn.Linear(pre, post, bias=(self.output_bias if index == len(self.structure)-2 else True)))

        for _ in range(len(self.structure) - 2):
            self.activations.append(layer.HatApproxCrossingAnalyticLayer(
                recruitment=self.recruitment,
                center=self.center,
                max_radius=self.max_radius,
                normalized=self.normalized,
            ))

    def forward(self, x: torch.Tensor, recruitments: list = None, stds: list = None,
                radii: list = None, centers: list = None) -> torch.Tensor:
        """Computes the forward pass.

        Args:
            x: Input tensor of shape [N, D].
            recruitments: Optional list of recruitment/noise-field tensors, one per hidden layer.
            stds: Alias of recruitments for compatibility with existing NNN scripts.
            radii: Optional list of explicit radii. If given, it overrides recruitment.
            centers: Optional list of center values/tensors, one per hidden layer.
        """
        if recruitments is None and stds is not None:
            recruitments = stds
        n_hidden = len(self.structure) - 2
        if recruitments is not None and len(recruitments) != n_hidden:
            raise ValueError("The length of `recruitments` must match len(structure)-2.")
        if radii is not None and len(radii) != n_hidden:
            raise ValueError("The length of `radii` must match len(structure)-2.")
        if centers is not None and len(centers) != n_hidden:
            raise ValueError("The length of `centers` must match len(structure)-2.")

        for i in range(len(self.structure) - 1):
            x = self.fcs[i](x)
            if i < n_hidden:
                rec = recruitments[i] if recruitments is not None else self.recruitment
                rad = radii[i] if radii is not None else None
                cen = centers[i] if centers is not None else self.center
                x = self.activations[i](x, recruitment=rec, radius=rad, center=cen)
        return x


# =========================================================
# Simple check
# Run `python -m nnn.model` from outside the nnn directory.
# =========================================================
if __name__ == "__main__":
    import torch.optim as optim
    import numpy as np

    # Dataset
    #x = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
    x = np.linspace(-2 * np.pi, 2 * np.pi, 10000).reshape(-1, 1)
    y = np.sin(x)

    # Transform to PyTorch Tensor type
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    
    net = SimpleNNN()
    #net = SimpleNNNSample()
    #net = SimpleNNNStatistic()
    #net = SimpleNNNAnalytic()
    #net = SimpleNNNParabolicAnalytic()
    #net = SimpleNNNHatApproxAnalytic()

    # Loss and Optimization
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # Training
    epochs = 1000
    print("Training starts")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = net(x_tensor)
        loss = criterion(output, y_tensor)
        print("\r"+f"loss: {loss}",end="")
        loss.backward()
        optimizer.step()
    print(".")
    print("Training ends")