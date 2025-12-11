import torch
import torch.nn as nn

from . import noise
from . import activation


class GaussianNoiseLayer(nn.Module):
    """Applies Gaussian noise to an input tensor.

    This layer generates Gaussian noise with the specified mean and standard deviation 
    and adds it element-wise to the input tensor.

    Args:
        mean (float, optional): The mean of the Gaussian noise. Defaults to `0.0`.
        std (float, optional): The standard deviation of the Gaussian noise. Defaults to `1.0`.
    """

    def __init__(self, mean=0.0, std=1.0):
        super(GaussianNoiseLayer, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor, mean: float = None, std: float = None) -> torch.Tensor:
        """Applies Gaussian noise to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            mean (float, optional): Override the mean value for noise generation.
            std (float, optional): Override the standard deviation for noise generation.

        Returns:
            torch.Tensor: The input tensor with added Gaussian noise.
        """
        if mean is not None:
            self.mean = mean
        if std is not None:
            self.std = std
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise


class SampleLayer(nn.Module):
    """Expands the input tensor along a new temporal dimension and duplicates it.

    This layer transforms an input tensor of shape `[N, D]` into `[N, T, D]`
    by adding a new dimension and repeating the input `T` times.

    Args:
        numT (int, optional): The number of times to duplicate the input along the new dimension. Defaults to `10`.
    """

    def __init__(self, numT=10):
        super(SampleLayer, self).__init__()
        self.numT = numT

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expands and duplicates the input tensor along a new temporal dimension.

        Args:
            x (torch.Tensor): Input tensor of shape `[N, D]`.

        Returns:
            torch.Tensor: Transformed tensor of shape `[N, T, D]`.
        """
        return x.unsqueeze(1).repeat(1, self.numT, 1)


class EnsembleMeanLayer(nn.Module):
    """Computes the mean along the temporal dimension of an input tensor.

    This layer takes an input tensor of shape `[N, T, D]` and computes the mean 
    along the T dimension, resulting in an output of shape `[N, D]`.
    """

    def __init__(self):
        super(EnsembleMeanLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the mean along the temporal dimension.

        Args:
            x (torch.Tensor): Input tensor of shape `[N, T, D]`.

        Returns:
            torch.Tensor: Output tensor of shape `[N, D]`, which is the mean along the T dimension.
        """
        return torch.mean(x, dim=1)


class FilterLayer(nn.Module):
    """ Computes the moving average along the temporal dimention of an input tensor.
    
    This layer takes an input tensor of shape `[N, T]` and applies the moving average
    along the T dimension, resulting in an output of shape `[N, T]`.
    Its backward is identical to the forward.

    Args:
        w: window size
    """
    def __init__(self, w: int):
        super(FilterLayer, self).__init__()
        if not isinstance(w, int) or w <= 0:
            raise ValueError("D must be a positive int.")
        self.w = w

        self.conv = nn.Conv1d(1, 1, w, stride=1, padding=w//2,
                              bias=False, padding_mode="circular")
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / w)  # uniform kernel
        self.conv.weight.requires_grad_(False)  # fixing the kernel
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the uniform kernel to take moving average.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the filter. Its shape is not changed.
        """
        if x.dim() != 2:
            raise ValueError("Input must be [N, T].")
        y = self.conv(x.unsqueeze(1)).squeeze(1) # add a dummy channel for conv1d and recover afterwards
        if self.w % 2 == 0: #if w is even
            y = y[:, :x.size(1)] # remove unnecessary element at the end
        return y


class GaussianCrossingLayer(nn.Module):
    """Applies the Crossing activation function (realtime) with Gaussian noise.

    This layer generates Gaussian noise with the given standard deviation and applies 
    the Crossing activation function. The noise can be adjusted dynamically.

    Args:
        std (float, optional): Standard deviation of the Gaussian noise. Defaults to `0.5`.
        h (float, optional): Threshold parameter for the Crossing activation function. Defaults to `0.2`.
    """

    def __init__(self, std=0.5, h=0.2):
        super(GaussianCrossingLayer, self).__init__()
        self.h = h
        self.std = std
        self.noise_layer = GaussianNoiseLayer(std=self.std)

    def forward(self, x: torch.Tensor, std: float = None) -> torch.Tensor:
        """Applies Gaussian noise and the Crossing activation function.

        Args:
            x (torch.Tensor): Input tensor.
            std (float, optional): Override the standard deviation for noise generation.

        Returns:
            torch.Tensor: Output tensor after applying noise and the Crossing activation function.
        """
        if std is not None:
            self.std = std
        x = self.noise_layer(x, std=self.std)
        return activation.Crossing.apply(x, self.h)


class GaussianCrossingSampleLayer(nn.Module):
    """Applies the Crossing activation function (sampling version) with Gaussian noise.

    This layer generates Gaussian noise with the given standard deviation and applies 
    the Crossing activation function. The noise can be adjusted dynamically.

    Args:
        std (float, optional): Standard deviation of the Gaussian noise. Defaults to `0.5`.
        h (float, optional): Threshold parameter for the Crossing activation function. Defaults to `0.2`.
    """

    def __init__(self, std=0.5, h=0.2):
        super(GaussianCrossingSampleLayer, self).__init__()
        self.h = h
        self.std = std
        self.noise_layer = GaussianNoiseLayer(std=self.std)

    def forward(self, x: torch.Tensor, std: float = None) -> torch.Tensor:
        """Applies Gaussian noise and the Crossing activation function.

        Args:
            x (torch.Tensor): Input tensor.
            std (float, optional): Override the standard deviation for noise generation.

        Returns:
            torch.Tensor: Output tensor after applying noise and the Crossing activation function.
        """
        if std is not None:
            self.std = std
        x = self.noise_layer(x, std=self.std)
        return activation.CrossingSample.apply(x, self.h)


class GaussianCrossingStatisticLayer(nn.Module):
    """Applies the statistical version of the Crossing activation function with Gaussian noise.

    This layer generates Gaussian noise samples, applies the Crossing activation function,
    and estimates statistical properties.

    Args:
        std (float, optional): Standard deviation of the Gaussian noise. Defaults to `0.5`.
        h (float, optional): Threshold parameter for the Crossing activation function. Defaults to `0.2`.
        numT (int, optional): Number of samples for statistical estimation. Defaults to `100`.
    """

    def __init__(self, std=0.5, h=0.2, numT=100):
        super(GaussianCrossingStatisticLayer, self).__init__()
        self.h = h
        self.std = std
        self.numT = numT

    def forward(self, x: torch.Tensor, std: float = None) -> torch.Tensor:
        """Applies Gaussian noise and the statistical version of the Crossing activation function.

        Args:
            x (torch.Tensor): Input tensor.
            std (float, optional): Override the standard deviation for noise generation.

        Returns:
            torch.Tensor: Output tensor after applying statistical noise processing and the Crossing activation function.
        """
        if std is not None:
            self.std = std
        noise_values = noise.gaussian_noise_like(0.0, self.std)
        return activation.CrossingStatistic.apply(x, self.h, self.numT, noise_values)


class GaussianCrossingAnalyticLayer(nn.Module):
    """Applies the analytic version of the Crossing activation function with Gaussian noise.

    This layer analytically computes the expected output using the probability density function (PDF) 
    and cumulative density function (CDF) of Gaussian noise.

    Args:
        std (float, optional): Standard deviation of the Gaussian noise. Defaults to `0.5`.
    """

    def __init__(self, std=0.5):
        super(GaussianCrossingAnalyticLayer, self).__init__()
        self.std = std

    def forward(self, x: torch.Tensor, std: float = None) -> torch.Tensor:
        """Applies the analytic version of the Crossing activation function.

        The function uses the probability density function (PDF) and cumulative density function (CDF)
        of Gaussian noise to compute the expected output.

        Args:
            x (torch.Tensor): Input tensor.
            std (float, optional): Override the standard deviation for noise generation.

        Returns:
            torch.Tensor: Output tensor after applying the analytic Crossing activation function.
        """
        if std is not None:
            self.std = std
        pdf = noise.gaussian_pdf_torch(0.0, self.std)
        cdf = noise.gaussian_cdf_torch(0.0, self.std)
        return activation.CrossingAnalytic.apply(x, pdf, cdf)
        


# =========================================================
# Simple check
# Run `python -m nnn.layer` from outside the nnn directory.
# =========================================================
if __name__ == "__main__":
    x = torch.rand(3, 5, 4) # Dummy data (N=3, T=5, D=4)
    w = torch.rand(3, 10) # Dummy data (N=3, T=10)
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
