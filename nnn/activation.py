import torch
from . import noise


class Crossing(torch.autograd.Function):
    """Crossing activation function.

    This activation function computes the XOR operation along the time axis (T dimension)
    after binarizing the input tensor at thresholds `+h` and `-h`. It is designed as a 
    stateless version, meaning that noise addition and statistical calculations should 
    be implemented as custom layers separately.

    Reference:
        Ikemoto, Neurocomputing 2021.

    Attributes:
        None (Stateless function).
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, h: float = 0.05) -> torch.Tensor:
        """Computes the forward pass of the Crossing activation function.

        The function binarizes the input tensor at thresholds `+h` and `-h`, computes
        the XOR operation along the T dimension, and averages the two results.

        Args:
            ctx: Autograd context to save tensors for backward computation.
            input (torch.Tensor): Input tensor of shape `[N, T]`.
            h (float, optional): Threshold value for binarization. Defaults to 0.05.
                Must be positive.

        Returns:
            torch.Tensor: The transformed tensor of the same shape `[N, T]`, where
            each element represents the XOR operation results.

        Raises:
            ValueError: If `h` is not positive.
        """
        h = abs(h)  # Ensure h > 0
        bin1 = (input > h).float()  # Binarization with threshold +h
        bin2 = (input > -h).float()  # Binarization with threshold -h

        # Compute XOR along T dimension
        xor1 = bin1[:, 1:] - bin1[:, :-1]
        xor1 = torch.abs(xor1)  # XOR from indices 0 to T-2
        xor1_last = torch.abs(bin1[:, -1] - bin1[:, 0])
        xor1 = torch.cat([xor1, xor1_last.unsqueeze(1)], dim=1)

        xor2 = bin2[:, 1:] - bin2[:, :-1]
        xor2 = torch.abs(xor2)  # XOR from indices 0 to T-2
        xor2_last = torch.abs(bin2[:, -1] - bin2[:, 0])
        xor2 = torch.cat([xor2, xor2_last.unsqueeze(1)], dim=1)

        # Save tensors for backward computation
        ctx.save_for_backward(xor1, xor2)
        ctx.h = h

        return (xor2 + xor1) / 2

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Computes the backward pass of the Crossing activation function.

        The gradient is computed as one sample in a kernel density estimation
        using a uniform kernel with width `2h`.

        Args:
            ctx: Autograd context with saved tensors from the forward pass.
            grad_output (torch.Tensor): Gradient of the output tensor.

        Returns:
            torch.Tensor: The computed gradient for the input tensor.
            None: Placeholder for `h`, which does not require gradients.
        """
        xor1, xor2 = ctx.saved_tensors
        h = ctx.h

        # Kernel density estimation
        coeff = (xor2 - xor1) / (2 * h)
        #
        #
        #conv = torch.nn.Conv1d(1, 1, 11, stride=1, padding=11//2,
        #                 bias=False, padding_mode="circular")
        #with torch.no_grad():
        #    conv.weight.fill_(1.0 / 11.0)  # uniform kernel
        #conv.weight.requires_grad_(False)  # fixing the kernel
        #coeff = conv(coef.unsqueeze(1)).squeeze(1) # add a dummy channel for conv1d and recover afterwards
        #
        #
        grad = coeff * grad_output  # Apply chain rule

        return grad, None
        

class CrossingSample(torch.autograd.Function):
    """Crossing activation function (sampling version).

    This activation function computes the XOR operation along the second axis (T dimension)
    after binarizing the input tensor at thresholds `+h` and `-h`. It is designed as a 
    stateless version, meaning that noise addition and statistical calculations should 
    be implemented as custom layers separately.

    Reference:
        Ikemoto, Neurocomputing 2021.

    Attributes:
        None (Stateless function).
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, h: float = 0.05) -> torch.Tensor:
        """Computes the forward pass of the Crossing activation function.

        The function binarizes the input tensor at thresholds `+h` and `-h`, computes
        the XOR operation along the T dimension, and averages the two results.

        Args:
            ctx: Autograd context to save tensors for backward computation.
            input (torch.Tensor): Input tensor of shape `[N, T, D]`.
            h (float, optional): Threshold value for binarization. Defaults to 0.05.
                Must be positive.

        Returns:
            torch.Tensor: The transformed tensor of the same shape `[N, T, D]`, where
            each element represents the XOR operation results.

        Raises:
            ValueError: If `h` is not positive.
        """
        h = abs(h)  # Ensure h > 0
        bin1 = (input > h).float()  # Binarization with threshold +h
        bin2 = (input > -h).float()  # Binarization with threshold -h

        # Compute XOR along T dimension
        xor1 = bin1[:, 1:, :] - bin1[:, :-1, :]
        xor1 = torch.abs(xor1)  # XOR from indices 0 to T-2
        xor1_last = torch.abs(bin1[:, -1, :] - bin1[:, 0, :])
        xor1 = torch.cat([xor1, xor1_last.unsqueeze(1)], dim=1)

        xor2 = bin2[:, 1:, :] - bin2[:, :-1, :]
        xor2 = torch.abs(xor2)  # XOR from indices 0 to T-2
        xor2_last = torch.abs(bin2[:, -1, :] - bin2[:, 0, :])
        xor2 = torch.cat([xor2, xor2_last.unsqueeze(1)], dim=1)

        # Save tensors for backward computation
        ctx.save_for_backward(xor1, xor2)
        ctx.h = h

        return (xor2 + xor1) / 2

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Computes the backward pass of the Crossing activation function.

        The gradient is computed based on a kernel density estimation using a 
        uniform kernel with width `2h`.

        Args:
            ctx: Autograd context with saved tensors from the forward pass.
            grad_output (torch.Tensor): Gradient of the output tensor.

        Returns:
            torch.Tensor: The computed gradient for the input tensor.
            None: Placeholder for `h`, which does not require gradients.
        """
        xor1, xor2 = ctx.saved_tensors
        h = ctx.h
        T = xor1.size(1)

        # Kernel density estimation
        coeff = torch.mean(xor2 - xor1, dim=1) / (2 * h)
        coeff = coeff.unsqueeze(1).repeat(1, T, 1)  # Expand to match original shape
        grad = coeff * grad_output  # Apply chain rule

        return grad, None
        

class CrossingStatistic(torch.autograd.Function):
    """Crossing activation function (statistic version).

    This version of the Crossing activation function applies noise to the input tensor
    and computes the output and gradient in a statistical manner. The function 
    binarizes the input at thresholds `+h` and `-h`, applies an XOR operation along 
    the simulated T dimension, and then computes the expected values.

    Reference:
        Ikemoto, Neurocomputing 2021.

    Attributes:
        None (Stateless function).
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, h: float = 0.05, numT: int = 100,
                noise_like: callable = noise.gaussian_noise_like(0.0, 1.0)) -> torch.Tensor:
        """Computes the forward pass of the Crossing activation function with noise.

        This function expands the input along a simulated temporal dimension (T), 
        adds noise, and applies the Crossing activation function. The final output 
        is obtained by computing the expectation over T.

        Args:
            ctx: Autograd context to save tensors for backward computation.
            input (torch.Tensor): Input tensor of shape `[N, D]`.
            h (float, optional): Threshold value for binarization. Defaults to 0.05.
                Must be positive.
            numT (int, optional): Number of samples used for statistical computation. 
                Defaults to 100.
            noise_like (callable, optional): A function generating noise tensors 
                of the same shape as `input`. Defaults to Gaussian noise.

        Returns:
            torch.Tensor: The transformed tensor of shape `[N, D]`, where each 
            element represents the expectation of the XOR operation results.

        Raises:
            ValueError: If `h` is not positive.
            TypeError: If `noise_like` is not callable.
        """
        if h <= 0:
            raise ValueError("Threshold `h` must be positive.")
        if not callable(noise_like):
            raise TypeError("`noise_like` must be a callable function.")

        h = abs(h)  # Ensure h > 0
        input = input.unsqueeze(1).repeat(1, numT, 1)  # Expand along T dimension
        input = input + noise_like(input)  # Add noise

        bin1 = (input > h).float()  # Binarization with threshold +h
        bin2 = (input > -h).float()  # Binarization with threshold -h

        # Compute XOR along T dimension and take the expectation
        xor1 = bin1[:, 1:, :] - bin1[:, :-1, :]
        xor1 = torch.abs(xor1)  # XOR from indices 0 to T-2
        xor1_last = torch.abs(bin1[:, -1, :] - bin1[:, 0, :])
        xor1 = torch.cat([xor1, xor1_last.unsqueeze(1)], dim=1)
        xor1 = torch.mean(xor1, dim=1)  # Compute expectation

        xor2 = bin2[:, 1:, :] - bin2[:, :-1, :]
        xor2 = torch.abs(xor2)  # XOR from indices 0 to T-2
        xor2_last = torch.abs(bin2[:, -1, :] - bin2[:, 0, :])
        xor2 = torch.cat([xor2, xor2_last.unsqueeze(1)], dim=1)
        xor2 = torch.mean(xor2, dim=1)  # Compute expectation

        # Save tensors for backward computation
        ctx.save_for_backward(xor1, xor2)
        ctx.h = h

        return (xor2 + xor1) / 2.0

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Computes the backward pass of the Crossing activation function with noise.

        The gradient is computed using kernel density estimation with a uniform kernel
        of width `2h`.

        Args:
            ctx: Autograd context with saved tensors from the forward pass.
            grad_output (torch.Tensor): Gradient of the output tensor.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The computed gradient for the input tensor.
                - None: Placeholder for `h`, which does not require gradients.
                - None: Placeholder for `numT`, which does not require gradients.
                - None: Placeholder for `noise_like`, which does not require gradients.
        """
        xor1, xor2 = ctx.saved_tensors
        h = ctx.h

        # Kernel density estimation
        coeff = (xor2 - xor1) / (2 * h)
        grad = coeff * grad_output  # Apply chain rule

        return grad, None, None, None


class CrossingAnalytic(torch.autograd.Function):
    """Crossing activation function (analytic version).

    This version analytically computes the expected output and gradient using 
    the probability density function (PDF) and cumulative density function (CDF) 
    of the noise distribution.

    Reference:
        Ikemoto, Neurocomputing 2021.

    Attributes:
        None (Stateless function).
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, 
                noisePDF: callable = noise.gaussian_pdf_torch(0, 1.0), 
                noiseCDF: callable = noise.gaussian_cdf_torch(0, 1.0)) -> torch.Tensor:
        """Computes the forward pass of the Crossing activation function using analytical methods.

        The function evaluates the probability of threshold crossing based on 
        the given probability density function (PDF) and cumulative density function (CDF) 
        of the noise. The expected value of the XOR operation is computed directly.

        Args:
            ctx: Autograd context to save tensors for backward computation.
            input (torch.Tensor): Input tensor of shape `[N, D]`.
            noisePDF (callable, optional): Probability density function of the noise. 
                Defaults to a standard Gaussian PDF.
            noiseCDF (callable, optional): Cumulative density function of the noise. 
                Defaults to a standard Gaussian CDF.

        Returns:
            torch.Tensor: The transformed tensor of shape `[N, D]`, where each element 
            represents the expected value of the XOR operation.

        Raises:
            TypeError: If `noisePDF` or `noiseCDF` is not callable.
        """
        if not callable(noisePDF) or not callable(noiseCDF):
            raise TypeError("`noisePDF` and `noiseCDF` must be callable functions.")

        P = noiseCDF(input)  # Compute cumulative probability
        xor = 2.0 * P * (1 - P)  # Compute expectation

        # Save tensors for backward computation
        p = noisePDF(input)
        ctx.save_for_backward(P, p)

        return xor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Computes the backward pass of the Crossing activation function using analytical methods.

        The gradient is computed using the derivative of the expected XOR function 
        with respect to the input.

        Args:
            ctx: Autograd context with saved tensors from the forward pass.
            grad_output (torch.Tensor): Gradient of the output tensor.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The computed gradient for the input tensor.
                - None: Placeholder for `noisePDF`, which does not require gradients.
                - None: Placeholder for `noiseCDF`, which does not require gradients.
        """
        P, p = ctx.saved_tensors

        # Compute gradient coefficient using the derivative of XOR expectation function
        coeff = (1 - 2 * P) * p
        grad = coeff * grad_output  # Apply chain rule

        return grad, None, None


# =========================================================
# Simple check
# Run `python -m nnn.activation` from outside the nnn directory.
#
# Note)
# CrossingStatistic approaches CrossingAnalytic as numT is increased. 
# 'Forward' approaches it considerably, but 'Backward' does not.
# #ven if numT is increased and h is decreased, due to the error in 
# kernel density estimation of the uniform kernel, the difference is large.
# =========================================================    
if __name__ == "__main__":
    # ------------------------------------------------------------------------------
    print("Crossing")
    N, T = 2, 15
    h = 0.5
    input = torch.randn(N, T, requires_grad=True)
    
    output = Crossing.apply(input, h)
    print("Forward: ", output)
    
    output.backward(torch.ones_like(output))
    print("Backward: ", input.grad)

    # ------------------------------------------------------------------------------
    print("CrossingSample")
    N, T, D = 2, 5, 3
    h = 0.5
    input = torch.randn(N, T, D, requires_grad=True)
    
    output = CrossingSample.apply(input, h)
    print("Forward: ", output)
    
    output.backward(torch.ones_like(output))
    print("Backward: ", input.grad)

    # ------------------------------------------------------------------------------
    print("CrossingStatistic")
    input = torch.randn(N, D, requires_grad=True)
    print("Input:", input)
    
    output = CrossingStatistic.apply(input,0.05,10000, noise.gaussian_noise_like(0.0,1.0))
    print("Forward: ", output)
    
    output.backward(torch.ones_like(output))
    print("Backward: ", input.grad)

    # ------------------------------------------------------------------------------
    print("CrossingAnalytic")
    import math
    print("Input:", input)
    
    output = CrossingAnalytic.apply(input, noise.gaussian_pdf_torch(0.0, 1.0), noise.gaussian_cdf_torch(0.0, 1.0))
    print("Forward: ", output)
    
    output.backward(torch.ones_like(output))
    print("Backward: ", input.grad)