import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../")

from nnn import noise #import nnn.noise as noise
from nnn import activation
from nnn import layer
from nnn import model
from nnn.model import SimpleNNNBase
import torch.nn.functional as F
import math
from typing import Sequence
from dataclasses import dataclass

class NNNRawSampleRegressor(nn.Module):
    """Minimal readout: y_samples = Linear(SimpleNNNBase(x)).

    This is the most direct implementation. Mean and variance are both computed
    from the same output samples, so they are trainable but coupled.
    """

    def __init__(self, structure: Sequence[int] = (1, 64, 64), std: float = 0.6,
                 h: float = 0.15, t: int = 64):
        super().__init__()
        self.base = SimpleNNNBase(structure=structure, std=std, h=h, t=t)
        self.readout = nn.Linear(self.base.out_features, 1, bias=True)

    def forward(self, x: torch.Tensor):
        z_samples = self.base(x)                 # [N, T, H]
        y_samples = self.readout(z_samples)      # [N, T, 1]
        y_mean = y_samples.mean(dim=1)           # [N, 1]
        y_var = y_samples.var(dim=1, unbiased=False)
        return y_mean, y_var, y_samples


class NNNMeanVarianceRegressor(nn.Module):
    """Separated mean/variance readout for stable variance learning.

    The stochastic base supplies sample-level fluctuations. The final distribution is

        y_sample = mu(x) + sigma(x) * eps_sample(x)

    where eps_sample is a zero-mean, unit-variance stochastic residual extracted from
    the Sample-level base. This keeps the output sample distribution while making
    mean and variance much easier to train independently.
    """

    def __init__(self, structure: Sequence[int] = (1, 64, 64), std: float = 0.6,
                 h: float = 0.15, t: int = 64, eps: float = 1e-6):
        super().__init__()
        self.base = SimpleNNNBase(structure=structure, std=std, h=h, t=t)
        hdim = self.base.out_features
        self.mean_head = nn.Linear(hdim, 1, bias=True)
        self.log_std_head = nn.Linear(hdim, 1, bias=True)
        self.residual_head = nn.Linear(hdim, 1, bias=False)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        z_samples = self.base(x)                       # [N, T, H]
        z_mean = z_samples.mean(dim=1)                 # [N, H]

        mu = self.mean_head(z_mean)                    # [N, 1]
        sigma = F.softplus(self.log_std_head(z_mean)) + self.eps

        r = self.residual_head(z_samples)              # [N, T, 1]
        r = r - r.mean(dim=1, keepdim=True)
        r_std = torch.sqrt(r.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        eps_samples = r / r_std                        # approximately zero-mean/unit-var

        y_samples = mu.unsqueeze(1) + sigma.unsqueeze(1) * eps_samples
        y_mean = y_samples.mean(dim=1)
        y_var = y_samples.var(dim=1, unbiased=False)
        return y_mean, y_var, y_samples


def make_dataset(n: int = 120, vmin: float = 0.005, vmax: float = 0.12, mode: str = "peak"):
    x = torch.linspace(-2.0 * math.pi, 2.0 * math.pi, n).unsqueeze(1)
    y = torch.sin(x)

    if mode == "peak":
        # y=1 or y=-1 near the peaks: large variance
        v = vmin + (vmax - vmin) * torch.abs(y) ** 2
    elif mode == "zero":
        # y=0 near the zero-crossings: large variance
        v = vmin + (vmax - vmin) * (1.0 - torch.abs(y)) ** 2
    else:
        raise ValueError("mode must be 'peak' or 'zero'.")
    return x, y, v


@dataclass
class TrainResult:
    x: torch.Tensor
    y_target: torch.Tensor
    v_target: torch.Tensor
    y_mean: torch.Tensor
    y_var: torch.Tensor
    losses: list


def train_one(mode: str, seed: int, epochs: int = 600) -> TrainResult:
    torch.manual_seed(seed)
    x, y_target, v_target = make_dataset(mode=mode)

    model = NNNMeanVarianceRegressor(
        structure=(1, 48, 48),
        std=0.6,
        h=0.15,
        t=32,
    )
    optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_mean, y_var, _ = model(x)

        loss_mean = F.mse_loss(y_mean, y_target)
        loss_var = F.mse_loss(torch.log(y_var + 1e-6), torch.log(v_target + 1e-6))
        loss = loss_mean + 0.1 * loss_var

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    # Evaluation with more samples T for smoother estimated variance.
    with torch.no_grad():
        old_t = model.base.sampled_layer.numT
        model.base.sampled_layer.numT = 128
        y_mean, y_var, _ = model(x)
        model.base.sampled_layer.numT = old_t

    return TrainResult(
        x=x.detach(),
        y_target=y_target.detach(),
        v_target=v_target.detach(),
        y_mean=y_mean.detach(),
        y_var=y_var.detach(),
        losses=losses,
    )


def plot_results(result_peak: TrainResult, result_zero: TrainResult, path: str = "nnn_meanvar_result.png"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)

    rows = [
        (0, result_peak, "large variance near y = ±1"),
        (1, result_zero, "large variance near y = 0"),
    ]

    for row, res, title in rows:
        x = res.x.squeeze().numpy()
        y = res.y_target.squeeze().numpy()
        vt = res.v_target.squeeze().numpy()
        ym = res.y_mean.squeeze().numpy()
        yv = res.y_var.squeeze().numpy()
        ys = yv ** 0.5

        axes[row, 0].plot(x, y, "--", label="target mean: sin(x)")
        axes[row, 0].plot(x, ym, label="learned mean")
        axes[row, 0].fill_between(x, ym - ys, ym + ys, alpha=0.25, label="learned ± std")
        axes[row, 0].set_title(f"Mean: {title}")
        axes[row, 0].legend()

        axes[row, 1].plot(x, vt, "--", label="target variance")
        axes[row, 1].plot(x, yv, label="learned variance")
        axes[row, 1].set_title(f"Variance: {title}")
        axes[row, 1].legend()

    for ax in axes[-1, :]:
        ax.set_xlabel("x")
    plt.tight_layout()
    #plt.savefig(path, dpi=180)
    plt.show()


if __name__ == "__main__":
    peak = train_one(mode="peak", seed=0, epochs=600)
    zero = train_one(mode="zero", seed=1, epochs=600)
    plot_results(peak, zero)
