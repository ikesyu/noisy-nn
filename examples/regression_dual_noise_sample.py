"""Two functions in one network — separated by noise type (sample-based).

The sample-based counterpart of regression_two_functions_noisetype.py.

A DualNoiseSampleNNN is trained so that:
    Gaussian noise mode  (α=0)  →  sin(x)
    Uniform  noise mode  (α=1)  →  cos(x)

The activation at each hidden layer uses CrossingSample after adding:
    ε = (1−α)·ε_G + α·ε_U
where ε_G ~ N(0, std²) and ε_U ~ U(−radius, radius).

This differs from the analytic version in two ways:
  1. No closed-form formula: noise is actually sampled each forward pass.
  2. α=0 and α=1 are the pure modes; 0<α<1 adds a convolution of
     scaled Gaussian and scaled Uniform noise (not a discrete mixture).

Run from the project root:

    python examples/regression_dual_noise_sample.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from nnn import activation, layer


# ---------------------------------------------------------------------------
# Mixture crossing layer
# ---------------------------------------------------------------------------

class MixtureCrossingSampleLayer(nn.Module):
    """CrossingSample activation with blended Gaussian + Uniform noise.

    Noise added to each element of the [N, T, D] input:
        ε = (1−α)·ε_G + α·ε_U
        ε_G ~ N(0, std²),  ε_U ~ U(center−radius, center+radius)

    α=0 → pure Gaussian noise
    α=1 → pure Uniform  noise
    """

    def __init__(self, std: float = 0.5, radius: float = 1.0,
                 center: float = 0.0, h: float = 0.1):
        super(MixtureCrossingSampleLayer, self).__init__()
        self.std = std
        self.radius = radius
        self.center = center
        self.h = h

    def forward(self, x: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
        eps_g = torch.randn_like(x) * self.std
        eps_u = (torch.rand_like(x) * 2.0 - 1.0) * self.radius + self.center
        eps = (1.0 - alpha) * eps_g + alpha * eps_u
        return activation.CrossingSample.apply(x + eps, self.h)


# ---------------------------------------------------------------------------
# Dual-noise sample model
# ---------------------------------------------------------------------------

class DualNoiseSampleNNN(nn.Module):
    """Shared-weight NNN with two noise-type modes (sample-based).

    Linear layers (fcs) are shared.  The noise injected at each hidden layer
    switches between pure Gaussian (α=0) and pure Uniform (α=1), or any
    blend in between via forward_blend().

    T Monte-Carlo samples are expanded at the first hidden layer and averaged
    at the output, identical to SimpleNNNUniformSample / SimpleNNNSample.

    Args:
        structure:    [D_in, H1, ..., Hk, D_out], len >= 3.
        std:          Gaussian noise standard deviation.
        max_radius:   Uniform noise half-width.
        center:       Center of the uniform noise distribution.
        h:            Threshold for the Crossing activation.
        t:            Number of Monte-Carlo samples.
        output_bias:  Whether the output linear layer has a bias term.
    """

    def __init__(self, structure: list = [1, 50, 50, 1],
                 std: float = 0.5, max_radius: float = 1.0,
                 center: float = 0.0, h: float = 0.1,
                 t: int = 64, output_bias: bool = False):
        if len(structure) < 3:
            raise ValueError("structure must have at least 3 elements.")
        super(DualNoiseSampleNNN, self).__init__()
        self.structure = structure
        n_hidden = len(structure) - 2

        self.fcs = nn.ModuleList([
            nn.Linear(pre, post,
                      bias=(output_bias if idx == len(structure) - 2 else True))
            for idx, (pre, post) in enumerate(zip(structure, structure[1:]))
        ])
        self.mixture_crossings = nn.ModuleList([
            MixtureCrossingSampleLayer(std=std, radius=max_radius,
                                       center=center, h=h)
            for _ in range(n_hidden)
        ])
        self.sampled_layer = layer.SampleLayer(numT=t)
        self.ensemble_layer = layer.EnsembleMeanLayer()

    def forward(self, x: torch.Tensor, noise_type: str = 'gaussian') -> torch.Tensor:
        """Forward with pure Gaussian (noise_type='gaussian') or Uniform noise."""
        alpha = 0.0 if noise_type == 'gaussian' else 1.0
        return self.forward_blend(x, alpha)

    def forward_blend(self, x: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
        """Forward with blended noise ε = (1−α)·ε_G + α·ε_U at each hidden layer.

        Args:
            x:     Input tensor [N, D_in].
            alpha: Noise blend weight in [0, 1].
        """
        n_hidden = len(self.structure) - 2
        for i in range(len(self.structure) - 1):
            x = self.fcs[i](x)
            if i == 0:
                x = self.sampled_layer(x)
            if i < n_hidden:
                x = self.mixture_crossings[i](x, alpha=alpha)
            if i == n_hidden:
                x = self.ensemble_layer(x)
        return x


# ---------------------------------------------------------------------------
# Dataset / train / evaluate
# ---------------------------------------------------------------------------

def make_dataset(n_points: int = 1000):
    x = np.linspace(-2 * np.pi, 2 * np.pi, n_points).reshape(-1, 1)
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(np.sin(x), dtype=torch.float32),
        torch.tensor(np.cos(x), dtype=torch.float32),
    )


def train(net: nn.Module, x: torch.Tensor,
          sin_y: torch.Tensor, cos_y: torch.Tensor,
          epochs: int, lr: float, print_every: int = 500):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    losses_sin, losses_cos = [], []

    for epoch in range(1, epochs + 1):
        net.train()
        optimizer.zero_grad()
        loss_sin = criterion(net(x, 'gaussian'), sin_y)
        loss_cos = criterion(net(x, 'uniform'),  cos_y)
        loss = loss_sin + loss_cos
        loss.backward()
        optimizer.step()
        losses_sin.append(float(loss_sin.detach()))
        losses_cos.append(float(loss_cos.detach()))

        if print_every > 0 and (epoch == 1 or epoch % print_every == 0):
            print(f"  epoch {epoch:6d} | "
                  f"sin {loss_sin.item():.4e}  cos {loss_cos.item():.4e}  "
                  f"total {loss.item():.4e}")

    return losses_sin, losses_cos


@torch.no_grad()
def evaluate(net: nn.Module, x: torch.Tensor, noise_type: str) -> np.ndarray:
    net.eval()
    return net(x, noise_type).cpu().numpy().ravel()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dual-noise sample NNN: sin + cos in shared weights"
    )
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=50)
    parser.add_argument("--t", type=int, default=64,
                        help="Monte-Carlo samples per input")
    parser.add_argument("--std", type=float, default=0.5,
                        help="Gaussian noise std")
    parser.add_argument("--max-radius", type=float, default=1.0,
                        help="Uniform noise half-width")
    parser.add_argument("--h", type=float, default=0.1,
                        help="Crossing activation threshold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, sin_y, cos_y = make_dataset()
    structure = [1, args.hidden_dim, args.hidden_dim, 1]

    net = DualNoiseSampleNNN(
        structure=structure,
        std=args.std, max_radius=args.max_radius,
        h=args.h, t=args.t, output_bias=False,
    )

    print(f"Structure      : {structure}")
    print(f"T (samples)    : {args.t}")
    print(f"Gaussian std   : {args.std}  →  sin(x)")
    print(f"Uniform radius : {args.max_radius}  →  cos(x)")
    print(f"Shared params  : {sum(p.numel() for p in net.fcs.parameters())}\n")

    losses_sin, losses_cos = train(
        net, x, sin_y, cos_y,
        epochs=args.epochs, lr=args.lr, print_every=args.print_every,
    )

    x_np   = x.numpy().ravel()
    sin_np = sin_y.numpy().ravel()
    cos_np = cos_y.numpy().ravel()

    # Evaluate multiple times to show stochastic spread
    n_eval = 5
    preds_g = np.stack([evaluate(net, x, 'gaussian') for _ in range(n_eval)])
    preds_u = np.stack([evaluate(net, x, 'uniform')  for _ in range(n_eval)])
    y_gauss   = preds_g.mean(axis=0)
    y_uniform = preds_u.mean(axis=0)

    rmse_sin  = float(np.sqrt(np.mean((y_gauss   - sin_np) ** 2)))
    rmse_cos  = float(np.sqrt(np.mean((y_uniform - cos_np) ** 2)))
    cross_cos = float(np.sqrt(np.mean((y_gauss   - cos_np) ** 2)))
    cross_sin = float(np.sqrt(np.mean((y_uniform - sin_np) ** 2)))

    print(f"\n{'':=<62}")
    print("Final RMSE  (mean of 5 forward passes)")
    print(f"  Gaussian mode → sin target : {rmse_sin:.5f}  (intended)")
    print(f"  Uniform  mode → cos target : {rmse_cos:.5f}  (intended)")
    print(f"  Gaussian mode → cos target : {cross_cos:.5f}  (cross-check)")
    print(f"  Uniform  mode → sin target : {cross_sin:.5f}  (cross-check)")
    print(f"{'':=<62}")

    if args.no_plot:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Left: Gaussian mode
    ax = axes[0]
    ax.plot(x_np, sin_np, 'k-', lw=2.5, label='target  sin(x)')
    for p in preds_g:
        ax.plot(x_np, p, lw=0.8, alpha=0.35, color='C0')
    ax.plot(x_np, y_gauss, lw=1.8, color='C0',
            label=f'Gaussian mean (RMSE={rmse_sin:.3f})')
    ax.set_title('Gaussian noise  →  sin(x)', fontsize=10)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Middle: Uniform mode
    ax = axes[1]
    ax.plot(x_np, cos_np, 'k-', lw=2.5, label='target  cos(x)')
    for p in preds_u:
        ax.plot(x_np, p, lw=0.8, alpha=0.35, color='C1')
    ax.plot(x_np, y_uniform, lw=1.8, color='C1',
            label=f'Uniform mean (RMSE={rmse_cos:.3f})')
    ax.set_title('Uniform noise  →  cos(x)', fontsize=10)
    ax.set_xlabel('x')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Right: loss curves
    ax = axes[2]
    epochs_ax = np.arange(1, args.epochs + 1)
    ax.plot(epochs_ax, losses_sin, label='sin loss  (Gaussian mode)', alpha=0.85)
    ax.plot(epochs_ax, losses_cos, label='cos loss  (Uniform  mode)', alpha=0.85)
    ax.set_yscale('log')
    ax.set_xlabel('epoch'); ax.set_ylabel('MSE loss')
    ax.set_title('Training loss')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which='both')

    fig.suptitle(
        f'Dual-noise sample NNN  '
        f'(T={args.t}, hidden={args.hidden_dim}, '
        f'std={args.std}, radius={args.max_radius}, h={args.h})',
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
