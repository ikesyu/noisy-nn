"""Two functions in one network — separated by noise type.

The same shared weight matrices encode two distinct input-output mappings
simultaneously.  The noise distribution type selects which mapping is
computed at runtime:

    Gaussian noise  →  GaussianCrossingAnalyticLayer  →  learns sin(x)
    Uniform  noise  →  ParabolicCrossingAnalyticLayer →  learns cos(x)

No subnetwork gating, no node masking.  All linear layers are fully shared.
Only the activation nonlinearity differs between the two modes.

Training simultaneously minimises both losses:

    L = MSE(net(x, 'gaussian'), sin(x))  +  MSE(net(x, 'uniform'), cos(x))

Run from the project root:

    python examples/regression_two_functions_noisetype.py
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

from nnn import layer, model


# ---------------------------------------------------------------------------
# Dual-noise model
# ---------------------------------------------------------------------------

class DualNoiseNNN(nn.Module):
    """NNN with shared weights and two noise-type modes.

    Linear layers are shared between both modes.  The activation nonlinearity
    switches between GaussianCrossingAnalytic and ParabolicCrossingAnalytic
    depending on the `noise_type` argument passed to forward().

    Args:
        structure:    Layer sizes [D_in, H1, ..., Hk, D_out], len >= 3.
        std:          Standard deviation of the Gaussian noise mode.
        max_radius:   Half-width of the uniform noise mode (recruitment=1.0).
        output_bias:  Whether the output linear layer has a bias term.
    """

    def __init__(self, structure: list = [1, 50, 50, 1], std: float = 0.5,
                 max_radius: float = 1.0, output_bias: bool = False):
        if len(structure) < 3:
            raise ValueError("structure must have at least 3 elements.")
        super(DualNoiseNNN, self).__init__()
        self.structure = structure
        n_hidden = len(structure) - 2

        # Shared linear layers
        self.fcs = nn.ModuleList([
            nn.Linear(pre, post,
                      bias=(output_bias if idx == len(structure) - 2 else True))
            for idx, (pre, post) in enumerate(zip(structure, structure[1:]))
        ])

        # Gaussian mode: analytic crossing with Gaussian noise
        self.gaussian_crossings = nn.ModuleList([
            layer.GaussianCrossingAnalyticLayer(std=std)
            for _ in range(n_hidden)
        ])

        # Uniform mode: analytic crossing with bounded uniform noise
        # (parabolic response = exact expectation for Uniform(-radius, radius))
        self.uniform_crossings = nn.ModuleList([
            layer.ParabolicCrossingAnalyticLayer(
                recruitment=1.0, center=0.0, max_radius=max_radius)
            for _ in range(n_hidden)
        ])

    def forward(self, x: torch.Tensor, noise_type: str = 'gaussian') -> torch.Tensor:
        """Forward pass.

        Args:
            x:          Input tensor of shape [N, D_in].
            noise_type: 'gaussian' or 'uniform'.  Selects which activation
                        nonlinearity is applied at each hidden layer.

        Returns:
            Output tensor of shape [N, D_out].
        """
        if noise_type == 'gaussian':
            crossings = self.gaussian_crossings
        elif noise_type == 'uniform':
            crossings = self.uniform_crossings
        else:
            raise ValueError(f"noise_type must be 'gaussian' or 'uniform', got {noise_type!r}")

        n_hidden = len(self.structure) - 2
        for i in range(len(self.structure) - 1):
            x = self.fcs[i](x)
            if i < n_hidden:
                x = crossings[i](x)
        return x


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_dataset(n_points: int = 1000):
    x = np.linspace(-2 * np.pi, 2 * np.pi, n_points).reshape(-1, 1)
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(np.sin(x), dtype=torch.float32),
        torch.tensor(np.cos(x), dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(net: nn.Module, x: torch.Tensor,
          sin_y: torch.Tensor, cos_y: torch.Tensor,
          epochs: int, lr: float, print_every: int):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    losses_sin, losses_cos, losses_total = [], [], []

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
        losses_total.append(float(loss.detach()))

        if print_every > 0 and (epoch == 1 or epoch % print_every == 0):
            print(f"  epoch {epoch:6d} | "
                  f"sin loss {loss_sin.item():.4e}  "
                  f"cos loss {loss_cos.item():.4e}  "
                  f"total {loss.item():.4e}")

    return losses_sin, losses_cos, losses_total


@torch.no_grad()
def evaluate(net, x, noise_type):
    net.eval()
    return net(x, noise_type).cpu().numpy().ravel()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two functions in one NNN, separated by noise type"
    )
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=50)
    parser.add_argument("--std", type=float, default=0.5,
                        help="Gaussian noise std (default: 0.5)")
    parser.add_argument("--max-radius", type=float, default=1.0,
                        help="Uniform noise half-width (default: 1.0)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    x, sin_y, cos_y = make_dataset()
    x, sin_y, cos_y = x.to(device), sin_y.to(device), cos_y.to(device)

    structure = [1, args.hidden_dim, args.hidden_dim, 1]

    net = DualNoiseNNN(
        structure=structure,
        std=args.std,
        max_radius=args.max_radius,
        output_bias=False,
    ).to(device)

    print(f"Structure : {structure}")
    print(f"Gaussian std={args.std}  →  sin(x)")
    print(f"Uniform radius={args.max_radius}  →  cos(x)")
    print(f"Shared parameters: {sum(p.numel() for p in net.fcs.parameters())}\n")

    losses_sin, losses_cos, losses_total = train(
        net, x, sin_y, cos_y,
        epochs=args.epochs, lr=args.lr, print_every=args.print_every,
    )

    # Quantitative check
    x_np   = x.cpu().numpy().ravel()
    sin_np = sin_y.cpu().numpy().ravel()
    cos_np = cos_y.cpu().numpy().ravel()
    y_gauss  = evaluate(net, x, 'gaussian')
    y_uniform = evaluate(net, x, 'uniform')

    rmse_sin = float(np.sqrt(np.mean((y_gauss  - sin_np) ** 2)))
    rmse_cos = float(np.sqrt(np.mean((y_uniform - cos_np) ** 2)))
    cross_sin = float(np.sqrt(np.mean((y_uniform - sin_np) ** 2)))
    cross_cos = float(np.sqrt(np.mean((y_gauss  - cos_np) ** 2)))

    print(f"\n{'':=<60}")
    print("Final RMSE")
    print(f"  Gaussian  mode → sin target : {rmse_sin:.5f}  (intended)")
    print(f"  Uniform   mode → cos target : {rmse_cos:.5f}  (intended)")
    print(f"  Gaussian  mode → cos target : {cross_cos:.5f}  (cross-check)")
    print(f"  Uniform   mode → sin target : {cross_sin:.5f}  (cross-check)")
    print(f"{'':=<60}")

    if args.no_plot:
        return

    # ------------------------------------------------------------------
    # Figure 1 – Predictions
    # ------------------------------------------------------------------
    fig1, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Left: Gaussian mode
    ax = axes[0]
    ax.plot(x_np, sin_np, 'k-', lw=2.5, label='target  sin(x)')
    ax.plot(x_np, y_gauss, lw=1.8, label='Gaussian mode')
    ax.set_title('Gaussian noise  →  sin(x)', fontsize=10)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(); ax.grid(alpha=0.3)

    # Middle: Uniform mode
    ax = axes[1]
    ax.plot(x_np, cos_np, 'k-', lw=2.5, label='target  cos(x)')
    ax.plot(x_np, y_uniform, lw=1.8, color='C1', label='Uniform mode')
    ax.set_title('Uniform noise  →  cos(x)', fontsize=10)
    ax.set_xlabel('x')
    ax.legend(); ax.grid(alpha=0.3)

    # Right: Both on one plot
    ax = axes[2]
    ax.plot(x_np, sin_np,   'k-',  lw=2.5, label='sin(x)')
    ax.plot(x_np, cos_np,   'k--', lw=2.5, label='cos(x)')
    ax.plot(x_np, y_gauss,  lw=1.8,        label=f'Gauss (RMSE={rmse_sin:.3f})')
    ax.plot(x_np, y_uniform, lw=1.8, color='C1',
            label=f'Uniform (RMSE={rmse_cos:.3f})')
    ax.set_title('Both modes — shared weights', fontsize=10)
    ax.set_xlabel('x')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig1.suptitle(
        f'Two functions in one NNN via noise-type switching  '
        f'(hidden={args.hidden_dim}, std={args.std}, radius={args.max_radius}, '
        f'epochs={args.epochs})',
        fontsize=10,
    )
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Figure 2 – Training loss
    # ------------------------------------------------------------------
    fig2, ax = plt.subplots(figsize=(8, 4))
    epochs_ax = np.arange(1, args.epochs + 1)
    ax.plot(epochs_ax, losses_sin,   label='sin loss  (Gaussian mode)', alpha=0.85)
    ax.plot(epochs_ax, losses_cos,   label='cos loss  (Uniform mode)',  alpha=0.85)
    ax.plot(epochs_ax, losses_total, label='total loss', lw=2, color='k', alpha=0.6)
    ax.set_yscale('log')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE loss')
    ax.set_title('Training loss — dual-noise NNN')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
