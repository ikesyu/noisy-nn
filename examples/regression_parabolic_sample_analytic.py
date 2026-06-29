"""Compare SimpleNNNParabolicAnalytic (analytic) vs sample-based NNN with uniform noise.

Both models use bounded uniform noise on [-1, 1] at each hidden layer.
SimpleNNNParabolicAnalytic computes the exact expected crossing response analytically:

    y = 0.5 * [1 - ((x - center) / radius)^2]_+

SimpleNNNUniformSample approximates the same quantity via Monte Carlo (T samples).

Run from the project root:

    python examples/regression_parabolic_sample_analytic.py

Optional arguments:

    --epochs       Number of training epochs (default: 3000)
    --lr           Learning rate (default: 3e-4)
    --hidden-dim   Width of each hidden layer (default: 25)
    --radius       Uniform noise half-width, i.e. noise ~ Uniform(-r, r) (default: 1.0)
    --t            Number of Monte Carlo samples for the sample model (default: 64)
    --h            Threshold for the crossing activation in the sample model (default: 0.1)
    --seed         Random seed (default: 0)
    --print-every  Print loss every N epochs, 0 to disable (default: 500)
    --no-plot      Skip the matplotlib plots
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

from nnn import model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_dataset(n_points: int = 1000):
    x = np.linspace(-2 * np.pi, 2 * np.pi, n_points).reshape(-1, 1)
    y = np.sin(x)
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(net: nn.Module, x: torch.Tensor, y: torch.Tensor,
              epochs: int, lr: float, print_every: int,
              forward_kwargs: dict = None) -> list[float]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    forward_kwargs = forward_kwargs or {}
    losses = []

    for epoch in range(1, epochs + 1):
        net.train()
        optimizer.zero_grad()
        pred = net(x, **forward_kwargs)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

        if print_every > 0 and (epoch == 1 or epoch % print_every == 0):
            print(f"  epoch {epoch:6d} | loss {loss.item():.6e}")

    return losses


@torch.no_grad()
def evaluate(net: nn.Module, x: torch.Tensor, forward_kwargs: dict = None) -> np.ndarray:
    net.eval()
    forward_kwargs = forward_kwargs or {}
    return net(x, **forward_kwargs).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parabolic-analytic vs uniform-sample NNN comparison"
    )
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=25)
    parser.add_argument("--radius", type=float, default=1.0,
                        help="Uniform noise half-width (default: 1.0 → noise ~ Uniform(-1, 1))")
    parser.add_argument("--t", type=int, default=64,
                        help="Number of Monte Carlo samples for the sample model")
    parser.add_argument("--h", type=float, default=0.1,
                        help="Crossing threshold h for the sample model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    x, y = make_dataset()
    x, y = x.to(device), y.to(device)

    structure = [1, args.hidden_dim, args.hidden_dim, 1]

    # --- Parabolic analytic model ---
    # max_radius=args.radius, recruitment=1.0  →  radius = args.radius at every unit.
    # This is the exact analytic response for Uniform(-radius, radius) noise.
    parabolic_net = model.SimpleNNNParabolicAnalytic(
        structure=structure,
        recruitment=1.0,
        center=0.0,
        max_radius=args.radius,
        output_bias=False,
    ).to(device)

    # --- Uniform sample model ---
    # Approximates the same activation via T Monte Carlo samples.
    sample_net = model.SimpleNNNUniformSample(
        structure=structure,
        radius=args.radius,
        center=0.0,
        h=args.h,
        t=args.t,
        output_bias=False,
    ).to(device)

    print(f"\n=== Training SimpleNNNParabolicAnalytic (analytic, radius={args.radius}) ===")
    parabolic_losses = train_one(
        parabolic_net, x, y,
        epochs=args.epochs, lr=args.lr, print_every=args.print_every,
    )

    print(f"\n=== Training SimpleNNNUniformSample (T={args.t}, radius={args.radius}, h={args.h}) ===")
    sample_losses = train_one(
        sample_net, x, y,
        epochs=args.epochs, lr=args.lr, print_every=args.print_every,
    )

    print("\nFinal losses")
    print(f"  Parabolic analytic : {parabolic_losses[-1]:.6e}")
    print(f"  Uniform sample     : {sample_losses[-1]:.6e}")

    if args.no_plot:
        return

    x_np = x.detach().cpu().numpy().ravel()
    y_np = y.detach().cpu().numpy().ravel()
    y_para = evaluate(parabolic_net, x).ravel()
    y_samp = evaluate(sample_net, x).ravel()

    # --- Prediction plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(x_np, y_np, label="target sin(x)", linewidth=2.5, color="black")
    ax.plot(x_np, y_para, label="Parabolic analytic", linewidth=1.8)
    ax.plot(x_np, y_samp, label=f"Uniform sample (T={args.t})", linewidth=1.8, linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Predictions")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Loss curve plot ---
    ax = axes[1]
    ax.plot(parabolic_losses, label="Parabolic analytic")
    ax.plot(sample_losses, label=f"Uniform sample (T={args.t})", linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    fig.suptitle(
        f"Parabolic analytic vs uniform sample NNN  "
        f"(radius={args.radius}, h={args.h}, T={args.t}, hidden={args.hidden_dim})"
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
