"""Train Parabolic and Hat NNN variants on a sine curve.

Put this file in your project's `examples/` directory and run it from the
project root:

    python examples/train_sine_parabolic_hat.py

The script assumes the project has this structure:

    project_root/
      nnn/
        activation.py
        layer.py
        model.py
        noise.py
      examples/
        train_sine_parabolic_hat.py

It uses the added classes:
    - nnn.model.SimpleNNNParabolicAnalytic
    - nnn.model.SimpleNNNHatApprox

If your local class is named `SimpleNNNHatApproxAnalytic`, the script will use
that name automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.append("../")

from nnn import noise #import nnn.noise as noise
from nnn import activation
from nnn import layer
from nnn import model


# Make `from nnn import model` work when this file is placed in examples/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def resolve_hat_model_class():
    """Return the Hat model class, accepting either naming convention."""
    if hasattr(model, "SimpleNNNHatApproxAnalytic"):
        return model.SimpleNNNHatApproxAnalytic
    if hasattr(model, "SimpleNNNHatApprox"):
        return model.SimpleNNNHatApprox
    raise AttributeError(
        "Could not find SimpleNNNHatApproxAnalytic or SimpleNNNHatApprox in nnn.model."
    )


def make_dataset(n_points: int = 1000):
    """Create y = sin(x) on [-2π, 2π]."""
    x = np.linspace(-2 * np.pi, 2 * np.pi, n_points).reshape(-1, 1)
    y = np.sin(x)
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


def make_recruitment(hidden_dim: int, mode: str, device: torch.device):
    """Create a 1D recruitment/noise-field vector for one hidden layer.

    The value plays the same conceptual role as the old `std` field: units with
    recruitment=0 are detached, while larger values expand the local active
    region of Parabolic/Hat units.
    """
    if mode == "full":
        rec = torch.ones(hidden_dim, device=device)
    elif mode == "half":
        rec = torch.zeros(hidden_dim, device=device)
        rec[: hidden_dim // 2] = 1.0
    elif mode == "ramp":
        rec = torch.linspace(0.0, 1.0, hidden_dim, device=device)
    elif mode == "two_blocks":
        rec = torch.zeros(hidden_dim, device=device)
        q = hidden_dim // 4
        rec[:q] = 1.0
        rec[2 * q : 3 * q] = 1.0
    else:
        raise ValueError(f"Unknown recruitment mode: {mode}")
    return rec


def train_one(
    net: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    recruitments: Optional[list[torch.Tensor]],
    epochs: int,
    lr: float,
    print_every: int,
):
    """Train one network and return loss history."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    losses = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = net(x, recruitments=recruitments)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

        if print_every > 0 and (epoch == 1 or epoch % print_every == 0):
            print(f"epoch {epoch:6d} | loss {loss.item():.6e}")

    return losses


@torch.no_grad()
def evaluate(net: nn.Module, x: torch.Tensor, recruitments: Optional[list[torch.Tensor]]):
    net.eval()
    return net(x, recruitments=recruitments).detach().cpu().numpy()


def build_models(args, device: torch.device):
    """Construct Parabolic and Hat models with matched structure."""
    structure = [1, args.hidden_dim, args.hidden_dim, 1]

    parabolic_net = model.SimpleNNNParabolicAnalytic(
        structure=structure,
        recruitment=1.0,
        center=0.0,
        max_radius=args.max_radius,
        output_bias=False,
    ).to(device)

    HatModel = resolve_hat_model_class()
    hat_kwargs = dict(
        structure=structure,
        recruitment=1.0,
        center=0.0,
        max_radius=args.max_radius,
        output_bias=False,
    )
    # `normalized` exists for SimpleNNNHatApprox; keep this optional for future variants.
    try:
        hat_net = HatModel(**hat_kwargs, normalized=True).to(device)
    except TypeError:
        hat_net = HatModel(**hat_kwargs).to(device)

    return parabolic_net, hat_net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=25)
    parser.add_argument("--max-radius", type=float, default=1.5)
    parser.add_argument(
        "--recruitment-mode",
        type=str,
        default="full",
        choices=["full", "half", "ramp", "two_blocks"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    x, y = make_dataset()
    x = x.to(device)
    y = y.to(device)

    rec1 = make_recruitment(args.hidden_dim, args.recruitment_mode, device)
    rec2 = make_recruitment(args.hidden_dim, args.recruitment_mode, device)
    recruitments = [rec1, rec2]

    parabolic_net, hat_net = build_models(args, device)

    print("\n=== Training SimpleNNNParabolicAnalytic ===")
    parabolic_losses = train_one(
        parabolic_net,
        x,
        y,
        recruitments=recruitments,
        epochs=args.epochs,
        lr=args.lr,
        print_every=args.print_every,
    )

    print("\n=== Training Hat model ===")
    hat_losses = train_one(
        hat_net,
        x,
        y,
        recruitments=recruitments,
        epochs=args.epochs,
        lr=args.lr,
        print_every=args.print_every,
    )

    print("\nFinal losses")
    print(f"  Parabolic: {parabolic_losses[-1]:.6e}")
    print(f"  Hat      : {hat_losses[-1]:.6e}")

    if args.no_plot:
        return

    x_np = x.detach().cpu().numpy().ravel()
    y_np = y.detach().cpu().numpy().ravel()
    y_para = evaluate(parabolic_net, x, recruitments).ravel()
    y_hat = evaluate(hat_net, x, recruitments).ravel()

    out_dir = PROJECT_ROOT / "examples" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(x_np, y_np, label="target sin(x)", linewidth=2.5)
    plt.plot(x_np, y_para, label="Parabolic analytical")
    plt.plot(x_np, y_hat, label="Hat approximation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sine approximation with Parabolic and Hat NNN variants")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = out_dir / "sine_parabolic_hat_prediction.png"
#    plt.savefig(fig_path, dpi=200)

    plt.figure(figsize=(8, 5))
    plt.plot(parabolic_losses, label="Parabolic analytical")
    plt.plot(hat_losses, label="Hat approximation")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Training loss")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    loss_path = out_dir / "sine_parabolic_hat_loss.png"
#    plt.savefig(loss_path, dpi=200)

#    print(f"\nSaved figures:")
#    print(f"  {fig_path}")
#    print(f"  {loss_path}")
    plt.show()


if __name__ == "__main__":
    main()
