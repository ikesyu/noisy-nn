"""Demonstrate layer-wise recruitment/noise-field vectors.

Put this file in your project's `examples/` directory and run it from the
project root:

    python examples/recruitment_field_demo.py

This example shows how the same trained/initialized network changes its output
when different recruitment vectors are passed to Parabolic and Hat activations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append("../")

from nnn import noise #import nnn.noise as noise
from nnn import activation
from nnn import layer
from nnn import model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from nnn import model  # noqa: E402


def resolve_hat_model_class():
    if hasattr(model, "SimpleNNNHatApproxAnalytic"):
        return model.SimpleNNNHatApproxAnalytic
    if hasattr(model, "SimpleNNNHatApprox"):
        return model.SimpleNNNHatApprox
    raise AttributeError(
        "Could not find SimpleNNNHatApproxAnalytic or SimpleNNNHatApprox in nnn.model."
    )


def make_block_recruitment(dim: int, start: int, end: int):
    rec = torch.zeros(dim)
    rec[start:end] = 1.0
    return rec


@torch.no_grad()
def main():
    torch.manual_seed(1)
    np.random.seed(1)

    hidden_dim = 25
    structure = [1, hidden_dim, hidden_dim, 1]
    x = torch.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)

    parabolic_net = model.SimpleNNNParabolicAnalytic(
        structure=structure,
        recruitment=1.0,
        max_radius=1.5,
        output_bias=False,
    )

    HatModel = resolve_hat_model_class()
    try:
        hat_net = HatModel(
            structure=structure,
            recruitment=1.0,
            max_radius=1.5,
            normalized=True,
            output_bias=False,
        )
    except TypeError:
        hat_net = HatModel(
            structure=structure,
            recruitment=1.0,
            max_radius=1.5,
            output_bias=False,
        )

    recruitment_patterns = {
        "all units": [torch.ones(hidden_dim), torch.ones(hidden_dim)],
        "first block": [
            make_block_recruitment(hidden_dim, 0, hidden_dim // 2),
            make_block_recruitment(hidden_dim, 0, hidden_dim // 2),
        ],
        "second block": [
            make_block_recruitment(hidden_dim, hidden_dim // 2, hidden_dim),
            make_block_recruitment(hidden_dim, hidden_dim // 2, hidden_dim),
        ],
        "ramp": [
            torch.linspace(0.0, 1.0, hidden_dim),
            torch.linspace(1.0, 0.0, hidden_dim),
        ],
    }

    out_dir = PROJECT_ROOT / "examples" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for label, recs in recruitment_patterns.items():
        y = parabolic_net(x, recruitments=recs).detach().numpy().ravel()
        plt.plot(x.numpy().ravel(), y, label=label)
    plt.xlabel("x")
    plt.ylabel("network output")
    plt.title("Parabolic model: effect of recruitment vectors")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    parabolic_path = out_dir / "recruitment_demo_parabolic.png"
    #plt.savefig(parabolic_path, dpi=200)

    plt.figure(figsize=(8, 5))
    for label, recs in recruitment_patterns.items():
        y = hat_net(x, recruitments=recs).detach().numpy().ravel()
        plt.plot(x.numpy().ravel(), y, label=label)
    plt.xlabel("x")
    plt.ylabel("network output")
    plt.title("Hat model: effect of recruitment vectors")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    hat_path = out_dir / "recruitment_demo_hat.png"
    #plt.savefig(hat_path, dpi=200)

    print("Saved figures:")
    print(f"  {parabolic_path}")
    print(f"  {hat_path}")
    plt.show()


if __name__ == "__main__":
    main()
