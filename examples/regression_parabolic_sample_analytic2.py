"""Output-noise characterisation: T and radius sweeps.

Trains SimpleNNNUniformSample across a grid of T (number of Monte-Carlo samples)
and radius (uniform-noise half-width) values, then measures how much the
(stochastic) output fluctuates across repeated forward passes.

SimpleNNNParabolicAnalytic is used as the zero-noise analytic reference.

Two sweeps
----------
  Sweep 1  Fix radius, vary T   → shows σ_out ∝ 1/√T
  Sweep 2  Fix T, vary radius   → shows σ_out ∝ r  (linear)

Run from the project root:

    python examples/regression_parabolic_sample_analytic2.py

Key arguments:

    --t-values        Comma-separated list of T to sweep (default: 16,64,256)
    --radius-values   Comma-separated list of radii to sweep (default: 0.25,0.5,1.0)
    --t-fixed         T used while sweeping radius (default: 64)
    --radius-fixed    radius used while sweeping T (default: 1.0)
    --epochs          Training epochs per model (default: 2000)
    --n-noise-runs    Forward passes used to estimate output std (default: 50)
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
# Dataset / train / evaluate
# ---------------------------------------------------------------------------

def make_dataset(n_points: int = 1000):
    x = np.linspace(-2 * np.pi, 2 * np.pi, n_points).reshape(-1, 1)
    y = np.sin(x)
    return (torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))


def train_one(net, x, y, epochs, lr, label="", print_every=0):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    losses = []
    for epoch in range(1, epochs + 1):
        net.train()
        optimizer.zero_grad()
        loss = criterion(net(x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if print_every > 0 and (epoch == 1 or epoch % print_every == 0):
            print(f"    [{label}] epoch {epoch:5d} | loss {loss.item():.4e}")
    return losses


@torch.no_grad()
def get_prediction(net, x):
    net.eval()
    return net(x).squeeze(-1).cpu().numpy()


def measure_output_noise(net, x, n_runs: int = 50) -> float:
    """Estimate output std by running the stochastic model n_runs times.

    For the analytic model (deterministic) this will return ~0.
    """
    preds = []
    with torch.no_grad():
        for _ in range(n_runs):
            preds.append(net(x).squeeze(-1).cpu())
    preds = torch.stack(preds, dim=0)       # [n_runs, N]
    return float(preds.std(dim=0).mean())   # mean std across x


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_prediction_row(axes, x_np, y_np, y_analytic,
                         nets, labels, final_losses, row_ylabel):
    """Fill one row of the prediction grid."""
    x_tensor = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float32)

    # Column 0: analytic reference
    ax = axes[0]
    ax.plot(x_np, y_np, 'k-', lw=2.5, label='target')
    ax.plot(x_np, y_analytic, lw=1.8, label='analytic')
    ax.set_ylabel(row_ylabel, fontsize=9)
    ax.set_title('Analytic (ref)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Remaining columns: sample models
    for col, (net, lbl, fl) in enumerate(zip(nets, labels, final_losses), 1):
        y_pred = get_prediction(net, x_tensor)
        ax = axes[col]
        ax.plot(x_np, y_np, 'k-', lw=2.5)
        ax.plot(x_np, y_analytic, lw=1.5, alpha=0.45, label='analytic')
        ax.plot(x_np, y_pred, lw=1.2, label='sample')
        ax.set_title(f'{lbl}\n(loss={fl:.2e})', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=25)
    parser.add_argument("--h", type=float, default=0.1,
                        help="Crossing threshold h for sample models")
    parser.add_argument("--t-values", type=str, default="16,64,256",
                        help="Comma-separated T values for sweep 1")
    parser.add_argument("--radius-values", type=str, default="0.25,0.5,1.0",
                        help="Comma-separated radius values for sweep 2")
    parser.add_argument("--t-fixed", type=int, default=64,
                        help="T used while sweeping radius")
    parser.add_argument("--radius-fixed", type=float, default=1.0,
                        help="radius used while sweeping T")
    parser.add_argument("--n-noise-runs", type=int, default=50,
                        help="Forward passes used to estimate output std")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=0,
                        help="Print loss every N epochs (0 = silent)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    t_values = [int(v) for v in args.t_values.split(",")]
    radius_values = [float(v) for v in args.radius_values.split(",")]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    x, y = make_dataset()
    x, y = x.to(device), y.to(device)
    x_np = x.cpu().numpy().ravel()
    y_np = y.cpu().numpy().ravel()

    structure = [1, args.hidden_dim, args.hidden_dim, 1]

    # ------------------------------------------------------------------
    # Analytic reference
    # ------------------------------------------------------------------
    print("=== Analytic reference ===")
    analytic_net = model.SimpleNNNParabolicAnalytic(
        structure=structure, recruitment=1.0, center=0.0,
        max_radius=args.radius_fixed, output_bias=False,
    ).to(device)
    train_one(analytic_net, x, y, args.epochs, args.lr,
              label="analytic", print_every=args.print_every)
    y_analytic = get_prediction(analytic_net, x)
    noise_analytic = measure_output_noise(analytic_net, x, args.n_noise_runs)

    # ------------------------------------------------------------------
    # Sweep 1: vary T, fix radius
    # ------------------------------------------------------------------
    print(f"\n=== Sweep 1: vary T  (radius={args.radius_fixed}) ===")
    nets_t, losses_t = [], []
    for t in t_values:
        label = f"T={t}"
        print(f"  Training {label} ...")
        net = model.SimpleNNNUniformSample(
            structure=structure, radius=args.radius_fixed, center=0.0,
            h=args.h, t=t, output_bias=False,
        ).to(device)
        losses = train_one(net, x, y, args.epochs, args.lr,
                           label=label, print_every=args.print_every)
        nets_t.append(net)
        losses_t.append(losses[-1])

    # ------------------------------------------------------------------
    # Sweep 2: vary radius, fix T
    # ------------------------------------------------------------------
    print(f"\n=== Sweep 2: vary radius  (T={args.t_fixed}) ===")
    nets_r, losses_r = [], []
    for r in radius_values:
        label = f"r={r}"
        print(f"  Training {label} ...")
        net = model.SimpleNNNUniformSample(
            structure=structure, radius=r, center=0.0,
            h=args.h, t=args.t_fixed, output_bias=False,
        ).to(device)
        losses = train_one(net, x, y, args.epochs, args.lr,
                           label=label, print_every=args.print_every)
        nets_r.append(net)
        losses_r.append(losses[-1])

    # ------------------------------------------------------------------
    # Noise measurement
    # ------------------------------------------------------------------
    print(f"\nMeasuring output noise ({args.n_noise_runs} runs each) ...")
    noise_t = [measure_output_noise(net, x, args.n_noise_runs) for net in nets_t]
    noise_r = [measure_output_noise(net, x, args.n_noise_runs) for net in nets_r]

    print(f"\nAnalytic reference noise std : {noise_analytic:.5f}")
    print(f"\nNoise vs T  (radius={args.radius_fixed}):")
    for t, n in zip(t_values, noise_t):
        print(f"  T={t:4d} : {n:.5f}")
    print(f"\nNoise vs radius  (T={args.t_fixed}):")
    for r, n in zip(radius_values, noise_r):
        print(f"  r={r:.2f} : {n:.5f}")

    if args.no_plot:
        return

    # ------------------------------------------------------------------
    # Figure 1 – Prediction grid
    # ------------------------------------------------------------------
    n_sample_cols = max(len(t_values), len(radius_values))
    n_cols = n_sample_cols + 1  # +1 for analytic

    fig1, axes1 = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 8),
                               sharey='row', sharex=True)

    _plot_prediction_row(
        axes1[0], x_np, y_np, y_analytic,
        nets_t,
        [f"T={t}" for t in t_values],
        losses_t,
        row_ylabel=f"radius={args.radius_fixed}  (vary T)",
    )
    _plot_prediction_row(
        axes1[1], x_np, y_np, y_analytic,
        nets_r,
        [f"r={r}" for r in radius_values],
        losses_r,
        row_ylabel=f"T={args.t_fixed}  (vary radius)",
    )

    for col in range(n_cols):
        axes1[1, col].set_xlabel("x")

    for row, n_used in enumerate([len(t_values), len(radius_values)]):
        for col in range(n_used + 1, n_cols):
            axes1[row, col].set_visible(False)

    fig1.suptitle(
        f"Sample NNN predictions  (epochs={args.epochs}, h={args.h}, hidden={args.hidden_dim})",
        fontsize=11,
    )
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Figure 2 – Noise quantification
    # ------------------------------------------------------------------
    fig2, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: noise vs T
    t_labels = [str(t) for t in t_values]
    bars = ax_t.bar(t_labels, noise_t, color='steelblue', zorder=3)
    for bar, n in zip(bars, noise_t):
        ax_t.text(bar.get_x() + bar.get_width() / 2,
                  n + max(noise_t) * 0.01,
                  f"{n:.4f}", ha='center', va='bottom', fontsize=9)
    ref_t = noise_t[0] * np.sqrt(t_values[0]) / np.sqrt(np.array(t_values, float))
    ax_t.plot(t_labels, ref_t, 'r--o', label=r'$\propto 1/\sqrt{T}$ (theory)', zorder=4)
    ax_t.axhline(noise_analytic, color='k', linestyle=':', lw=1.5,
                 label=f'analytic ({noise_analytic:.4f})')
    ax_t.set_xlabel("T  (number of Monte-Carlo samples)")
    ax_t.set_ylabel("Output noise std  (mean over x)")
    ax_t.set_title(f"Noise vs T   (radius={args.radius_fixed})")
    ax_t.legend(fontsize=8)
    ax_t.grid(axis='y', alpha=0.35, zorder=0)
    ax_t.set_ylim(bottom=0)

    # Right: noise vs radius
    r_labels = [str(r) for r in radius_values]
    bars = ax_r.bar(r_labels, noise_r, color='coral', zorder=3)
    for bar, n in zip(bars, noise_r):
        ax_r.text(bar.get_x() + bar.get_width() / 2,
                  n + max(noise_r) * 0.01,
                  f"{n:.4f}", ha='center', va='bottom', fontsize=9)
    ref_r = noise_r[-1] * (np.array(radius_values) / radius_values[-1])
    ax_r.plot(r_labels, ref_r, 'r--o', label=r'$\propto r$ (theory)', zorder=4)
    ax_r.axhline(noise_analytic, color='k', linestyle=':', lw=1.5,
                 label=f'analytic ({noise_analytic:.4f})')
    ax_r.set_xlabel("radius  (uniform-noise half-width)")
    ax_r.set_ylabel("Output noise std  (mean over x)")
    ax_r.set_title(f"Noise vs radius   (T={args.t_fixed})")
    ax_r.legend(fontsize=8)
    ax_r.grid(axis='y', alpha=0.35, zorder=0)
    ax_r.set_ylim(bottom=0)

    fig2.suptitle(
        f"Output noise characterisation  (epochs={args.epochs}, h={args.h}, "
        f"hidden={args.hidden_dim}, n_runs={args.n_noise_runs})",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
