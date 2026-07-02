"""
examples/regression_node_perturbation_statistic.py

Focused experiment: can SimpleNNNStatistic be trained by finite-difference
node perturbation alone — without any backprop through the activation?

Task
    y = sin(x)   x in [-2pi, 2pi]

Node perturbation update rule (hidden layers)
    For each hidden layer l, sample delta_l ~ N(0, I)  [N, H].
    Compute a scalar learning signal from symmetric finite differences:

        s = (L(d_l + eps*delta_l) - L(d_l - eps*delta_l)) / (2 * eps)

    Update weights:

        W_l -= lr * (s * delta_l)^T @ z_{l-1} / N
        b_l -= lr * mean(s * delta_l, dim=0)

    where z_{l-1} is the clean activation fed into layer l.

Output layer
    Direct MSE gradient for stability — not node perturbation.
    (noted in code comments and printed to console)

Note on stochasticity
    SimpleNNNStatistic draws T Gaussian noise samples per call and averages the
    resulting activations.  Because L_plus and L_minus use independent noise
    draws, the gradient estimate s has additional variance ~std_act / sqrt(T).
    Use larger --eps or --t to improve the signal-to-noise ratio.

Run
    python examples/regression_node_perturbation_statistic.py
    python examples/regression_node_perturbation_statistic.py --epochs 5000 --eps 0.1 --t 50
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from nnn import model


# ============================================================
# Custom forward helpers
# ============================================================

def forward_clean(
    net, x: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Clean forward through SimpleNNNStatistic using the layer's stored std.
    Returns (y [N,1], zs)  where zs[l] is the activation fed into net.fcs[l].
    """
    n_hidden = len(net.structure) - 2
    zs = [x]
    for l in range(n_hidden):
        d  = net.fcs[l](zs[-1])
        zl = net.gaussian_crossing[l](d)   # uses layer's own stored std; no side effect
        zs.append(zl)
    y = net.fcs[-1](zs[-1])
    return y, zs


def forward_perturbed(
    net, x: torch.Tensor,
    deltas: list[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """Forward with eps * delta_l added to each hidden pre-activation."""
    n_hidden = len(net.structure) - 2
    z = x
    for l in range(n_hidden):
        d = net.fcs[l](z) + eps * deltas[l]
        z = net.gaussian_crossing[l](d)    # uses layer's own stored std
    return net.fcs[-1](z)


# ============================================================
# One training step
# ============================================================

def node_perturbation_step(
    net, x: torch.Tensor, y: torch.Tensor,
    eps: float, lr: float,
) -> float:
    """
    One node-perturbation update.
    Returns the clean MSE loss evaluated before the weight update.
    """
    N        = x.shape[0]
    n_hidden = len(net.structure) - 2

    with torch.no_grad():
        # Per-sample, per-neuron random perturbation for each hidden layer
        deltas = [torch.randn(N, net.structure[l + 1]) for l in range(n_hidden)]

        # Clean forward: get current prediction and layer inputs z_{l-1}
        y_pred, zs = forward_clean(net, x)

        # Symmetric perturbed forwards for the finite-difference signal
        y_p = forward_perturbed(net, x, deltas, +eps)
        y_m = forward_perturbed(net, x, deltas, -eps)

        # Scalar learning signal
        s = (F.mse_loss(y_p, y) - F.mse_loss(y_m, y)) / (2.0 * eps)

        # --- Hidden layers: node perturbation weight update ---
        # grad_W_l ~ (s * delta_l)^T @ z_{l-1} / N
        # grad_b_l ~ mean(s * delta_l, dim=0)
        for l in range(n_hidden):
            sig_d = s * deltas[l]                              # [N, H_out]
            net.fcs[l].weight -= lr * (sig_d.T @ zs[l]) / N  # [H_out, H_in]
            net.fcs[l].bias   -= lr * sig_d.mean(0)           # [H_out]

        # --- Output layer: direct readout gradient ---
        # hidden layers updated by node perturbation;
        # output layer uses direct readout gradient for stability
        dL_dy = 2.0 * (y_pred - y) / N                      # [N, 1]
        net.fcs[-1].weight -= lr * (dL_dy.T @ zs[-1])        # [1, H]
        if net.fcs[-1].bias is not None:
            net.fcs[-1].bias -= lr * dL_dy.mean(0)           # [1]

        return F.mse_loss(y_pred, y).item()


# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train SimpleNNNStatistic by node perturbation (no backprop)'
    )
    p.add_argument('--epochs',     type=int,   default=3000,
                   help='Training epochs [3000]')
    p.add_argument('--lr',         type=float, default=1e-2,
                   help='Learning rate [1e-2]')
    p.add_argument('--eps',        type=float, default=0.1,
                   help='Node perturbation magnitude; larger value → higher SNR'
                        ' relative to stochastic activation noise [0.1]')
    p.add_argument('--std',        type=float, default=0.5,
                   help='Gaussian noise std in SimpleNNNStatistic [0.5]')
    p.add_argument('--hidden-dim', type=int,   default=64,
                   help='Units per hidden layer [64]')
    p.add_argument('--t',          type=int,   default=10,
                   help='T (num noise samples) in StatisticLayer [10]')
    p.add_argument('--seed',       type=int,   default=42,
                   help='Random seed [42]')
    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Dataset ---
    N    = 500
    x_np = np.linspace(-2 * math.pi, 2 * math.pi, N).reshape(-1, 1)
    y_np = np.sin(x_np)
    x    = torch.tensor(x_np, dtype=torch.float32)
    y    = torch.tensor(y_np, dtype=torch.float32)

    # --- Model ---
    H      = args.hidden_dim
    struct = [1, H, H, 1]
    net    = model.SimpleNNNStatistic(structure=struct, std=args.std, t=args.t)

    print(f'Model  : SimpleNNNStatistic  structure={struct}  std={args.std}  t={args.t}')
    print(f'Update : node perturbation   eps={args.eps}  lr={args.lr}  epochs={args.epochs}')
    print('         hidden layers → node perturbation')
    print('         output layer  → direct readout gradient (not node perturbation)\n')

    # --- Training ---
    losses   = []
    LOG_STEP = max(1, args.epochs // 20)

    for epoch in range(args.epochs):
        loss = node_perturbation_step(net, x, y, args.eps, args.lr)
        losses.append(loss)
        if epoch % LOG_STEP == 0 or epoch == args.epochs - 1:
            print(f'  epoch {epoch:5d}  loss={loss:.6f}')

    # --- Final predictions ---
    with torch.no_grad():
        # Stochastic prediction: noise active, averaged over T samples
        y_noisy, _ = forward_clean(net, x)

    # Noiseless readout: same weights, std → 0 → deterministic crossing
    net_noiseless = model.SimpleNNNStatistic(structure=struct, std=0.0, t=1)
    net_noiseless.load_state_dict(net.state_dict())
    with torch.no_grad():
        y_noiseless, _ = forward_clean(net_noiseless, x)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(losses, color='steelblue', lw=1.2)
    axes[0].set_yscale('log')
    axes[0].set_title('Training loss  (node perturbation, no backprop)')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('MSE')
    axes[0].grid(alpha=0.3)

    # Predictions
    xr = x_np.ravel()
    axes[1].plot(xr, y_np.ravel(),                  'k-',  lw=2.0,
                 label='target  sin(x)')
    axes[1].plot(xr, y_noisy.numpy().ravel(),   '--',  color='tomato',     lw=1.5,
                 alpha=0.8, label=f'prediction  (std={args.std}, T={args.t})')
    axes[1].plot(xr, y_noiseless.numpy().ravel(), '-', color='darkorange', lw=1.5,
                 label='noiseless readout  (std→0)')
    axes[1].set_title('Final prediction')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f'SimpleNNNStatistic trained by node perturbation  '
        f'(H={H}, std={args.std}, T={args.t}, eps={args.eps}, '
        f'lr={args.lr}, epochs={args.epochs})',
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
