"""Animate a smooth noise-type transition in a dual-noise NNN.

A DualNoiseNNN is trained so that:
    Gaussian noise mode  →  sin(x)
    Uniform  noise mode  →  cos(x)

After training, the animation linearly interpolates the outputs of the two modes:

    y(α) = (1 − α) · net(x, 'gaussian') + α · net(x, 'uniform')

α sweeps 0 → 1 → 0 cyclically.  The blended activation function shape is shown
alongside the prediction so the mechanism of the transition is visible.

This file is self-contained; it does not depend on other example scripts.

Run from the project root:

    python examples/animation_dual_noise.py
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
import matplotlib.animation as animation
from matplotlib.colors import to_rgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from nnn import layer, model


# ---------------------------------------------------------------------------
# DualNoiseNNN  (self-contained copy; no dependency on other example files)
# ---------------------------------------------------------------------------

class DualNoiseNNN(nn.Module):
    """Shared-weight NNN with two noise-type modes.

    Linear layers (fcs) are shared.  The activation nonlinearity switches
    between GaussianCrossingAnalytic and ParabolicCrossingAnalytic depending
    on the `noise_type` argument.

        'gaussian'  →  E[Crossing | Gaussian noise]  →  learns sin(x)
        'uniform'   →  E[Crossing | Uniform  noise]  →  learns cos(x)
    """

    def __init__(self, structure: list = [1, 50, 50, 1],
                 std: float = 0.5, max_radius: float = 1.0,
                 output_bias: bool = False):
        if len(structure) < 3:
            raise ValueError("structure must have at least 3 elements.")
        super(DualNoiseNNN, self).__init__()
        self.structure = structure
        n_hidden = len(structure) - 2

        self.fcs = nn.ModuleList([
            nn.Linear(pre, post,
                      bias=(output_bias if idx == len(structure) - 2 else True))
            for idx, (pre, post) in enumerate(zip(structure, structure[1:]))
        ])
        self.gaussian_crossings = nn.ModuleList([
            layer.GaussianCrossingAnalyticLayer(std=std)
            for _ in range(n_hidden)
        ])
        self.uniform_crossings = nn.ModuleList([
            layer.ParabolicCrossingAnalyticLayer(
                recruitment=1.0, center=0.0, max_radius=max_radius)
            for _ in range(n_hidden)
        ])

    def forward(self, x: torch.Tensor, noise_type: str = 'gaussian') -> torch.Tensor:
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

    def forward_blend(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """Forward pass under mixture-noise activation.

        At each hidden layer applies the analytic expected crossing value for
        mixture noise  ε ~ (1−α)·N(0,σ²) + α·U(−r, r):

            E[crossing(z+ε)] = (1−α)·f_Gaussian(z) + α·f_Parabolic(z)

        This follows from the law of total expectation and is applied
        independently at every hidden layer — NOT as a blend of the two
        final network outputs.

        Args:
            x:     Input tensor of shape [N, D_in].
            alpha: Mixture weight in [0, 1].
                   0 → pure Gaussian activation (sin mode)
                   1 → pure Uniform/Parabolic activation (cos mode)
        """
        n_hidden = len(self.structure) - 2
        for i in range(len(self.structure) - 1):
            x = self.fcs[i](x)
            if i < n_hidden:
                z_g = self.gaussian_crossings[i](x)
                z_u = self.uniform_crossings[i](x)
                x = (1.0 - alpha) * z_g + alpha * z_u
        return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(net: nn.Module, x: torch.Tensor,
          sin_y: torch.Tensor, cos_y: torch.Tensor,
          epochs: int, lr: float, print_every: int = 500):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        net.train()
        optimizer.zero_grad()
        loss = (criterion(net(x, 'gaussian'), sin_y) +
                criterion(net(x, 'uniform'),  cos_y))
        loss.backward()
        optimizer.step()
        if print_every > 0 and (epoch == 1 or epoch % print_every == 0):
            print(f"  epoch {epoch:5d} | loss {loss.item():.4e}")


# ---------------------------------------------------------------------------
# Activation function shapes (computed analytically, no torch.autograd)
# ---------------------------------------------------------------------------

def gaussian_crossing_np(z: np.ndarray, sigma: float) -> np.ndarray:
    """Analytic Crossing expected value under Gaussian(0, sigma) noise.
    = 2 * Φ(z/σ) * (1 − Φ(z/σ))
    """
    cdf = 0.5 * (1.0 + np.vectorize(
        lambda v: float(torch.erf(torch.tensor(v / (sigma * (2 ** 0.5))))))(z))
    return 2.0 * cdf * (1.0 - cdf)


def parabolic_crossing_np(z: np.ndarray, radius: float) -> np.ndarray:
    """Analytic Crossing expected value under Uniform(−radius, radius) noise.
    = 0.5 * [1 − (z/radius)²]₊
    """
    return np.maximum(0.0, 0.5 * (1.0 - (z / radius) ** 2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Animate noise-type transition in a dual-noise NNN"
    )
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=50)
    parser.add_argument("--std", type=float, default=0.5,
                        help="Gaussian noise std")
    parser.add_argument("--max-radius", type=float, default=1.0,
                        help="Uniform noise half-width")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--n-half", type=int, default=80,
                        help="Frames per half-sweep (Gaussian→Uniform or back)")
    parser.add_argument("--interval", type=int, default=50,
                        help="Animation frame interval in ms")
    parser.add_argument("--save", type=str, default="",
                        help="Save animation to this file (e.g. out.mp4) instead of showing")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Dataset ---
    x_np = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
    x_tensor  = torch.tensor(x_np, dtype=torch.float32)
    sin_tensor = torch.tensor(np.sin(x_np), dtype=torch.float32)
    cos_tensor = torch.tensor(np.cos(x_np), dtype=torch.float32)
    x_plot = x_np.ravel()

    # --- Model ---
    structure = [1, args.hidden_dim, args.hidden_dim, 1]
    net = DualNoiseNNN(structure=structure, std=args.std,
                       max_radius=args.max_radius, output_bias=False)

    print(f"Structure : {structure}")
    print(f"Gaussian std={args.std}  →  sin(x)")
    print(f"Uniform radius={args.max_radius}  →  cos(x)")
    print(f"Shared params : {sum(p.numel() for p in net.fcs.parameters())}\n")
    print("Training ...")
    train(net, x_tensor, sin_tensor, cos_tensor,
          epochs=args.epochs, lr=args.lr, print_every=args.print_every)

    # --- Evaluate both modes ---
    net.eval()
    with torch.no_grad():
        y_gauss   = net(x_tensor, 'gaussian').numpy().ravel()
        y_uniform = net(x_tensor, 'uniform').numpy().ravel()

    rmse_sin = float(np.sqrt(np.mean((y_gauss   - np.sin(x_plot)) ** 2)))
    rmse_cos = float(np.sqrt(np.mean((y_uniform - np.cos(x_plot)) ** 2)))
    print(f"\nGaussian mode → sin RMSE : {rmse_sin:.5f}")
    print(f"Uniform  mode → cos RMSE : {rmse_cos:.5f}\n")

    # --- Alpha sweep ---
    alphas = np.concatenate([
        np.linspace(0.0, 1.0, args.n_half),
        np.linspace(1.0, 0.0, args.n_half),
    ])
    n_frames = len(alphas)

    # Precompute predictions via activation-level blending (forward_blend).
    # Each hidden layer uses (1-α)·Gaussian_analytic + α·Parabolic_analytic,
    # which is the exact analytic activation for mixture noise
    # ε ~ (1-α)·N(0,σ²) + α·U(-r,r).
    # This differs from output-level blending because the non-linearity in
    # subsequent layers amplifies the per-layer difference.
    with torch.no_grad():
        interp_preds = [
            net.forward_blend(x_tensor, float(a)).numpy().ravel()
            for a in alphas
        ]

    # Precompute blended activation shapes
    z_np = np.linspace(-2.5, 2.5, 400)
    act_g = gaussian_crossing_np(z_np, args.std)
    act_u = parabolic_crossing_np(z_np, args.max_radius)
    interp_acts = [(1.0 - a) * act_g + a * act_u for a in alphas]

    # Colors: C0 (blue) for Gaussian, C1 (orange) for Uniform
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    c_g = np.array(to_rgb(cycle[0]))
    c_u = np.array(to_rgb(cycle[1]))

    def lerp_color(alpha: float) -> tuple:
        return tuple((1.0 - alpha) * c_g + alpha * c_u)

    # --- Figure ---
    fig, (ax_main, ax_act) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={'width_ratios': [2.5, 1]},
    )

    # Left: prediction
    ax_main.plot(x_plot, np.sin(x_plot), color='steelblue',
                 lw=1.5, ls='--', alpha=0.55, label='sin(x)  [Gaussian target]')
    ax_main.plot(x_plot, np.cos(x_plot), color='darkorange',
                 lw=1.5, ls=':', alpha=0.55, label='cos(x)  [Uniform target]')
    pred_line, = ax_main.plot(x_plot, interp_preds[0],
                              lw=2.5, color=lerp_color(0.0), label='network output')
    ax_main.set_xlim(x_plot[0], x_plot[-1])
    ax_main.set_ylim(-1.4, 1.4)
    ax_main.set_xlabel('x')
    ax_main.set_ylabel('y')
    ax_main.legend(loc='upper right', fontsize=8)
    ax_main.grid(alpha=0.3)
    pred_title = ax_main.set_title('')

    # Right: activation function
    ax_act.plot(z_np, act_g, color='steelblue',  lw=1.5, ls='--',
                alpha=0.55, label=f'Gaussian (σ={args.std})')
    ax_act.plot(z_np, act_u, color='darkorange', lw=1.5, ls=':',
                alpha=0.55, label=f'Uniform (r={args.max_radius})')
    act_line, = ax_act.plot(z_np, interp_acts[0],
                            lw=2.5, color=lerp_color(0.0), label='blended')
    ax_act.set_xlim(z_np[0], z_np[-1])
    ax_act.set_ylim(-0.03, 0.55)
    ax_act.set_xlabel('pre-activation z')
    ax_act.set_ylabel('activation output')
    ax_act.set_title('Activation function shape')
    ax_act.legend(fontsize=7)
    ax_act.grid(alpha=0.3)

    fig.suptitle(
        f'Dual-noise NNN — noise-type transition  '
        f'(hidden={args.hidden_dim}, σ={args.std}, r={args.max_radius})',
        fontsize=11,
    )
    plt.tight_layout()

    # --- Update function ---
    def update(frame: int):
        alpha = alphas[frame]
        color = lerp_color(alpha)

        pred_line.set_ydata(interp_preds[frame])
        pred_line.set_color(color)

        act_line.set_ydata(interp_acts[frame])
        act_line.set_color(color)

        if alpha < 0.02:
            mode_str = 'Gaussian noise  →  sin(x)'
        elif alpha > 0.98:
            mode_str = 'Uniform noise  →  cos(x)'
        else:
            mode_str = (f'{(1 - alpha)*100:.0f}% Gaussian + '
                        f'{alpha*100:.0f}% Uniform')

        pred_title.set_text(f'α = {alpha:.2f}  |  {mode_str}')
        return pred_line, act_line, pred_title

    ani = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=args.interval,
        blit=False,
        repeat=True,
    )

    if args.save:
        print(f"Saving animation to {args.save} ...")
        ani.save(args.save, writer='ffmpeg', dpi=150)
        print("Done.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
