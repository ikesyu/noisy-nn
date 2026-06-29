"""Animate a smooth noise-type transition in a dual-noise sample NNN.

The sample-based counterpart of animation_dual_noise.py.

A DualNoiseSampleNNN is trained so that:
    Gaussian noise mode  →  sin(x)
    Uniform  noise mode  →  cos(x)

The animation sweeps α from 0 to 1 and back using forward_blend(x, α),
which adds the following noise at each hidden layer each frame:

    ε = (1−α)·ε_G + α·ε_U,   ε_G ~ N(0,σ²),  ε_U ~ U(−r,r)

Because noise is sampled every frame, the output is stochastic —
fluctuations visible in the animation are genuine sampling noise.
This contrasts with animation_dual_noise.py where the analytic
(deterministic) activation is used.

Right panel: noise PDF shape at current α, estimated from 50 000 samples,
showing how the noise distribution itself morphs during the sweep.

This file is self-contained; it does not depend on other example scripts.

Run from the project root:

    python examples/animation_dual_noise_sample.py
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

from nnn import activation, layer


# ---------------------------------------------------------------------------
# Mixture crossing layer  (self-contained copy)
# ---------------------------------------------------------------------------

class MixtureCrossingSampleLayer(nn.Module):
    """CrossingSample with blended Gaussian + Uniform noise.

    ε = (1−α)·ε_G + α·ε_U,  ε_G ~ N(0,std²), ε_U ~ U(center−r, center+r)
    Operates on [N, T, D] tensors.
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
# Dual-noise sample model  (self-contained copy)
# ---------------------------------------------------------------------------

class DualNoiseSampleNNN(nn.Module):
    """Shared-weight NNN with sample-based Gaussian / Uniform noise modes."""

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
        alpha = 0.0 if noise_type == 'gaussian' else 1.0
        return self.forward_blend(x, alpha)

    def forward_blend(self, x: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
        """Forward pass with blended noise at each hidden layer."""
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
# Noise PDF estimation via sampling
# ---------------------------------------------------------------------------

def estimate_noise_pdf(alpha: float, std: float, radius: float,
                       n_samples: int = 50_000,
                       bins: int = 200,
                       x_range: tuple = (-3.0, 3.0)) -> tuple[np.ndarray, np.ndarray]:
    """Estimate PDF of ε = (1−α)·ε_G + α·ε_U by histogram."""
    eps_g = np.random.randn(n_samples) * std
    eps_u = (np.random.uniform(-1.0, 1.0, n_samples)) * radius
    eps = (1.0 - alpha) * eps_g + alpha * eps_u
    counts, edges = np.histogram(eps, bins=bins, range=x_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Animate noise-type transition in a dual-noise sample NNN"
    )
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=50)
    parser.add_argument("--t", type=int, default=64,
                        help="Monte-Carlo samples per input (T)")
    parser.add_argument("--std", type=float, default=0.5,
                        help="Gaussian noise std")
    parser.add_argument("--max-radius", type=float, default=1.0,
                        help="Uniform noise half-width")
    parser.add_argument("--h", type=float, default=0.1,
                        help="Crossing activation threshold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--n-half", type=int, default=80,
                        help="Frames per half-sweep (Gaussian→Uniform or back)")
    parser.add_argument("--interval", type=int, default=80,
                        help="Animation frame interval in ms")
    parser.add_argument("--save", type=str, default="",
                        help="Save animation to file (e.g. out.mp4)")
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
    print("Training ...")
    train(net, x_tensor, sin_tensor, cos_tensor,
          epochs=args.epochs, lr=args.lr, print_every=args.print_every)
    net.eval()

    # Quick accuracy report
    with torch.no_grad():
        rmse_sin = float(torch.sqrt(
            ((net(x_tensor, 'gaussian') - sin_tensor) ** 2).mean()))
        rmse_cos = float(torch.sqrt(
            ((net(x_tensor, 'uniform')  - cos_tensor) ** 2).mean()))
    print(f"\nGaussian mode → sin RMSE : {rmse_sin:.5f}")
    print(f"Uniform  mode → cos RMSE : {rmse_cos:.5f}\n")

    # --- Alpha sweep ---
    alphas = np.concatenate([
        np.linspace(0.0, 1.0, args.n_half),
        np.linspace(1.0, 0.0, args.n_half),
    ])
    n_frames = len(alphas)

    # Precompute noise PDF estimates for all alpha values
    print("Precomputing noise PDFs ...")
    pdf_x, pdf_g = estimate_noise_pdf(0.0, args.std, args.max_radius)
    _,     pdf_u = estimate_noise_pdf(1.0, args.std, args.max_radius)
    pdf_cache = {}
    for a in np.unique(np.round(alphas, 4)):
        _, pdf_cache[round(a, 4)] = estimate_noise_pdf(
            float(a), args.std, args.max_radius)

    # Colors
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    c_g = np.array(to_rgb(cycle[0]))
    c_u = np.array(to_rgb(cycle[1]))

    def lerp_color(alpha: float) -> tuple:
        return tuple((1.0 - alpha) * c_g + alpha * c_u)

    # --- Figure ---
    fig, (ax_main, ax_noise) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={'width_ratios': [2.5, 1]},
    )

    # Left: prediction
    ax_main.plot(x_plot, np.sin(x_plot), color='steelblue',
                 lw=1.5, ls='--', alpha=0.55, label='sin(x)  [Gaussian target]')
    ax_main.plot(x_plot, np.cos(x_plot), color='darkorange',
                 lw=1.5, ls=':',  alpha=0.55, label='cos(x)  [Uniform target]')
    pred_line, = ax_main.plot(x_plot, np.zeros_like(x_plot),
                              lw=2.0, color=lerp_color(0.0), label='network output')
    ax_main.set_xlim(x_plot[0], x_plot[-1])
    ax_main.set_ylim(-1.6, 1.6)
    ax_main.set_xlabel('x')
    ax_main.set_ylabel('y')
    ax_main.legend(loc='upper right', fontsize=8)
    ax_main.grid(alpha=0.3)
    pred_title = ax_main.set_title('')

    # Right: noise PDF
    ax_noise.plot(pdf_x, pdf_g, color='steelblue',  lw=1.5, ls='--',
                  alpha=0.55, label=f'Gaussian (σ={args.std})')
    ax_noise.plot(pdf_x, pdf_u, color='darkorange', lw=1.5, ls=':',
                  alpha=0.55, label=f'Uniform (r={args.max_radius})')
    noise_line, = ax_noise.plot(pdf_x, pdf_g,
                                lw=2.0, color=lerp_color(0.0), label='current')
    ax_noise.set_xlim(pdf_x[0], pdf_x[-1])
    ax_noise.set_ylim(bottom=0)
    ax_noise.set_xlabel('ε  (noise value)')
    ax_noise.set_ylabel('density')
    ax_noise.set_title('Noise distribution PDF')
    ax_noise.legend(fontsize=7)
    ax_noise.grid(alpha=0.3)

    fig.suptitle(
        f'Dual-noise sample NNN — noise-type transition  '
        f'(T={args.t}, hidden={args.hidden_dim}, σ={args.std}, r={args.max_radius})',
        fontsize=10,
    )
    plt.tight_layout()

    # --- Animation update (forward_blend called fresh each frame) ---
    def update(frame: int):
        alpha = alphas[frame]
        color = lerp_color(alpha)

        # Stochastic forward pass — new samples each frame
        with torch.no_grad():
            y = net.forward_blend(x_tensor, float(alpha)).numpy().ravel()
        pred_line.set_ydata(y)
        pred_line.set_color(color)

        # Noise PDF for this alpha
        pdf_key = round(float(alpha), 4)
        pdf_vals = pdf_cache.get(pdf_key, pdf_g)
        noise_line.set_ydata(pdf_vals)
        noise_line.set_color(color)

        if alpha < 0.02:
            mode_str = 'Gaussian noise  →  sin(x)'
        elif alpha > 0.98:
            mode_str = 'Uniform noise  →  cos(x)'
        else:
            mode_str = (f'{(1 - alpha)*100:.0f}% Gaussian + '
                        f'{alpha*100:.0f}% Uniform')
        pred_title.set_text(f'α = {alpha:.2f}  |  {mode_str}')

        return pred_line, noise_line, pred_title

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
