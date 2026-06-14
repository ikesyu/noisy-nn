"""
Generate a figure demonstrating masked 2D learning: train the network on only
2 functions chosen from a 10x10 grid, then plot intermediate functions
obtained for noise fields that lie on the straight line connecting the two
corresponding 2D grid centers.

Usage (from the examples/ directory):
    python generate_masked_2d_interpolation_figure.py \
        --idx_a 2 3 --idx_b 7 6 --n_steps 7 --epochs 5000 --show
"""

from multidimensional_storage_functions import SineFunctions, plain_sine, gray_ndindex
from multidimensional_storage import (
    Structure,
    noise_pattern,
    noise_pattern_table,
    fractional_index,
)
import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

sys.path.append("../")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def build_noise_panels(ax, stdvec, title=None):
    """Render a noise vector as a 2-row hidden-layer heatmap.

    Matches the exact style of regression_hidden* figures produced by
    generate_noise_virtual_to_hidden_regression_figure.py.
    """
    stdvec_np = stdvec.detach().cpu().numpy().reshape(-1)
    hidden_layer_viz = np.tile(stdvec_np, (2, 1))
    im = ax.imshow(hidden_layer_viz, cmap="viridis",
                   aspect="auto", interpolation="nearest")
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("Hidden layer")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Layer 1", "Layer 2"])
    for j in range(hidden_layer_viz.shape[1] + 1):
        ax.axvline(j - 0.5, color="gray", linewidth=0.35, alpha=0.45)
    ax.axhline(0.5, color="gray", linewidth=1.2, alpha=0.9)
    return im


def get_function_params(functions, idx):
    """
    For a SineFunctions with shape (n_phase, n_freq) or (n_phase,), return
    the (phase, amplitude) parameters that correspond to grid index `idx`.

    SineFunctions calls sine_pfa which uses:
      phase  = linspace(0, 2π, n_phase, endpoint=False)
      freq   = linspace(1, 2,  n_freq)
      amp    = linspace(1, 2,  n_amp)
    and the shape of ys is (*function_structure, n_x, 1).

    For a 2D grid shape=(n_phase, n_freq) the first axis is phase,
    the second is frequency.  amplitude is always 1 (n_amp dimension
    squeezed out) in this 2D case.
    """
    shape = functions.shape          # e.g. (10, 10)
    ndim = len(shape)

    params = {}

    # phase is always along axis 0
    n_phase = shape[0]
    phases = np.linspace(0, 2 * np.pi, n_phase, endpoint=False)
    params["phase"] = float(phases[idx[0]])

    if ndim >= 2:
        n_freq = shape[1]
        freqs = np.linspace(1, 2, n_freq)
        params["freq"] = float(freqs[idx[1]])

    if ndim >= 3:
        n_amp = shape[2]
        amps = np.linspace(1, 2, n_amp)
        params["amp"] = float(amps[idx[2]])
    else:
        params["amp"] = 1.0        # amplitude squeezed out → constant 1

    return params


# ---------------------------------------------------------------------------
# Masked training
# ---------------------------------------------------------------------------

def train_net_masked(structure, functions, mask_indices, epochs=5000, seed=None):
    """
    Train exactly like `train_net` but iterate only over the indices listed
    in `mask_indices` (a list of tuples).

    Parameters
    ----------
    structure    : Structure
    functions    : SineFunctions (or any *Functions object)
    mask_indices : list of index tuples to train on
    epochs       : int
    seed         : optional int

    Returns
    -------
    model, losses (list, one value per mask index)
    """
    from nnn import model as nnn_model

    if seed is not None:
        set_global_seed(seed)

    net = nnn_model.SimpleNNNAnalytic(structure.real_structure)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=0.0)

    yshape = functions.ys.shape[:-2]          # (10, 10)
    all_noise = noise_pattern_table(structure, yshape)   # full table, cheap

    indices = list(mask_indices)

    best_loss = float("inf")
    patience_ctr = 0
    patience = 10
    min_delta = -1e-3

    for epoch in range(epochs):
        random.shuffle(indices)
        loss_sum = 0.0
        for idx in indices:
            optimizer.zero_grad()
            out = net(functions.x_tensor, all_noise[idx])
            loss = criterion(out, functions.ys[idx])
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        if epoch % 50 == 0:
            delta = loss_sum - best_loss
            print(
                f"epoch={epoch:5d}  loss_sum={loss_sum:.6f}  delta={delta:.6f}")
            if delta > min_delta:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                best_loss = loss_sum
                patience_ctr = 0

    losses = []
    with torch.no_grad():
        for idx in indices:
            out = net(functions.x_tensor, all_noise[idx])
            loss = criterion(out, functions.ys[idx])
            losses.append(float(loss.item()))

    return net, losses


# ---------------------------------------------------------------------------
# 2-D noise interpolation along a straight line
# ---------------------------------------------------------------------------

def interpolate_noise_2d(structure, grid_shape, idx_a, idx_b, n_steps):
    """
    Build noise vectors that correspond to positions on the straight line
    connecting the fractional 2D centers of idx_a and idx_b.

    Returns
    -------
    positions : array of shape (n_steps,) in [0, 1]  (0 = idx_a, 1 = idx_b)
    noise_list : list of length n_steps, each a list [stdvec, stdvec]
    """
    frac_a = np.array(fractional_index(idx_a, grid_shape))   # e.g. [0.2, 0.3]
    frac_b = np.array(fractional_index(idx_b, grid_shape))

    positions = np.linspace(0.0, 1.0, n_steps)
    noise_list = []

    for t in positions:
        center = tuple((1 - t) * frac_a + t * frac_b)
        noise_list.append(noise_pattern(structure, center))

    return positions, noise_list


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def get_cache_path(grid_shape, noise_structure, activation_ratio,
                   idx_a, idx_b, epochs, seed):
    """
    Return a deterministic path in ../data/ that encodes all parameters
    that affect the trained model.
    """
    g = "_".join(str(n) for n in grid_shape)
    ns = "_".join(str(n) for n in noise_structure)
    ia = "_".join(str(i) for i in idx_a)
    ib = "_".join(str(i) for i in idx_b)
    label = (f"masked2d_g{g}_n{ns}_a{activation_ratio}"
             f"_A{ia}_B{ib}_e{epochs}_r{seed}")
    data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data"))
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, f"{label}.pt")


def save_cache(path, net, losses, ys):
    torch.save({"model": net, "losses": losses, "ys": ys}, path)
    print(f"Cache saved → {path}")


def load_cache(path):
    data = torch.load(path, weights_only=False,
                      map_location=torch.device("cpu"))
    print(f"Cache loaded ← {path}")
    return data["model"], data["losses"], data["ys"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train on 2 masked functions out of a 10x10 2D grid, then plot "
            "intermediate functions along the line between their centers."
        )
    )
    parser.add_argument("--grid", nargs=2, type=int, default=[10, 10],
                        help="Grid shape (default: 10 10)")
    parser.add_argument("--noise_structure", nargs="+", type=int, default=[8, 8],
                        help="2D noise structure (default: 8 8)")
    parser.add_argument("--idx_a", nargs=2, type=int, default=[0, 1],
                        help="Grid index of function A (default: 0 1)")
    parser.add_argument("--idx_b", nargs=2, type=int, default=[7, 6],
                        help="Grid index of function B (default: 7 6)")
    parser.add_argument("--n_steps", type=int, default=5,
                        help="Number of interpolation steps (default: 5)")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--activation_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--retrain", action="store_true",
                        help="Ignore any cached model and re-train from scratch")
    args = parser.parse_args()

    set_global_seed(args.seed)

    grid_shape = tuple(args.grid)          # (10, 10)
    idx_a = tuple(args.idx_a)         # e.g. (1, 2)
    idx_b = tuple(args.idx_b)         # e.g. (7, 6)

    # ------------------------------------------------------------------
    # Validate indices
    # ------------------------------------------------------------------
    for dim, (i, n) in enumerate(zip(idx_a, grid_shape)):
        if not (0 <= i < n):
            raise ValueError(f"idx_a[{dim}]={i} out of range [0, {n-1}]")
    for dim, (i, n) in enumerate(zip(idx_b, grid_shape)):
        if not (0 <= i < n):
            raise ValueError(f"idx_b[{dim}]={i} out of range [0, {n-1}]")

    # ------------------------------------------------------------------
    # Build functions (SineFunctions uses shape=(n_phase, n_freq) in 2D)
    # ------------------------------------------------------------------
    structure = Structure(args.noise_structure,
                          activation_ratio=args.activation_ratio)
    functions = SineFunctions(
        list(grid_shape), shuffle=False, epochs=args.epochs)

    # ------------------------------------------------------------------
    # Print parameters for the two chosen functions
    # ------------------------------------------------------------------
    params_a = get_function_params(functions, idx_a)
    params_b = get_function_params(functions, idx_b)

    print("=" * 60)
    print(f"Function A  index={idx_a}")
    for k, v in params_a.items():
        print(f"  {k:10s} = {v:.6f} rad" if "phase" in k else f"  {k:10s} = {v:.6f}")
    print()
    print(f"Function B  index={idx_b}")
    for k, v in params_b.items():
        print(f"  {k:10s} = {v:.6f} rad" if "phase" in k else f"  {k:10s} = {v:.6f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Train with mask = {idx_a, idx_b}  — or load from cache
    # ------------------------------------------------------------------
    cache_path = get_cache_path(
        grid_shape, args.noise_structure, args.activation_ratio,
        idx_a, idx_b, args.epochs, args.seed,
    )

    if not args.retrain and Path(cache_path).is_file():
        net, losses, ys_cached = load_cache(cache_path)
    else:
        print(
            f"Training on mask = {{{idx_a}, {idx_b}}} out of {grid_shape} grid …\n")
        net, losses = train_net_masked(
            structure, functions,
            mask_indices=[idx_a, idx_b],
            epochs=args.epochs,
            seed=args.seed,
        )
        # Cache ys for the two trained indices so plots are reproducible
        ys_cached = {
            idx_a: functions.ys[idx_a].detach().cpu(),
            idx_b: functions.ys[idx_b].detach().cpu(),
        }
        save_cache(cache_path, net, losses, ys_cached)

    print(f"Final losses: A={losses[0]:.6f}  B={losses[1]:.6f}\n")

    # ------------------------------------------------------------------
    # Interpolate noise along the line between the two centers (2D)
    # ------------------------------------------------------------------
    positions, noise_list = interpolate_noise_2d(
        structure, grid_shape, idx_a, idx_b, args.n_steps
    )

    dev = next(net.parameters()).device
    x_tensor = functions.x_tensor.to(dev)
    x = functions.x.reshape(-1)

    y_preds = []
    with torch.no_grad():
        for noise_vecs in noise_list:
            noise_on_dev = [v.to(dev) for v in noise_vecs]
            out = net(x_tensor, noise_on_dev).detach(
            ).cpu().numpy().reshape(-1)
            y_preds.append(out)

    # ------------------------------------------------------------------
    # Ground-truth target functions
    # ------------------------------------------------------------------
    y_true_a = functions.ys[idx_a].detach().cpu().numpy().reshape(-1)
    y_true_b = functions.ys[idx_b].detach().cpu().numpy().reshape(-1)

    # ------------------------------------------------------------------
    # Shared color scheme: viridis, endpoints pinned for A (t=0) and B (t=1)
    # ------------------------------------------------------------------
    cmap_v = plt.get_cmap("viridis")
    t_values = np.linspace(0.0, 1.0, args.n_steps)
    colors = [cmap_v(v) for v in np.linspace(0.0, 0.8, args.n_steps)]
    color_a = colors[0]    # viridis color at t=0  (used for target A)
    color_b = colors[-1]   # viridis color at t=1  (used for target B)

    # Human-readable function formulas  e.g.  sin(1.22 x + 0.63)
    def fmt_sin(p):
        freq = p.get("freq", 1.0)
        phase = p["phase"]
        sign = "+" if phase >= 0 else "-"
        return rf"$\sin({freq:.2f}\,x {sign} {abs(phase):.2f})$"

    label_a = fmt_sin(params_a)
    label_b = fmt_sin(params_b)

    # ------------------------------------------------------------------
    # Figure 1 – interpolated function curves  (no title)
    # ------------------------------------------------------------------
    fig_reg, ax_reg = plt.subplots(figsize=(9, 5), constrained_layout=True)

    for t, color, y_pred in zip(t_values, colors, y_preds):
        ax_reg.plot(x, y_pred, color=color, linewidth=1.8,
                    label=f"$t={t:.2f}$", alpha=0.85)

    ax_reg.plot(x, y_true_a, "o", color=color_a, markersize=3,
                alpha=0.9, label=f"target {label_a}")
    ax_reg.plot(x, y_true_b, "s", color=color_b, markersize=3,
                alpha=0.9, label=f"target {label_b}")

    ax_reg.set_xlabel("$x$")
    ax_reg.set_ylabel("$y$")
    ax_reg.set_xlim(-2 * np.pi, 2 * np.pi)
    ax_reg.grid(True, alpha=0.3)
    ax_reg.legend(loc="upper right", ncol=2, fontsize=8)

    # ------------------------------------------------------------------
    # Figure 1b – masked_2d_regression_ex (only t=0 and t=1)
    # ------------------------------------------------------------------
    fig_reg_ex, ax_reg_ex = plt.subplots(
        figsize=(9, 5), constrained_layout=True)
    ax_reg_ex.plot(x, y_preds[0], color=color_a,
                   linewidth=1.8, label="Result A", alpha=0.85)
    ax_reg_ex.plot(x, y_true_a, "o", color=color_a,
                   markersize=3, alpha=0.9, label="Target A")
    ax_reg_ex.plot(x, y_preds[-1], color=color_b,
                   linewidth=1.8, label="Result B", alpha=0.85)
    ax_reg_ex.plot(x, y_true_b, "s", color=color_b,
                   markersize=3, alpha=0.9, label="Target B")
    ax_reg_ex.set_xlabel("$x$")
    ax_reg_ex.set_ylabel("$y$")
    ax_reg_ex.set_xlim(-2 * np.pi, 2 * np.pi)
    ax_reg_ex.grid(True, alpha=0.3)
    ax_reg_ex.legend(loc="upper right", ncol=2, fontsize=8)

    # ------------------------------------------------------------------
    # Figure 1c – maskex_2d_regression_mid
    # ------------------------------------------------------------------
    fig_reg_mid, ax_reg_mid = plt.subplots(
        figsize=(9, 5), constrained_layout=True)
    for t, color, y_pred in zip(t_values, colors, y_preds):
        if t == 0.0 or t == 1.0:
            ax_reg_mid.plot(x, y_pred, color=color, linewidth=1.8, linestyle=":",
                            label=f"$t={t:.2f}$", alpha=0.85)
        else:
            ax_reg_mid.plot(x, y_pred, color=color, linewidth=1.8,
                            label=f"$t={t:.2f}$", alpha=0.85)
    ax_reg_mid.set_xlabel("$x$")
    ax_reg_mid.set_ylabel("$y$")
    ax_reg_mid.set_xlim(-2 * np.pi, 2 * np.pi)
    ax_reg_mid.grid(True, alpha=0.3)
    ax_reg_mid.legend(loc="upper right", ncol=2, fontsize=8)

    # ------------------------------------------------------------------
    # Figure 2 (×n_steps) – noise fields styled like build_virtual_panel
    # ------------------------------------------------------------------
    noise_shape = tuple(args.noise_structure)   # (8, 8)
    n_rows = args.n_steps

    # Find global vmin/vmax for consistent colormap across panels
    all_fields = [
        noise_list[i][0].detach().cpu().numpy().reshape(noise_shape)
        for i in range(n_rows)
    ]
    vmin = min(f.min() for f in all_fields)
    vmax = max(f.max() for f in all_fields)

    figs_noise = []
    for field, t, color in zip(all_fields, t_values, colors):
        # Increased width from 2.4 to 3.2 to prevent colorbar overlapping axis ticks
        fig_n, ax_n = plt.subplots(figsize=(3.2, 2.4), layout="tight")
        im_noise = ax_n.imshow(
            field, cmap="viridis", origin="lower", aspect="equal",
            interpolation="nearest", extent=[0, 1, 0, 1],
            vmin=vmin, vmax=vmax,
        )
        ax_n.set_xlim(0, 1)
        ax_n.set_ylim(0, 1)
        ax_n.set_xticks([0.0, 0.5, 1.0])
        ax_n.set_yticks([0.0, 0.5, 1.0])
        ax_n.set_xlabel("Virtual axis 1")
        ax_n.set_ylabel("Virtual axis 2")
        ax_n.set_aspect("equal")
        # Draw cell grid lines, matching build_virtual_panel style
        for j in range(noise_shape[1] + 1):
            frac = j / noise_shape[1]
            ax_n.axvline(frac, color="gray", linewidth=0.35, alpha=0.45)
        for i in range(noise_shape[0] + 1):
            frac = i / noise_shape[0]
            ax_n.axhline(frac, color="gray", linewidth=0.35, alpha=0.45)
        ax_n.set_title(f"$t={t:.2f}$", color=color, fontsize=9)

        divider = make_axes_locatable(ax_n)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig_n.colorbar(im_noise, cax=cax, label="Noise intensity")
        figs_noise.append((fig_n, ax_n, t))

    # ------------------------------------------------------------------
    # Figure 3 – 2D grid overview
    #   • axes: k_v^(1) (frequency) and k_v^(2) (phase)
    #   • hide gray function nodes; show 8×8 sampling centers instead
    #   • viridis colors for interpolated centers
    #   • A / B labelled with actual sin formula
    # ------------------------------------------------------------------
    fig_grid, ax_grid = plt.subplots(
        figsize=(5.5, 5.5), constrained_layout=True)

    # ---- 8×8 sampling centers in the physical neuron space mapped back
    #      to the virtual [0,1]^2 space, and then rescaled to "grid-coordinate"
    #      space. The noise_pattern maps virtual x to physical y via:
    #          y = a/2 + (1 - a) * x
    #      So we invert this to find the virtual positions of the neurons:
    #          x = (y - a/2) / (1 - a)
    noise_n0, noise_n1 = noise_shape  # 8, 8
    a = args.activation_ratio
    sc_i_phys = np.linspace(0, 1, noise_n0)
    sc_j_phys = np.linspace(0, 1, noise_n1)
    sc_i = (sc_i_phys - a / 2) / (1 - a)
    sc_j = (sc_j_phys - a / 2) / (1 - a)

    # Map to grid-index space (0..9)
    sc_i_grid = sc_i * (grid_shape[0] - 1)
    sc_j_grid = sc_j * (grid_shape[1] - 1)
    scj, sci = np.meshgrid(sc_j_grid, sc_i_grid)
    ax_grid.scatter(scj.ravel(), sci.ravel(), s=18, color="#aaaaaa",
                    zorder=2, label="sampling centers", marker="s", alpha=0.7)

    # ---- straight-line path from A to B in grid-index space
    frac_a = np.array(fractional_index(idx_a, grid_shape))
    frac_b = np.array(fractional_index(idx_b, grid_shape))
    path_i = np.array([frac_a[0], frac_b[0]]) * (grid_shape[0] - 1)
    path_j = np.array([frac_a[1], frac_b[1]]) * (grid_shape[1] - 1)
    ax_grid.plot(path_j, path_i, "k--", linewidth=1.0, zorder=3, alpha=0.55)

    # ---- active region radius in grid space (cutoff=0.1)
    # Volume of active region is exactly 'a'. In 2D, area = pi * r^2 = a
    r_phys = np.sqrt(a / np.pi)
    r_virt = r_phys / (1 - a)
    r_grid_i = r_virt * (grid_shape[0] - 1)
    r_grid_j = r_virt * (grid_shape[1] - 1)

    # ---- intermediate interpolation points & activation circles
    for t, color in zip(t_values, colors):
        frac = (1 - t) * frac_a + t * frac_b
        grid_pos = frac * (np.array(grid_shape) - 1)

        # Activation region circle (extremes only)
        if t == 0.0 or t == 1.0:
            ellipse = plt.matplotlib.patches.Ellipse(
                (grid_pos[1], grid_pos[0]),
                width=2 * r_grid_j, height=2 * r_grid_i,
                edgecolor=color, facecolor='none', linestyle='--',
                linewidth=1.2, zorder=3, alpha=0.6
            )
            ax_grid.add_patch(ellipse)

        # Intermediate points scatter (excluding t=0 and t=1)
        if 0 < t < 1:
            ax_grid.scatter(grid_pos[1], grid_pos[0], s=28,
                            color=color, zorder=4, alpha=0.9,
                            label=f"$t={t:.2f}$")

    # ---- endpoints A and B with formula labels
    ax_grid.scatter(idx_a[1], idx_a[0], s=160, color=color_a,
                    zorder=5, label=f"target {label_a}",
                    edgecolors="k", linewidths=0.8)
    ax_grid.scatter(idx_b[1], idx_b[0], s=160, color=color_b,
                    zorder=5, label=f"target {label_b}",
                    edgecolors="k", linewidths=0.8)

    ax_grid.set_xticks(np.arange(grid_shape[1]))
    ax_grid.set_xticklabels(np.arange(1, grid_shape[1] + 1))
    ax_grid.set_yticks(np.arange(grid_shape[0]))
    ax_grid.set_yticklabels(np.arange(1, grid_shape[0] + 1))

    ax_grid.set_xlabel(r"$k_v^{(1)}$ (frequency index)")
    ax_grid.set_ylabel(r"$k_v^{(2)}$ (phase index)")

    # Create legend with slightly more vertical spacing
    leg = ax_grid.legend(loc="upper right", fontsize=7.5,
                         framealpha=0.85, labelspacing=1.2)

    # Loop through legend handles and shrink the scatter dots down to a readable size
    for handle in leg.legend_handles:
        if hasattr(handle, 'set_sizes'):
            handle.set_sizes([30])

    ax_grid.grid(True, alpha=0.15)
    ax_grid.set_aspect("equal")

    ax_grid.set_xlabel(r"$k_v^{(1)}$ (frequency index)")
    ax_grid.set_ylabel(r"$k_v^{(2)}$ (phase index)")
#    ax_grid.legend(loc="upper right", fontsize=7.5, framealpha=0.85)
    ax_grid.legend(
        loc="upper right",
        fontsize=7.5,
        framealpha=0.85,
        labelspacing=1.02,     # Increases the vertical space between the rows
        handletextpad=1.2,    # Adds a little more space between the big dot and the text
        borderpad=1.2         # Gives the whole legend box a bit more padding
    )
    ax_grid.grid(True, alpha=0.15)
    ax_grid.set_aspect("equal")

    # ------------------------------------------------------------------
    # Figure 4 (×n_steps) – one hidden-layer panel per interpolation step,
    #   styled identically to regression_hidden* from
    #   generate_noise_virtual_to_hidden_regression_figure.py.
    # ------------------------------------------------------------------
    figs_hidden = []
    for i, (t, color, noise_vecs) in enumerate(
            zip(t_values, colors, noise_list)):
        stdvec = noise_vecs[0]          # flattened noise vector
        fig_h, ax_h = plt.subplots(figsize=(6.0, 2.6), constrained_layout=True)
        im_h = build_noise_panels(ax_h, stdvec)
        # Subtle viridis-matched title strip to link visually to the other figs
        ax_h.set_title(f"$t = {t:.2f}$", color=color, fontsize=10)
        fig_h.colorbar(im_h, ax=ax_h, label="Noise intensity")
        figs_hidden.append((fig_h, ax_h, t))

    # ------------------------------------------------------------------
    # Save figures
    # ------------------------------------------------------------------
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../fig"))
    os.makedirs(out_dir, exist_ok=True)

    suffix = f"_A{'_'.join(str(i) for i in idx_a)}_B{'_'.join(str(i) for i in idx_b)}"
    reg_path = os.path.join(out_dir, f"masked_2d_interp{suffix}.pdf")
    grid_path = os.path.join(out_dir, f"masked_2d_grid{suffix}.pdf")

    fig_reg.savefig(reg_path,   bbox_inches="tight", pad_inches=0.02)
    fig_grid.savefig(grid_path,  bbox_inches="tight", pad_inches=0.02)

    reg_ex_path = os.path.join(out_dir, f"masked_2d_regression_ex{suffix}.pdf")
    reg_mid_path = os.path.join(
        out_dir, f"maskex_2d_regression_mid{suffix}.pdf")
    fig_reg_ex.savefig(reg_ex_path, bbox_inches="tight", pad_inches=0.02)
    fig_reg_mid.savefig(reg_mid_path, bbox_inches="tight", pad_inches=0.02)

    print(f"Saved: {reg_path}")
    print(f"Saved: {grid_path}")
    print(f"Saved: {reg_ex_path}")
    print(f"Saved: {reg_mid_path}")

    for fig_n, ax_n, t in figs_noise:
        t_str = f"{t:.2f}".replace(".", "p")
        n_path = os.path.join(out_dir, f"masked_2d_noise_t{t_str}{suffix}.pdf")
        fig_n.savefig(n_path, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {n_path}")

        # Extra figure without title for extremes
        if t == 0.0 or t == 1.0:
            ax_n.set_title("")
            n_ex_path = os.path.join(
                out_dir, f"masked_2d_noise_ex_t{t_str}{suffix}.pdf")
            fig_n.savefig(n_ex_path, bbox_inches="tight", pad_inches=0.02)
            print(f"Saved: {n_ex_path}")

    for fig_h, ax_h, t in figs_hidden:
        t_str = f"{t:.2f}".replace(".", "p")
        h_path = os.path.join(
            out_dir, f"masked_2d_hidden_t{t_str}{suffix}.pdf")
        fig_h.savefig(h_path, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {h_path}")

        # Extra figure without title for extremes
        if t == 0.0 or t == 1.0:
            ax_h.set_title("")
            h_ex_path = os.path.join(
                out_dir, f"masked_2d_hidden_ex_t{t_str}{suffix}.pdf")
            fig_h.savefig(h_ex_path, bbox_inches="tight", pad_inches=0.02)
            print(f"Saved: {h_ex_path}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
