"""
Generate the fig:noise_virtual_to_hidden visualization.

This script creates a figure showing the mapping from the virtual noise field
to hidden-layer noise vectors for D=1, 2, and 3 dimensions.
For each D:
  - Left: truncated Gaussian evaluated on the discrete grid
  - Right: corresponding flattened noise vector mapped to the neural network
"""

from multidimensional_storage import Structure, gaussian_fill, compute_sigma, noise_pattern
import sys
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend

sys.path.append("../")


def add_grid_lines(ax, grid_shape, linewidth=0.5, color='white', alpha=0.6):
    """
    Add grid lines to show cell boundaries in an image.
    Grid lines are drawn at the edges between cells.
    """
    if len(grid_shape) == 1:
        # 1D: no grid needed
        return
    elif len(grid_shape) == 2:
        # 2D grid: draw horizontal and vertical lines
        ny, nx = grid_shape
        for i in np.arange(0, ny + 1, 1) - 0.5:
            ax.axhline(i, color=color, linewidth=linewidth, alpha=alpha)
        for j in np.arange(0, nx + 1, 1) - 0.5:
            ax.axvline(j, color=color, linewidth=linewidth, alpha=alpha)


def add_slice_grid_lines(ax, x0, y0, nx, ny, linewidth=0.5, color='white', alpha=0.6):
    """Add grid lines for a slice placed at offset (x0, y0) inside a larger canvas."""
    for i in np.arange(0, ny + 1, 1) - 0.5:
        ax.plot([x0 - 0.5, x0 + nx - 0.5], [y0 + i, y0 + i],
                color=color, linewidth=linewidth, alpha=alpha)
    for j in np.arange(0, nx + 1, 1) - 0.5:
        ax.plot([x0 + j, x0 + j], [y0 - 0.5, y0 + ny - 0.5],
                color=color, linewidth=linewidth, alpha=alpha)


def visualize_noise_field(dim, ax_left, ax_right):
    """
    Visualize the virtual noise field and resulting hidden-layer noise vector for dimension D.

    Args:
        dim: dimensionality (1, 2, or 3)
        ax_left: matplotlib axis for the virtual field visualization
        ax_right: matplotlib axis for the hidden-layer visualization
    """
    # Setup structure
    structure = Structure([2**(6//dim)]*dim, activation_ratio=0.5)

    # For visualization, place a Gaussian near the boundary
    center = tuple([0.15] * dim)

    # Compute sigma
    sigma = compute_sigma(0.1, 0.5, dim)

    # The plotted center should match the actual Gaussian center c' used after inward scaling.
    scaled_center = structure.activation_ratio / 2 + \
        (1 - structure.activation_ratio) * np.array(center)

    # Generate the noise pattern (Gaussian on discrete grid)
    noise_vec = noise_pattern(structure, center)[0].detach().cpu().numpy()

    # Reshape to visualize the grid
    grid_shape = structure.noise_structure
    grid_values = noise_vec.reshape(grid_shape)

    # ===== LEFT: Visualize the virtual noise field =====
    # Use viridis colormap (perceptually uniform, grayscale-friendly)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='white')

    if dim == 1:
        # 1D case: reshape to 2D for uniform visualization (1 x 64 heatmap)
        grid_2d = grid_values.reshape(1, -1)
        im = ax_left.imshow(grid_2d, cmap=cmap, origin='lower',
                            interpolation='nearest', aspect='auto')
        ax_left.set_xlabel('Position along Axis 0')
        ax_left.set_yticks([])
        ax_left.set_title(
            f'Virtual noise field for D=1\n(64 nodes in 1D grid)')

        # Set x-axis to fractional coordinates [0, 1]
        n_nodes = grid_shape[0]
        tick_positions = np.linspace(0, n_nodes - 1, 5)
        tick_labels = np.linspace(0, 1, 5)
        ax_left.set_xticks(tick_positions)
        ax_left.set_xticklabels([f'{x:.2f}' for x in tick_labels])

        center_grid_idx = scaled_center[0] * (n_nodes - 1)
        ax_left.plot(center_grid_idx, 0, 'r*', markersize=14, markeredgecolor='white',
                     markeredgewidth=0.6, label='Gaussian center')
        ax_left.legend(loc='upper right')

        # Add vertical grid lines
        for j in range(0, n_nodes + 1, max(1, n_nodes // 16)):
            ax_left.axvline(j - 0.5, color='gray', linewidth=0.3, alpha=0.5)

        plt.colorbar(im, ax=ax_left, label='Magnitude')

    elif dim == 2:
        # 2D case: heatmap with grid lines
        im = ax_left.imshow(grid_values, cmap=cmap,
                            origin='lower', interpolation='nearest')
        ax_left.set_title(
            f'Virtual noise field for D=2\n(8×8 = 64 nodes in 2D grid)')
        add_grid_lines(ax_left, grid_values.shape,
                       linewidth=0.5, color='white', alpha=0.6)

        # Set axes to fractional coordinates [0, 1]
        ny, nx = grid_values.shape
        tick_pos_y = np.linspace(0, ny - 1, 5)
        tick_labels_y = np.linspace(0, 1, 5)
        tick_pos_x = np.linspace(0, nx - 1, 5)
        tick_labels_x = np.linspace(0, 1, 5)

        ax_left.set_yticks(tick_pos_y)
        ax_left.set_yticklabels([f'{x:.2f}' for x in tick_labels_y])
        ax_left.set_xticks(tick_pos_x)
        ax_left.set_xticklabels([f'{x:.2f}' for x in tick_labels_x])
        ax_left.set_xlabel('Axis 1')
        ax_left.set_ylabel('Axis 0')

        plt.colorbar(im, ax=ax_left, label='Magnitude')

        # Mark center with a red marker
        center_grid_idx = (
            scaled_center[0] * (ny - 1), scaled_center[1] * (nx - 1))
        ax_left.plot(center_grid_idx[1], center_grid_idx[0],
                     'r*', markersize=15, label='Gaussian center')
        ax_left.legend(loc='upper right')

    elif dim == 3:
        # 3D case: show 4 representative slices in a 2x2 layout separated by whitespace
        d0_size = grid_values.shape[0]
        slice_indices = [d0_size // 4, d0_size //
                         2 - 1, d0_size // 2, 3 * d0_size // 4]
        slice_fractions = [0.25, 0.50 - 1 /
                           (2*d0_size), 0.50 + 1/(2*d0_size), 0.75]
        slice_labels = [f'Axis 0={f:.3f}' for f in slice_fractions]

        ny, nx = grid_values.shape[1], grid_values.shape[2]
        gap_x = 2
        gap_y = 3
        canvas_height = 2 * ny + gap_y
        canvas_width = 2 * nx + gap_x
        slice_canvas = np.full(
            (canvas_height, canvas_width), np.nan, dtype=float)

        positions = [
            (0, ny + gap_y),
            (nx + gap_x, ny + gap_y),
            (0, 0),
            (nx + gap_x, 0),
        ]

        for idx, (x0, y0) in enumerate(positions):
            slice_canvas[y0:y0 + ny, x0:x0 +
                         nx] = grid_values[slice_indices[idx], :, :]

        im = ax_left.imshow(slice_canvas, cmap=cmap,
                            origin='lower', interpolation='nearest')
        ax_left.set_title(
            f'Virtual noise field for D=3 (4 slices)\n(4×4×4 = 64 nodes, four views along Axis 0)',
            pad=18)

        # Add extra headroom so top slice labels can sit above the slices.
        ax_left.set_ylim(-0.5, canvas_height + 1.5)

        for x0, y0 in positions:
            add_slice_grid_lines(ax_left, x0, y0, nx, ny,
                                 linewidth=0.5, color='white', alpha=0.7)

        subtitle_y_top = canvas_height + 0.45
        subtitle_y_bottom = ny + gap_y / 2 - 1
        subtitle_x_left = (nx - 1) / 2
        subtitle_x_right = nx + gap_x + (nx - 1) / 2
        subtitle_props = dict(color='0.25', fontsize=9,
                              ha='center', va='center')
        ax_left.text(subtitle_x_left, subtitle_y_top,
                     slice_labels[0], **subtitle_props)
        ax_left.text(subtitle_x_right, subtitle_y_top,
                     slice_labels[1], **subtitle_props)
        ax_left.text(subtitle_x_left, subtitle_y_bottom,
                     slice_labels[2], **subtitle_props)
        ax_left.text(subtitle_x_right, subtitle_y_bottom,
                     slice_labels[3], **subtitle_props)

        center_slice_idx = int(np.round(scaled_center[0] * (d0_size - 1)))
        if center_slice_idx in slice_indices:
            marker_panel = slice_indices.index(center_slice_idx)
        else:
            marker_panel = int(
                np.argmin([abs(idx - center_slice_idx) for idx in slice_indices]))
        marker_x0, marker_y0 = positions[marker_panel]
        marker_x = marker_x0 + scaled_center[2] * (nx - 1)
        marker_y = marker_y0 + scaled_center[1] * (ny - 1)
        ax_left.plot(marker_x, marker_y, 'r*', markersize=14,
                     markeredgecolor='white', markeredgewidth=0.6)

        ax_left.set_xlabel('Axis 2')
        ax_left.set_ylabel('Axis 1')

        x_tick_positions = [0, (nx - 1) / 2, nx - 1, nx + gap_x,
                            nx + gap_x + (nx - 1) / 2, 2 * nx + gap_x - 1]
        x_tick_labels = ['0.00', '0.50', '1.00', '0.00', '0.50', '1.00']
        y_tick_positions = [0, (ny - 1) / 2, ny - 1, ny + gap_y,
                            ny + gap_y + (ny - 1) / 2, 2 * ny + gap_y - 1]
        y_tick_labels = ['0.00', '0.50', '1.00', '0.00', '0.50', '1.00']
        ax_left.set_xticks(x_tick_positions)
        ax_left.set_xticklabels(x_tick_labels)
        ax_left.set_yticks(y_tick_positions)
        ax_left.set_yticklabels(y_tick_labels)
        ax_left.tick_params(labelsize=8)

        # Remove the outer frame around the four-slice canvas.
        for spine in ax_left.spines.values():
            spine.set_visible(False)

        # Add colorbar
        plt.colorbar(im, ax=ax_left, label='Magnitude')

    # ===== RIGHT: Visualize the flattened hidden-layer noise vector =====
    # Reshape to 2D for visualization (2 rows: one per hidden layer)
    n_neurons = len(noise_vec)
    # Organize into rows for visibility
    neurons_per_row = max(8, n_neurons // 8)

    # Create a 2×(n_neurons) array where both rows are identical (both hidden layers use same vector)
    hidden_layer_viz = np.tile(noise_vec, (2, 1))

    im_right = ax_right.imshow(
        hidden_layer_viz, cmap='viridis', aspect='auto', interpolation='nearest')
    ax_right.set_xlabel('Neuron index (0 to 63)')
    ax_right.set_ylabel('Hidden layer')
    ax_right.set_yticks([0, 1])
    ax_right.set_yticklabels(['Layer 1', 'Layer 2'])
    ax_right.set_title(
        f'D={dim}: Flattened noise vector\n(both hidden layers, $N_h=64$ neurons)')

    # Add vertical grid lines to show all neuron divisions (gray)
    for j in range(0, n_neurons + 1, 1):
        ax_right.axvline(j - 0.5, color='gray', linewidth=0.4, alpha=0.5)
    ax_right.axhline(0.5, color='gray', linewidth=1.6, alpha=0.9)

    plt.colorbar(im_right, ax=ax_right, label='Noise intensity')


def main():
    """Generate and save the figure."""
    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(3, 2, figure=fig, hspace=0.58, wspace=0.3,
                  height_ratios=[1.0, 1.0, 1.25])

    # Create plots for D=1, 2, 3
    dimensions = [1, 2, 3]
    for row_idx, dim in enumerate(dimensions):
        ax_left = fig.add_subplot(gs[row_idx, 0])
        ax_right = fig.add_subplot(gs[row_idx, 1])

        try:
            visualize_noise_field(dim, ax_left, ax_right)
        except Exception as e:
            print(f"Error generating visualization for D={dim}: {e}")
            ax_left.text(0.5, 0.5, f'Error: {e}', ha='center', va='center',
                         transform=ax_left.transAxes)
            ax_right.text(0.5, 0.5, f'Error: {e}', ha='center', va='center',
                          transform=ax_right.transAxes)

    # Overall figure title
    fig.suptitle('Mapping from virtual noise field to hidden-layer noise vectors ($N_h=64$)',
                 fontsize=14, fontweight='bold', y=0.995)

    # Save figure
    output_path = "../fig/noise_virtual_to_hidden.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    # Also save as PNG for preview
    output_path_png = "../fig/noise_virtual_to_hidden.png"
    plt.savefig(output_path_png, dpi=150, bbox_inches='tight')
    print(f"Figure also saved to {output_path_png}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
