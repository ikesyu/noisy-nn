"""Demonstrates the functions in nnn/noise_field.py.

Walks through the noise-field building blocks one by one:
compute_sigma -> gaussian_fill -> Structure/noise_pattern/fractional_index
-> noise_pattern_table, in both 1D and 2D, and simply displays the results
with plt.show() (nothing is saved to disk).
"""

import sys  # noqa
sys.path.append("../")  # noqa

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from nnn import noise_field  # noqa


def set_colors(N):
    # hue: blue (2/3) -> red (0)
    h = np.linspace(2/3, 0, N)
    colors = hsv_to_rgb(np.column_stack([h, np.ones(N), np.ones(N)]))
    plt.gca().set_prop_cycle(color=colors)


def demo_compute_sigma():
    """compute_sigma(): how the Gaussian spread grows with the target activation ratio."""
    plt.figure()
    ps = np.linspace(0.05, 0.95, 30)
    set_colors(3)
    for d in [1, 2, 3]:
        sigmas = [noise_field.compute_sigma(0.1, p, d) for p in ps]
        plt.plot(ps, sigmas, label=f"d={d}")
    plt.title("compute_sigma: spread vs. activation ratio")
    plt.xlabel("activation ratio (p)")
    plt.ylabel("sigma")
    plt.legend()
    plt.grid(True)


def demo_gaussian_fill_1d():
    """gaussian_fill() in 1D for several centers, using compute_sigma for the spread."""
    plt.figure()
    centers = np.linspace(0.1, 0.9, 5)
    sigma = noise_field.compute_sigma(0.1, 0.4, 1)
    set_colors(len(centers))
    for c in centers:
        field = noise_field.gaussian_fill(
            shape=[128], center=[c], sigma=[sigma])
        plt.plot(field.numpy(), label=f"center={c:.2f}")
    plt.title("gaussian_fill: 1D noise field")
    plt.xlabel("unit index")
    plt.ylabel("amplitude")
    plt.legend()
    plt.grid(True)


def demo_gaussian_fill_2d():
    """gaussian_fill() in 2D for several centers, shown as a row of images."""
    centers = [(0.2, 0.2), (0.5, 0.5), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
    sigma = noise_field.compute_sigma(0.1, 0.15, 2)
    fig, axes = plt.subplots(1, len(centers), figsize=(3*len(centers), 3))
    for ax, c in zip(axes, centers):
        field = noise_field.gaussian_fill(
            shape=[32, 32], center=c, sigma=[sigma, sigma])
        ax.imshow(field.numpy(), vmin=0, vmax=1)
        ax.set_title(f"center={c}")
        ax.axis("off")
    fig.suptitle("gaussian_fill: 2D noise field for different centers")


def demo_fractional_index_and_noise_pattern():
    """fractional_index() + Structure/noise_pattern() called directly for one grid cell."""
    structure = noise_field.Structure([16, 16], activation_ratio=0.25)
    shape = (4, 4)
    idx = (1, 2)
    center = noise_field.fractional_index(idx, shape)
    stdvecs = noise_field.noise_pattern(structure, center)
    field = stdvecs[0].numpy().reshape(structure.noise_structure)

    plt.figure()
    plt.imshow(field, vmin=0, vmax=1)
    rounded_center = tuple(round(c, 2) for c in center)
    plt.title(f"noise_pattern: idx={idx} -> center={rounded_center}")
    plt.colorbar(label="noise std")


def demo_noise_pattern_table_1d():
    """Structure + noise_pattern_table() in 1D: noise vectors across the storage grid."""
    structure = noise_field.Structure([64], activation_ratio=0.3)
    shape = (7,)
    table = noise_field.noise_pattern_table(structure, shape)

    plt.figure()
    set_colors(shape[0])
    for idx in np.ndindex(shape):
        vec = table[idx][0].numpy()
        plt.plot(vec, label=f"idx={idx}")
    plt.title("noise_pattern_table: 1D noise vectors across the storage grid")
    plt.xlabel("unit index")
    plt.ylabel("noise std")
    plt.legend()
    plt.grid(True)


def demo_noise_pattern_table_2d():
    """Structure + noise_pattern_table() in 2D, one image per storage-grid cell."""
    structure = noise_field.Structure([16, 16], activation_ratio=0.2)
    shape = (3, 3)
    table = noise_field.noise_pattern_table(structure, shape)

    fig, axes = plt.subplots(*shape, figsize=(3*shape[1], 3*shape[0]))
    for idx in np.ndindex(shape):
        vec = table[idx][0].numpy().reshape(structure.noise_structure)
        ax = axes[idx]
        ax.imshow(vec, vmin=0, vmax=1)
        ax.set_title(f"idx={idx}")
        ax.axis("off")
    fig.suptitle("noise_pattern_table: 2D noise fields across a 3x3 storage grid")


if __name__ == "__main__":
    demo_compute_sigma()
    demo_gaussian_fill_1d()
    demo_gaussian_fill_2d()
    demo_fractional_index_and_noise_pattern()
    demo_noise_pattern_table_1d()
    demo_noise_pattern_table_2d()
    plt.show()
