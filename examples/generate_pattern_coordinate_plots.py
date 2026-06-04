import os
from types import SimpleNamespace

import matplotlib.pyplot as plt

from generate_noise_virtual_to_hidden_regression_figure import (
    build_functions,
    build_noise_panels,
    build_virtual_panel,
)
from multidimensional_storage import Structure, fractional_index, noise_pattern

def main():
    # Fixed setup requested by user.
    function_structure = [10, 10]
    noise_structure = [8, 8]
    coordinates = [(2, 3), (9, 8), (2, 8), (9, 3)]

    args = SimpleNamespace(
        function_type="plainsine",
        function_structure=function_structure,
        construct_structure=None,
        shuffle=False,
        epochs=10000,
        shuffle_learning=False,
    )

    structure = Structure(noise_structure, activation_ratio=0.5)
    functions = build_functions(args)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../fig"))
    os.makedirs(out_dir, exist_ok=True)

    for idx in coordinates:
        if any(i < 0 or i >= n for i, n in zip(idx, functions.shape)):
            raise ValueError(f"Coordinate {idx} is out of range for function shape {functions.shape}")

        stdvec = noise_pattern(structure, fractional_index(idx, functions.shape))[0]

        fig_virtual, ax_virtual = plt.subplots(figsize=(4.4, 3.6), constrained_layout=True)
        im_v = build_virtual_panel(ax_virtual, stdvec, structure.noise_structure, "")
        cbar_v = fig_virtual.colorbar(im_v, ax=ax_virtual, label="Activation")
        cbar_v.ax.tick_params(labelsize=8)

        fig_neuron, ax_neuron = plt.subplots(figsize=(6.0, 2.6), constrained_layout=True)
        im_n = build_noise_panels(ax_neuron, stdvec, "")
        cbar_n = fig_neuron.colorbar(im_n, ax=ax_neuron, label="Activation")
        cbar_n.ax.tick_params(labelsize=8)

        out_virtual_pdf = os.path.join(out_dir, f"pattern_virtual_{idx[0]}_{idx[1]}.pdf")
        out_neuron_pdf = os.path.join(out_dir, f"pattern_neuron_{idx[0]}_{idx[1]}.pdf")
        fig_virtual.savefig(out_virtual_pdf, bbox_inches="tight", pad_inches=0.02)
        fig_neuron.savefig(out_neuron_pdf, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig_virtual)
        plt.close(fig_neuron)
        print(f"Saved figure to {out_virtual_pdf}")
        print(f"Saved figure to {out_neuron_pdf}")


if __name__ == "__main__":
    main()
