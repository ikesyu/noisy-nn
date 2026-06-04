from multidimensional_storage_generate_job import get_job_string
from multidimensional_storage_train import get_filename
from multidimensional_storage_functions import SineFunctions, SquineFunctions, TrineFunctions, PlainSineFunctions, gray_ndindex, plain_sine
from multidimensional_storage import Structure, fractional_index, noise_pattern
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

sys.path.append("../")


def build_noise_panels(ax, stdvec, title):
    stdvec_np = stdvec.detach().cpu().numpy()
    hidden_layer_viz = np.tile(stdvec_np, (2, 1))
    im = ax.imshow(hidden_layer_viz, cmap="viridis",
                   aspect="auto", interpolation="nearest")
    # ax.set_title(title)
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("Hidden layer")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Layer 1", "Layer 2"])
    for j in range(0, hidden_layer_viz.shape[1] + 1):
        ax.axvline(j - 0.5, color="gray", linewidth=0.35, alpha=0.45)
    ax.axhline(0.5, color="gray", linewidth=1.2, alpha=0.9)
    return im


def build_virtual_panel(ax, stdvec, noise_shape, title):
    virtual = stdvec.detach().cpu().numpy().reshape(noise_shape)
    is_1d = len(noise_shape) == 1
    if is_1d:
        virtual = virtual.reshape(1, -1)
        ax.set_yticks([])
    elif len(noise_shape) == 3:
        # Show the central slice for compact 2D visualization.
        virtual = virtual[noise_shape[0] // 2]
    im = ax.imshow(virtual, cmap="viridis", origin="lower",
                   aspect="auto", interpolation="nearest", extent=[0, 1, 0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.0, 0.5, 1.0])
    if not is_1d:
        ax.set_yticks([0.0, 0.5, 1.0])
    # ax.set_title(title)
    ax.set_xlabel("Virtual axis 1")
    if is_1d:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Virtual axis 2")
    return im


def parse_index(index_vals, shape):
    if len(index_vals) != len(shape):
        raise ValueError(
            f"Index length mismatch: index has {
                len(index_vals)} dims but function shape is {shape}."
        )
    idx = tuple(index_vals)
    for i, n in zip(idx, shape):
        if i < 0 or i >= n:
            raise ValueError(f"Index {idx} is out of range for shape {shape}.")
    return idx


def build_functions(args):
    if args.construct_structure is None:
        if args.function_type == "plainsine":
            args.construct_structure = list(
                args.function_structure) + [1] * max(0, 3 - len(args.function_structure))
        else:
            args.construct_structure = args.function_structure

    if args.function_type == "sine":
        return SineFunctions(
            args.function_structure,
            shuffle=args.shuffle,
            epochs=args.epochs,
            shuffle_learning=args.shuffle_learning,
        )
    if args.function_type == "squine":
        return SquineFunctions(
            construct_shape=args.construct_structure,
            present_shape=args.function_structure,
            shuffle=args.shuffle,
            epochs=args.epochs,
            shuffle_learning=args.shuffle_learning,
        )
    if args.function_type == "trine":
        return TrineFunctions(
            construct_shape=args.construct_structure,
            present_shape=args.function_structure,
            shuffle=args.shuffle,
            epochs=args.epochs,
            shuffle_learning=args.shuffle_learning,
        )
    if args.function_type == "plainsine":
        return PlainSineFunctions(
            construct_shape=args.construct_structure,
            present_shape=args.function_structure,
            shuffle=args.shuffle,
            epochs=args.epochs,
            shuffle_learning=args.shuffle_learning,
        )
    raise ValueError(f"Unknown --function_type: {args.function_type}")


def move_stdvecs_to_device(stdvecs, device):
    return [v.to(device) for v in stdvecs]


def retrieve(structure, functions):
    # Same cache behavior as multidimensional_storage_plot.retrieve.
    fname = get_filename(structure, functions)
    if Path(fname).is_file():
        return torch.load(fname, weights_only=False, map_location=torch.device('cpu'))
    print(get_job_string(structure, functions, cuda=False))
    return {"model": None, "losses": [np.nan]}


def map_present_to_construct(functions, net=None, structure=None):
    """Recover construct index for each present index.

    When ``net`` and ``structure`` are provided, the mapping is derived from
    the network's own outputs (this is the *actual* function the model learned
    for each index, regardless of any unseeded shuffle that happened during
    training). Otherwise we fall back to matching ``functions.ys`` against
    fresh ``plain_sine`` outputs (only correct when the rebuilt ``functions``
    happens to share the training shuffle).
    """
    if not hasattr(functions, "construct_shape"):
        return None
    if getattr(functions, "function_type", None) != "plainsine":
        # Only implemented for PlainSineFunctions for now.
        return None

    construct_indices = list(gray_ndindex(functions.construct_shape))
    candidates = plain_sine(functions.x, *functions.construct_shape)
    if len(candidates) != len(construct_indices):
        return None

    cand_arr = np.array([np.asarray(c).reshape(-1) for c in candidates])

    if net is not None and structure is not None:
        dev = next(net.parameters()).device
        x_tensor = functions.x_tensor.to(dev)
        targets = {}
        with torch.no_grad():
            for present_idx in np.ndindex(tuple(functions.shape)):
                stdvecs = noise_pattern(
                    structure, fractional_index(present_idx, functions.shape))
                stdvecs = [v.to(dev) for v in stdvecs]
                out = net(x_tensor, stdvecs).detach().cpu().numpy().reshape(-1)
                targets[tuple(present_idx)] = out
    else:
        ys_np = functions.ys.detach().cpu().numpy()
        targets = {tuple(idx): ys_np[idx].reshape(-1)
                   for idx in np.ndindex(tuple(functions.shape))}

    mapping = {}
    for present_idx, target in targets.items():
        diffs = np.linalg.norm(cand_arr - target[None, :], axis=1)
        best = int(np.argmin(diffs))
        mapping[present_idx] = construct_indices[best]
    return mapping


def pick_plainsine_indices(functions, net=None, structure=None):
    mapping = map_present_to_construct(functions, net=net, structure=structure)
    if mapping is None:
        return None

    inv_mapping = {c: p for p, c in mapping.items()}
    cshape = tuple(functions.construct_shape)
    if len(cshape) < 2:
        return None

    # Prefer same phase/amp and far-apart frequency so frequency change is obvious.
    ca = tuple([0, 0] + [0] * (len(cshape) - 2))
    cb = tuple([0, cshape[1] - 1] + [0] * (len(cshape) - 2))
    if ca in inv_mapping and cb in inv_mapping:
        return inv_mapping[ca], inv_mapping[cb]

    # Fallback: pick first pair with different frequency index.
    items = list(mapping.items())
    for i in range(len(items)):
        p1, c1 = items[i]
        for j in range(i + 1, len(items)):
            p2, c2 = items[j]
            if c1[1] != c2[1]:
                return p1, p2
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Plot two cached noise patterns with dashed target and solid model output."
    )
    parser.add_argument("--noise_structure", nargs="+",
                        type=int, default=[64])
    parser.add_argument("--function_structure",
                        nargs="+", type=int, default=[10])
    parser.add_argument("--construct_structure",
                        nargs="+", type=int, default=None)
    parser.add_argument(
        "--function_type", choices=["sine", "squine", "trine", "plainsine"], default="plainsine")
    parser.add_argument("--activation_ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument(
        "--shuffle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shuffle_learning",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--index_a", nargs="+", type=int, default=[1])
    parser.add_argument("--index_b", nargs="+", type=int, default=[3])
    parser.add_argument(
        "--example_2d", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--squash", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.example_2d:
        # Preset matching common cached 2D runs.
        args.noise_structure = [8, 8]
        args.function_structure = [10, 10]
        args.function_type = "plainsine"
        args.construct_structure = None

        if len(args.index_a) == 1:
            args.index_a = [1, 8]
        if len(args.index_b) == 1:
            args.index_b = [6, 3]

    if args.squash:
        # Squash mode: 2D function space mapped onto 1D virtual/noise space.
        args.noise_structure = [64]
        args.function_structure = [100]
        args.construct_structure = [10, 10, 1]
        if len(args.index_a) == 1:
            args.index_a = [1, 8]
        if len(args.index_b) == 1:
            args.index_b = [6, 3]

    structure = Structure(args.noise_structure,
                          activation_ratio=args.activation_ratio)
    functions = build_functions(args)

    result = retrieve(structure, functions)
    net = result["model"]
    if net is None:
        # Match multidimensional_storage_plot behavior: emit missing-job command and exit.
        return
    model_device = next(net.parameters()).device

    # if args.example_2d and args.function_type == "plainsine" and len(args.index_a) == 2 and len(args.index_b) == 2:
    #     picked = pick_plainsine_indices(
    #         functions, net=net, structure=structure)
    #     if picked is not None:
    #         args.index_a = list(picked[0])
    #         args.index_b = list(picked[1])

    idx_a = parse_index(args.index_a, functions.shape)
    idx_b = parse_index(args.index_b, functions.shape)

    present_to_construct = map_present_to_construct(
        functions, net=net, structure=structure)
    cidx_a = present_to_construct[idx_a] if present_to_construct and idx_a in present_to_construct else None
    cidx_b = present_to_construct[idx_b] if present_to_construct and idx_b in present_to_construct else None

    stdvecs_a = noise_pattern(
        structure, fractional_index(idx_a, functions.shape))
    stdvecs_b = noise_pattern(
        structure, fractional_index(idx_b, functions.shape))
    stdvecs_a = move_stdvecs_to_device(stdvecs_a, model_device)
    stdvecs_b = move_stdvecs_to_device(stdvecs_b, model_device)
    x_tensor = functions.x_tensor.to(model_device)

    with torch.no_grad():
        y_pred_a = net(x_tensor, stdvecs_a).detach().cpu().numpy()
        y_pred_b = net(x_tensor, stdvecs_b).detach().cpu().numpy()

    def target_for(idx, cidx):
        # Prefer the construct function the model actually learned at this idx
        # (recovered from the model's output) over functions.ys[idx], because
        # the training shuffle isn't persisted in the cache.
        if cidx is not None and getattr(functions, "function_type", None) == "plainsine":
            candidates = plain_sine(functions.x, *functions.construct_shape)
            construct_indices = list(gray_ndindex(functions.construct_shape))
            k = construct_indices.index(cidx)
            return np.asarray(candidates[k]).reshape(-1, 1)
        return functions.ys[idx].detach().cpu().numpy()

    y_true_a = target_for(idx_a, cidx_a)
    y_true_b = target_for(idx_b, cidx_b)
    x = functions.x

    def save_cropped_pdf(fig, pdf_path):
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved figure to {pdf_path}")

    fig_va, ax_va = plt.subplots(figsize=(4.4, 3.6), constrained_layout=True)
    im_va = build_virtual_panel(ax_va, stdvecs_a[0], structure.noise_structure,
                                f"Virtual field A (idx={idx_a}, cidx={cidx_a})")
    fig_va.colorbar(im_va, ax=ax_va, label="Noise intensity")

    fig_vb, ax_vb = plt.subplots(figsize=(4.4, 3.6), constrained_layout=True)
    im_vb = build_virtual_panel(ax_vb, stdvecs_b[0], structure.noise_structure,
                                f"Virtual field B (idx={idx_b}, cidx={cidx_b})")
    fig_vb.colorbar(im_vb, ax=ax_vb, label="Noise intensity")

    print(f"# {args.index_a=} {args.index_b=} {idx_a=} {idx_b=}")

    fig_ha, ax_ha = plt.subplots(figsize=(6.0, 2.6), constrained_layout=True)
    im_ha = build_noise_panels(ax_ha, stdvecs_a[0], "Flattened hidden noise A")
    fig_ha.colorbar(im_ha, ax=ax_ha, label="Noise intensity")

    fig_hb, ax_hb = plt.subplots(figsize=(6.0, 2.6), constrained_layout=True)
    im_hb = build_noise_panels(ax_hb, stdvecs_b[0], "Flattened hidden noise B")
    fig_hb.colorbar(im_hb, ax=ax_hb, label="Noise intensity")

    fig_reg, ax_reg = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax_reg.plot(x, y_pred_a, "-", color="tab:blue", linewidth=2.0,
                label="Network output A")
    ax_reg.plot(x, y_pred_b, "-", color="tab:orange", linewidth=2.0,
                label="Network output B")
    ax_reg.plot(x, y_true_a, ".", color="darkblue", linewidth=2.0,
                label="Target A")
    ax_reg.plot(x, y_true_b, ".", color="brown", linewidth=2.0,
                label="Target B")

    ax_reg.set_xlabel("x")
    ax_reg.set_ylabel("y")
    ax_reg.set_xlim(-2 * np.pi, 2 * np.pi)
    ax_reg.grid(True, alpha=0.3)
    ax_reg.legend(loc="upper right")

    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../fig"))
    os.makedirs(out_dir, exist_ok=True)

    dims = f"{len(args.function_structure)}{len(args.noise_structure)}"
    save_cropped_pdf(
        fig_va,
        os.path.join(out_dir, f"regression_virtualA{dims}.pdf"),
    )
    save_cropped_pdf(
        fig_vb,
        os.path.join(out_dir, f"regression_virtualB{dims}.pdf"),
    )
    save_cropped_pdf(
        fig_ha,
        os.path.join(out_dir, f"regression_hiddenA{dims}.pdf"),
    )
    save_cropped_pdf(
        fig_hb,
        os.path.join(out_dir, f"regression_hiddenB{dims}.pdf"),
    )
    save_cropped_pdf(
        fig_reg,
        os.path.join(out_dir, f"regression_plot{dims}.pdf"),
    )

    if args.show:
        plt.show()
    else:
        plt.close(fig_va)
        plt.close(fig_vb)
        plt.close(fig_ha)
        plt.close(fig_hb)
        plt.close(fig_reg)


if __name__ == "__main__":
    main()
