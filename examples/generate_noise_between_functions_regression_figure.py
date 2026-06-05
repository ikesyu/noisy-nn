from multidimensional_storage_generate_job import get_job_string
from multidimensional_storage_train import get_filename
from multidimensional_storage_functions import (
    PlainSineFunctions,
    SineFunctions,
    SquineFunctions,
    TrineFunctions,
    gray_ndindex,
    plain_sine,
)
from multidimensional_storage import Structure, noise_pattern
import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

sys.path.append("../")


def build_functions(args):
    if args.construct_structure is None:
        if args.function_type == "plainsine":
            args.construct_structure = list(args.function_structure) + [1] * max(
                0, 3 - len(args.function_structure)
            )
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


def retrieve(structure, functions):
    fname = get_filename(structure, functions, seed=None)
    if Path(fname).is_file():
        return torch.load(fname, weights_only=False, map_location=torch.device("cpu"))
    print(get_job_string(structure, functions, cuda=False))
    return {"model": None, "losses": [np.nan]}


def retrieve_seeded(structure, functions, seed=None):
    fname = get_filename(structure, functions, seed=seed)
    if Path(fname).is_file():
        return torch.load(fname, weights_only=False, map_location=torch.device("cpu"))
    seed_arg = f" --seed {int(seed)}" if seed is not None else ""
    print(get_job_string(structure, functions, cuda=False) + seed_arg)
    return {"model": None, "losses": [np.nan]}


def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def c_label_from_idx(idx, count):
    den = max(count - 1, 1)
    return f"c={idx / den:.3f}"


def c_label_from_pos(pos, count):
    den = max(count - 1, 1)
    return f"c={pos / den:.3f}"


def c_fraction(pos, count):
    den = max(count - 1, 1)
    return pos / den


def cached_target_for_idx(result, idx):
    ys = result.get("ys")
    if ys is None:
        return None
    if isinstance(ys, torch.Tensor):
        return ys[idx].detach().cpu().numpy().reshape(-1)
    return np.asarray(ys[idx]).reshape(-1)


def map_present_to_construct(functions, net, structure):
    if not hasattr(functions, "construct_shape"):
        return None
    if getattr(functions, "function_type", None) != "plainsine":
        return None

    construct_indices = list(gray_ndindex(functions.construct_shape))
    candidates = plain_sine(functions.x, *functions.construct_shape)
    if len(candidates) != len(construct_indices):
        return None

    cand_arr = np.array([np.asarray(c).reshape(-1) for c in candidates])
    dev = next(net.parameters()).device
    x_tensor = functions.x_tensor.to(dev)

    targets = {}
    with torch.no_grad():
        for present_idx in np.ndindex(tuple(functions.shape)):
            frac = tuple(i / (n - 1) if n > 1 else 0.0 for i,
                         n in zip(present_idx, functions.shape))
            stdvecs = [v.to(dev) for v in noise_pattern(structure, frac)]
            out = net(x_tensor, stdvecs).detach().cpu().numpy().reshape(-1)
            targets[tuple(present_idx)] = out

    mapping = {}
    for present_idx, target in targets.items():
        diffs = np.linalg.norm(cand_arr - target[None, :], axis=1)
        best = int(np.argmin(diffs))
        mapping[present_idx] = construct_indices[best]
    return mapping


def target_for_idx(functions, idx, present_to_construct):
    cidx = None
    if present_to_construct is not None and idx in present_to_construct:
        cidx = present_to_construct[idx]

    if cidx is not None and getattr(functions, "function_type", None) == "plainsine":
        candidates = plain_sine(functions.x, *functions.construct_shape)
        construct_indices = list(gray_ndindex(functions.construct_shape))
        k = construct_indices.index(cidx)
        return np.asarray(candidates[k]).reshape(-1, 1)
    return functions.ys[idx].detach().cpu().numpy()


def parse_positions(pos_text, count):
    values = [float(p.strip()) for p in pos_text.split(",") if p.strip()]
    if not values:
        raise ValueError("--positions must include at least one value")
    lo = 0.0
    hi = float(count - 1)
    for v in values:
        if v < lo or v > hi:
            raise ValueError(f"Position {v} out of range [{lo}, {hi}]")
    return values


def eval_predictions(net, x_tensor, structure, function_count, positions):
    preds = {}
    stdvec_rows = []
    dev = next(net.parameters()).device

    for pos in positions:
        frac = (pos / (function_count - 1),)
        stdvecs = [v.to(dev) for v in noise_pattern(structure, frac)]
        with torch.no_grad():
            y = net(x_tensor, stdvecs).detach().cpu().numpy().reshape(-1)
        preds[pos] = y
        stdvec_rows.append(stdvecs[0].detach().cpu().numpy().reshape(-1))

    return preds, np.vstack(stdvec_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Regression for interpolated noise positions between consecutive 1D functions."
    )
    parser.add_argument("--noise_structure", nargs="+", type=int, default=[64])
    parser.add_argument("--function_structure",
                        nargs="+", type=int, default=[5])
    parser.add_argument("--construct_structure",
                        nargs="+", type=int, default=None)
    parser.add_argument(
        "--function_type",
        choices=["sine", "squine", "trine", "plainsine"],
        default="plainsine",
    )
    parser.add_argument("--activation_ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument(
        "--shuffle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--shuffle_learning", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--index_a", type=int, default=1)
    parser.add_argument("--index_b", type=int, default=2)
    parser.add_argument(
        "--positions",
        type=str,
        default=",".join([f"{a}" for a in np.linspace(1, 2, 5)]),
        help="Comma-separated positions in index-space.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.seed is not None:
        set_global_seed(args.seed)

    if len(args.noise_structure) != 1 or len(args.function_structure) != 1:
        raise ValueError(
            "This script currently supports only 1D noise and 1D function spaces.")

    if args.index_a < 0 or args.index_a >= args.function_structure[0]:
        raise ValueError("--index_a is out of range")
    if args.index_b < 0 or args.index_b >= args.function_structure[0]:
        raise ValueError("--index_b is out of range")

    structure = Structure(args.noise_structure,
                          activation_ratio=args.activation_ratio)
    functions = build_functions(args)
    cache_seed = args.seed if args.shuffle else None
    result = retrieve_seeded(structure, functions, seed=cache_seed)
    net = result["model"]
    if net is None:
        return

    positions = parse_positions(args.positions, args.function_structure[0])
    model_device = next(net.parameters()).device
    x_tensor = functions.x_tensor.to(model_device)
    x = functions.x

    y_preds, noise_rows = eval_predictions(
        net, x_tensor, structure, args.function_structure[0], positions
    )

    idx_a = (args.index_a,)
    idx_b = (args.index_b,)
    y_true_a = cached_target_for_idx(result, idx_a)
    y_true_b = cached_target_for_idx(result, idx_b)
    if y_true_a is None or y_true_b is None:
        present_to_construct = map_present_to_construct(functions, net, structure)
        y_true_a = target_for_idx(
            functions, idx_a, present_to_construct).reshape(-1)
        y_true_b = target_for_idx(
            functions, idx_b, present_to_construct).reshape(-1)
    c_a = c_label_from_idx(args.index_a, args.function_structure[0])
    c_b = c_label_from_idx(args.index_b, args.function_structure[0])

    fig_reg, ax_reg = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)

    cmap = plt.get_cmap("viridis")
    pos_colors = [cmap(v) for v in np.linspace(0.15, 0.9, len(positions))]

    for pos, color in zip(positions, pos_colors):
        c_pos = c_label_from_pos(pos, args.function_structure[0])
        ax_reg.plot(
            x,
            y_preds[pos],
            color=color,
            linewidth=2.0,
            label=f"noise ({c_pos})",
        )

    ax_reg.plot(
        x,
        y_true_a,
        ".",
        color="tab:blue",
        markersize=4,
        alpha=0.9,
        label=f"target ({c_a})",
    )
    ax_reg.plot(
        x,
        y_true_b,
        ".",
        color="tab:orange",
        markersize=4,
        alpha=0.9,
        label=f"target ({c_b})",
    )

    ax_reg.set_xlabel("x")
    ax_reg.set_ylabel("y")
    ax_reg.set_xlim(-2 * np.pi, 2 * np.pi)
    ax_reg.grid(True, alpha=0.3)
    ax_reg.legend(loc="upper right", ncol=2, fontsize=9)
    # ax_reg.set_title(
    #    "Interpolated noise regression between consecutive functions"
    # )

    fig_noise, ax_noise = plt.subplots(
        figsize=(8.2, 2.7), constrained_layout=True)
    im = ax_noise.imshow(noise_rows, aspect="auto",
                         cmap="viridis", interpolation="nearest")
    ax_noise.set_xlabel("Neuron index")
    ax_noise.set_ylabel("Center position c")
    ax_noise.set_yticks(np.arange(len(positions)))
    ax_noise.set_yticklabels(
        [f"{c_fraction(p, args.function_structure[0]):.3f}" for p in positions])

    for j in range(0, noise_rows.shape[1] + 1):
        ax_noise.axvline(j - 0.5, color="gray", linewidth=0.35, alpha=0.25)
    for i in range(0, noise_rows.shape[0] + 1):
        ax_noise.axhline(i - 0.5, color="gray", linewidth=0.35, alpha=0.25)

    fig_noise.colorbar(im, ax=ax_noise, label="Noise intensity")

    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../fig"))
    os.makedirs(out_dir, exist_ok=True)
    seed_suffix = f"_seed{args.seed}" if cache_seed is not None else ""
    shuffle_suffix = "_shuffle" if args.shuffle else ""
    out_name = f"between{shuffle_suffix}{seed_suffix}.pdf"
    out_path = os.path.join(out_dir, out_name)
    fig_reg.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved figure to {out_path}")

    between_path = os.path.join(out_dir, "between_activation.pdf")
    fig_noise.savefig(between_path, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved figure to {between_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig_reg)
        plt.close(fig_noise)


if __name__ == "__main__":
    main()
