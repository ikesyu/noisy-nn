"""neuromod_lesion -- multiplexing or partition?  Lesion the shared units and see.

This is the L2 experiment (`docs/idea_neuromod.md` sections 1.1 and 7.3), the one
claim in the neuromodulation line with no evidence behind it yet.

The competing hypotheses make opposite predictions about one manipulation:

    PARTITION    each behaviour lives in its own private region of the shared
                 weight set, so the fields are really a look-up over disjoint
                 sub-networks.  Then damaging the units two fields have in common
                 either does nothing (there are none) or degrades at most one of
                 the two behaviours.
    MULTIPLEXING the behaviours are superposed on overlapping units.  Then the
                 shared units carry BOTH, and damaging them degrades BOTH at once,
                 while private units degrade only their own behaviour.

Two definitions decide whether this experiment means anything, and the naive
choice fails for both (`docs/idea_core.md` sections 3.3-3.5, 4.8):

    recruited set   S_i = {k : nu_k > 0}, the support of the CROSSING RATE.  Not
                    {k : sigma_k > c}: the absolute value of sigma is gauge
                    dependent, so a sigma-thresholded set moves under a gauge
                    transformation that leaves the output bit-for-bit identical,
                    and any Jaccard computed from it is not a function of the
                    network.  nu is homogeneous of degree zero, hence invariant.
    lesion          the kill triple: sigma_k <- 0, h_k <- H_DEAD, W[l+1][:,k] <- 0.
                    Setting sigma_k = 0 alone does not silence a unit past the
                    first hidden layer, so a sigma-only lesion reports "damaged but
                    still working" and refutes nothing.

Controls: private-unit lesions (should be specific), and random lesions of the
same size drawn from the union of the supports (controls for "we removed some
units" rather than "we removed the shared ones").

    .venv/bin/python tmp/neuromod_lesion.py                       # default, ~2 min
    .venv/bin/python tmp/neuromod_lesion.py --seeds 3 --save       # with error bars
    .venv/bin/python tmp/neuromod_lesion.py --radii 0.16,0.22,0.28,0.34 --save
                                                                  # overlap sweep
    .venv/bin/python tmp/neuromod_lesion.py --model sample --save  # mechanism level

Output: tmp/out/<tag>_bars.png, tmp/out/<tag>_overlap.png (with --radii),
        tmp/out/<tag>.csv
Context: `docs/idea_neuromod.md` section 7.3.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = Path(__file__).resolve().parent
for _p in (str(ROOT), str(TMP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from neuromod import fields as F
from neuromod import protocol as P
from neuromod import viz, world

# Which hidden layer the lesion targets.  Layer 0 is where a zero-sigma unit is
# genuinely silent, so it is the honest place to both define the recruited set and
# apply the lesion; --hidden-layers 1 removes the question entirely.
LESION_LAYER = 0


def mix_key(mix) -> float:
    """Numeric key for an alpha mix; -1 stands for the default near-one-hot targets."""
    return -1.0 if mix is None else float(mix)


def build_data(args, mix):
    """Scripted scene -> observations and the three per-state targets.

    `mix` raises the off-drives, which is what makes the three behaviours compete
    for the same input dimensions.  Under the default near-one-hot targets each
    behaviour reads a different slice of the 6D input, so the network can solve all
    three WITHOUT sharing anything, and multiplexing is not required of it
    (`docs/idea_neuromod.md` section 4).  Sweeping this is therefore the axis that
    decides whether the shared units have any reason to carry both behaviours.
    """
    objects = world.make_scripted_objects()
    alphas = world.alpha_states(mix)
    obs_np = world.encode_observations(
        world.make_training_grid(args.grid_side), objects)
    obs = torch.tensor(obs_np, dtype=torch.float32)
    targets = {s: torch.tensor(
                   world.make_mixed_behavior_targets(obs_np, alphas[s],
                                                     gamma=args.target_gamma),
                   dtype=torch.float32)
               for s in world.STATES}
    return obs, targets


def train_one(seed: int, radius: float, obs, targets, args):
    """One trained network plus the fields it was trained under."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    fields = F.build_fields(world.CATEGORIES, args.hidden_dim, args.base_std,
                            args.sigma, args.theta, radius=radius)
    state_fields = {s: fields[world.STATE_TO_FIELD[s]] for s in world.STATES}
    net = P.build_network(args.hidden_dim, n_hidden=args.hidden_layers,
                          base_std=args.base_std, kind=args.model,
                          t=args.samples, crossing_h=args.crossing_h)
    P.train(net, obs, targets, state_fields, world.STATES, args.hidden_layers,
            args.epochs, args.lr, verbose=False)
    return net, fields


def per_behaviour_error(net, fields, obs, targets, args) -> dict[str, float]:
    """Task error of each behaviour under its own (possibly lesioned) field."""
    state_fields = {s: fields[world.STATE_TO_FIELD[s]] for s in world.STATES}
    losses = P.final_losses(net, obs, targets, state_fields, world.STATES,
                            args.hidden_layers)
    return {world.STATE_TO_FIELD[s]: v for s, v in losses.items()}


def lesion_error(net, fields, mask, obs, targets, args) -> dict[str, float]:
    """Per-behaviour error after killing `mask` on a private copy of the network."""
    net_l = copy.deepcopy(net)
    fields_l = {k: v.clone() for k, v in fields.items()}
    F.kill_units(net_l, mask, fields_l, layer=LESION_LAYER)
    return per_behaviour_error(net_l, fields_l, obs, targets, args)


def _draws(pool: np.ndarray, k: int, n_draws: int, shape, rng):
    """`n_draws` random subsets of `k` units from `pool`, as boolean masks."""
    out = []
    for _ in range(n_draws):
        pick = np.zeros(shape, dtype=bool)
        if k and pool.size:
            pick[rng.choice(pool, size=min(k, pool.size), replace=False)] = True
        out.append(pick)
    return out


def lesion_groups(masks: dict[str, np.ndarray], a: str, b: str, rng, n_draws: int):
    """The four contrasted unit groups for the pair (a, b), ALL of the same size.

    Size matching matters.  The shared set is usually smaller than either private
    set, so comparing raw damage between differently sized lesions would confound
    "where we cut" with "how much we cut".  Every group here has |shared| units, and
    the private and random groups are averaged over several draws.

    A caveat the results must not overclaim: a private lesion is specific *by
    construction* under sigma-only recruitment, because units outside a behaviour's
    support are already silent when that behaviour is evaluated, so their removal
    changes it by exactly zero.  The private groups are therefore a SCALE, telling
    us what k units of a behaviour's own territory are worth; they are not evidence.
    The evidence is whether the shared lesion damages BOTH behaviours, and by how
    much relative to that scale.
    """
    shared = masks[a] & masks[b]
    k = int(shared.sum())
    shape = shared.shape
    return {
        "shared": [shared],
        f"{a}-only": _draws(np.flatnonzero(masks[a] & ~masks[b]), k, n_draws, shape, rng),
        f"{b}-only": _draws(np.flatnonzero(masks[b] & ~masks[a]), k, n_draws, shape, rng),
        "random": _draws(np.flatnonzero(masks[a] | masks[b]), k, n_draws, shape, rng),
    }


def run_cell(seed: int, radius: float, obs, targets, args, rng):
    """Train one network for this condition and lesion every group."""
    net, fields = train_one(seed, radius, obs, targets, args)
    report = F.overlap_report(net, obs, fields, world.CATEGORIES,
                              args.hidden_layers, layer=LESION_LAYER)
    base = per_behaviour_error(net, fields, obs, targets, args)

    rows = []
    for a, b in itertools.combinations(world.CATEGORIES, 2):
        groups = lesion_groups(report["masks"], a, b, rng, args.random_draws)
        for name, masks in groups.items():
            damaged = [lesion_error(net, fields, m, obs, targets, args) for m in masks]
            rows.append({
                "seed": seed, "radius": radius, "alpha_mix": mix_key(args.alpha_mix),
                "pair": f"{a}|{b}", "group": name,
                "n_units": int(masks[0].sum()),
                "jaccard": report["jaccard"][f"{a}|{b}"],
                **{f"d_{c}": float(np.mean([d[c] for d in damaged]) - base[c])
                   for c in world.CATEGORIES},
                **{f"base_{c}": base[c] for c in world.CATEGORIES},
            })
    return rows, report


def group_order(a: str, b: str):
    return ["shared", f"{a}-only", f"{b}-only", "random"]


def aggregate(rows, pair, radius, mix):
    """Mean and std over seeds for one condition, keyed by lesion group."""
    a, b = pair.split("|")
    out = {}
    for name in group_order(a, b):
        cell = [r for r in rows if r["pair"] == pair and r["group"] == name
                and np.isclose(r["radius"], radius)
                and np.isclose(r["alpha_mix"], mix_key(mix))]
        if not cell:
            continue
        entry = {"n_units": cell[0]["n_units"],
                 "jaccard": float(np.mean([c["jaccard"] for c in cell]))}
        for c in world.CATEGORIES:
            values = np.array([r[f"d_{c}"] for r in cell], dtype=float)
            entry[c] = float(values.mean())
            entry[c + "_std"] = float(values.std())
        out[name] = entry
    return out


def verdict(agg, a: str, b: str, tol: float, frac: float) -> str:
    """Read the multiplexing signature off the size-matched lesion effects.

    The only real evidence is the shared lesion: private lesions are specific by
    construction (see `lesion_groups`), so they serve as the SCALE against which
    the shared lesion's damage is judged, not as a result.  The criteria are

        both      the shared lesion raises BOTH behaviours' error above `tol`
        material  it does so by at least `frac` of what an equally sized lesion of
                  each behaviour's own private territory costs that behaviour
        located   it does more than an equally sized random lesion of the union

    Partition predicts the shared units belong to at most one behaviour, so `both`
    should fail.
    """
    shared = agg.get("shared")
    if shared is None or shared["n_units"] == 0:
        return ("PARTITION-LIKE: the two fields share no units, so there is nothing "
                "to damage in common. Lower --field-radius to make them overlap.")

    both = shared[a] > tol and shared[b] > tol
    scale_a = agg.get(f"{a}-only", {}).get(a, 0.0)
    scale_b = agg.get(f"{b}-only", {}).get(b, 0.0)
    material = (shared[a] >= frac * scale_a) and (shared[b] >= frac * scale_b)
    rand = agg.get("random")
    located = rand is not None and shared[a] > rand[a] and shared[b] > rand[b]

    detail = (f"[shared {shared[a]:+.4f}/{shared[b]:+.4f} vs private scale "
              f"{scale_a:.4f}/{scale_b:.4f} vs random {rand[a]:+.4f}/{rand[b]:+.4f}]"
              if rand is not None else "")
    if both and material and located:
        return ("MULTIPLEXING: the shared units carry BOTH behaviours, at a level "
                f"comparable to each behaviour's own private units, and beyond an "
                f"equally sized random lesion. A partition cannot do this. {detail}")
    if both and (material or located):
        return ("MULTIPLEXING (weak): the shared units damage both, but only one of "
                f"the two quantitative checks passes. {detail}")
    if both:
        return ("AMBIGUOUS: the shared lesion touches both behaviours, but only "
                f"marginally relative to the private scale and the random control. "
                f"{detail}")
    return ("PARTITION-LIKE: damaging the shared units does not degrade both "
            f"behaviours. {detail}")


def plot_bars(rows, radius, mix, args, save_path=None):
    """One panel per pair: lesion effect on each behaviour, by unit group."""
    viz.use_headless(save_path)
    import matplotlib.pyplot as plt
    plt.rcParams.update(viz.FONT)

    pairs = [f"{a}|{b}" for a, b in itertools.combinations(world.CATEGORIES, 2)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 5.0),
                             sharey=True)
    for ax, pair in zip(np.atleast_1d(axes), pairs):
        a, b = pair.split("|")
        agg = aggregate(rows, pair, radius, mix)
        names = [n for n in group_order(a, b) if n in agg]
        x = np.arange(len(names))
        for i, behaviour in enumerate((a, b)):
            vals = [agg[n][behaviour] for n in names]
            errs = [agg[n][behaviour + "_std"] for n in names]
            ax.bar(x + (i - 0.5) * 0.36, vals, width=0.34,
                   color=viz.MARKERS[behaviour]["color"], alpha=0.85,
                   label=world.FIELD_EPISODE[behaviour],
                   yerr=errs if args.seeds > 1 else None,
                   capsize=3 if args.seeds > 1 else 0)
        ax.axhline(0.0, color="0.2", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("-only", "\nonly") for n in names])
        ax.set_title(f"{pair}   (Jaccard {agg[names[0]]['jaccard']:.2f})")
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
    np.atleast_1d(axes)[0].set_ylabel("increase in task error\nafter lesion")
    fig.tight_layout()
    return viz.save_or_show(fig, save_path)


def plot_overlap_sweep(rows, radii, mix, args, save_path=None):
    """Shared-lesion damage as a function of how much the fields actually overlap."""
    viz.use_headless(save_path)
    import matplotlib.pyplot as plt
    plt.rcParams.update(viz.FONT)

    pairs = [f"{a}|{b}" for a, b in itertools.combinations(world.CATEGORIES, 2)]
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for pair in pairs:
        a, b = pair.split("|")
        js, both = [], []
        for radius in radii:
            agg = aggregate(rows, pair, radius, mix)
            if "shared" not in agg:
                continue
            js.append(agg["shared"]["jaccard"])
            both.append(min(agg["shared"][a], agg["shared"][b]))
        order = np.argsort(js)
        ax.plot(np.array(js)[order], np.array(both)[order], "-o", lw=1.6, label=pair)
    ax.axhline(0.0, color="0.2", lw=1.0)
    ax.set_xlabel("Jaccard overlap of the recruited sets  (measured through nu)")
    ax.set_ylabel("damage to the LESS affected behaviour\nof the pair")
    ax.set_title("More sharing means the shared units carry more of both")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return viz.save_or_show(fig, save_path)


def write_csv(path: Path, rows, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w") as f:
        f.write(f"# neuromod_lesion  model={args.model} "
                f"hidden_layers={args.hidden_layers} epochs={args.epochs} "
                f"seeds={args.seeds} radii={args.radii} "
                f"(kill triple; recruited sets from nu)\n")
        f.write("# d_<behaviour> = increase in that behaviour's task error\n")
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(f"{row[k]}" for k in keys) + "\n")
    print(f"saved {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("\n\n")[0])
    p.add_argument("--model", choices=("analytic", "sample"), default="analytic",
                   help="analytic is fine and fast here: L2 is about WHERE the "
                        "behaviours are stored, not about noise being a resource "
                        "[analytic]")
    p.add_argument("--hidden-layers", type=int, default=1,
                   help="1 keeps sigma-only recruitment exact, which is why the RL "
                        "version of this experiment used a single layer [1]")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grid-side", type=int, default=21)
    p.add_argument("--seeds", type=int, default=1,
                   help="Networks per condition; >1 gives error bars [1]")
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--base-std", type=float, default=0.8)
    p.add_argument("--sigma", type=float, default=0.22)
    p.add_argument("--theta", type=float, default=0.15)
    p.add_argument("--field-radius", type=float, default=0.28,
                   help="Ring radius of the bump centres; smaller = more overlap [0.28]")
    p.add_argument("--radii", default=None,
                   help="Comma-separated radii to sweep overlap; retrains per radius")
    p.add_argument("--random-draws", type=int, default=5,
                   help="Random control lesions to average over [5]")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Error increase counted as real damage [1e-3]")
    p.add_argument("--frac", type=float, default=0.3,
                   help="Shared damage must reach this fraction of the private-unit "
                        "scale to count as material [0.3]")
    p.add_argument("--alpha-mix", type=float, default=None)
    p.add_argument("--alpha-mixes", default=None,
                   help="Comma-separated off-drive mixes to sweep ('none' for the "
                        "default near-one-hot targets); retrains per value. This is "
                        "the axis that makes the behaviours compete for the same "
                        "inputs, so it decides whether sharing is needed at all")
    p.add_argument("--target-gamma", type=float, default=2.0)
    p.add_argument("--save", action="store_true")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tag", default="lesion")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    radii = ([float(r) for r in str(args.radii).split(",") if r.strip()]
             if args.radii else [args.field_radius])
    mixes = ([None if m.strip().lower() in ("none", "onehot") else float(m)
              for m in str(args.alpha_mixes).split(",") if m.strip()]
             if args.alpha_mixes else [args.alpha_mix])
    rng = np.random.default_rng(args.seed0)

    n_nets = len(radii) * len(mixes) * args.seeds
    print(f"model={args.model}  hidden_layers={args.hidden_layers}  "
          f"epochs={args.epochs}  seeds={args.seeds}\n"
          f"radii={radii}  alpha_mixes="
          + ", ".join("one-hot" if m is None else f"{m:g}" for m in mixes)
          + f"  ({n_nets} networks to train)")

    rows, conditions = [], []
    for mix in mixes:
        obs, targets = build_data(args, mix)
        for radius in radii:
            conditions.append((radius, mix))
            for k in range(args.seeds):
                seed = args.seed0 + k
                args.alpha_mix = mix          # recorded into the rows by run_cell
                cell, report = run_cell(seed, radius, obs, targets, args, rng)
                rows.extend(cell)
                label = "one-hot" if mix is None else f"{mix:g}"
                print(f"  radius={radius:.2f} alpha_mix={label:>7s} seed={seed}  "
                      + "  ".join(f"J({p})={v:.2f}"
                                  for p, v in report["jaccard"].items()), flush=True)

    for radius, mix in conditions:
        label = "one-hot" if mix is None else f"{mix:g}"
        print(f"\n=== radius {radius:.2f}, alpha_mix {label} "
              f"(hidden layer {LESION_LAYER + 1}, kill triple, size-matched) ===")
        for a, b in itertools.combinations(world.CATEGORIES, 2):
            pair = f"{a}|{b}"
            agg = aggregate(rows, pair, radius, mix)
            if "shared" not in agg:
                continue
            print(f"  {pair}   Jaccard {agg['shared']['jaccard']:.3f}   "
                  f"lesion size {agg['shared']['n_units']}")
            print(f"    {'group':14s}  " + "  ".join(f"d_{c:<8s}" for c in (a, b)))
            for name in group_order(a, b):
                if name not in agg:
                    continue
                e = agg[name]
                print(f"    {name:14s}  " + "  ".join(f"{e[c]:+9.4f}" for c in (a, b)))
            print(f"    -> {verdict(agg, a, b, args.tol, args.frac)}")

    radius_last, mix_last = conditions[-1]
    if args.save:
        write_csv(viz.resolve_out(f"{args.tag}.csv", args.out_dir), rows, args)
        plot_bars(rows, radius_last, mix_last, args,
                  save_path=viz.resolve_out(f"{args.tag}_bars.png", args.out_dir))
        if len(radii) > 1:
            plot_overlap_sweep(rows, radii, mix_last, args,
                               save_path=viz.resolve_out(f"{args.tag}_overlap.png",
                                                         args.out_dir))
    else:
        plot_bars(rows, radius_last, mix_last, args)
        if len(radii) > 1:
            plot_overlap_sweep(rows, radii, mix_last, args)


if __name__ == "__main__":
    main()
