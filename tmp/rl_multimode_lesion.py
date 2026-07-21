"""rl_multimode_lesion -- noise-field L2: multiplexing vs partition (front_comp L2 in RL).

Sub-B/§21.2 used DISJOINT recruitment fields, so the two behaviors sat in separate unit
groups (a partition).  Here the two fields OVERLAP (share a middle block of units).  We
train both behaviors, then LESION unit groups (zero their actor-readout weights) and see
which behaviors degrade:

    lesion SHARED units   -> BOTH behaviors should degrade   (multiplexing signature)
    lesion P0-only units  -> only the P0 behavior degrades
    lesion P1-only units  -> only the P1 behavior degrades

If shared-unit damage hits both behaviors, they are MULTIPLEXED on the shared subnet, not
partitioned into disjoint regions (the direct refutation front_comp wants).

    .venv/bin/python tmp/rl_multimode_lesion.py --episodes 3000 --recruit_frac 0.7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from rl.multimode import train_multimode, behavior_under_field
from rl import field as F


def ablate(policy, units):
    W = policy.fcs[-1].weight
    backup = W.data[:, units].clone()
    W.data[:, units] = 0.0
    return backup


def restore(policy, units, backup):
    policy.fcs[-1].weight.data[:, units] = backup


def errors(policy, fields, env):
    """Task error |endpoint - target| under each fixed field."""
    e0, _ = behavior_under_field(policy, fields, env, 0)
    e1, _ = behavior_under_field(policy, fields, env, 1)
    return abs(e0 - env.targets[0]), abs(e1 - env.targets[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--recruit_frac", type=float, default=0.7)
    ap.add_argument("--n_layers", type=int, default=1,
                    help="1 = clean recruitment gating at the readout (needed for a valid "
                         "lesion test); 2+ leaks upstream fluctuation into off-units")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT = Path(__file__).resolve().parent / "out"
    OUT.mkdir(exist_ok=True)

    fields, idx = F.overlapping_pair(args.H, args.sigma, args.recruit_frac,
                                     n_layers=args.n_layers)
    n_sh, n0, n1 = len(idx["shared"]), len(idx["p0_only"]), len(idx["p1_only"])
    union = n_sh + n0 + n1
    print(f"overlapping fields: shared={n_sh} p0_only={n0} p1_only={n1}  "
          f"Jaccard(active sets) = {n_sh/union:.2f}")

    policy, _, env, hist = train_multimode(
        seed=args.seed, H=args.H, sigma=args.sigma, episodes=args.episodes,
        fields=fields, opt_kind="sgd", lr_actor=0.05, n_layers=args.n_layers, verbose=True)
    b0, b1 = errors(policy, fields, env)
    print(f"\nbaseline task error (no lesion): P0={b0:.2f}  P1={b1:.2f}")

    rng = np.random.default_rng(args.seed)
    groups = {
        "shared": idx["shared"],
        "P0-only": idx["p0_only"],
        "P1-only": idx["p1_only"],
        "random": rng.choice(args.H, size=n_sh, replace=False).tolist(),
    }
    print("\nLESION TEST -- task error (increase over baseline) under each field:")
    print(f"{'lesion group':>12} | {'err P0':>8} {'err P1':>8} | {'dP0':>7} {'dP1':>7}")
    print("-" * 52)
    rows = {}
    for name, units in groups.items():
        backup = ablate(policy, units)
        e0, e1 = errors(policy, fields, env)
        restore(policy, units, backup)
        rows[name] = (e0, e1, e0 - b0, e1 - b1)
        print(f"{name:>12} | {e0:>8.2f} {e1:>8.2f} | {e0-b0:>+7.2f} {e1-b1:>+7.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(groups)
    x = np.arange(len(names))
    dP0 = [rows[n][2] for n in names]
    dP1 = [rows[n][3] for n in names]
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - 0.2, dP0, 0.4, label="behavior 0 degradation", color="C0")
    plt.bar(x + 0.2, dP1, 0.4, label="behavior 1 degradation", color="C1")
    plt.xticks(x, names)
    plt.ylabel("increase in task error when lesioned")
    plt.title(f"Lesion test (Jaccard={n_sh/union:.2f}): shared units carry BOTH behaviors\n"
              "(multiplexing, not partition) if the 'shared' bars are both high")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "rl_multimode_lesion.png", dpi=130)
    print(f"saved {OUT / 'rl_multimode_lesion.png'}")


if __name__ == "__main__":
    main()
