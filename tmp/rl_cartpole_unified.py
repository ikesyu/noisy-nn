"""rl_cartpole_unified -- Task #1: fully forward-native single-NNN actor-critic.

Trains the unified agent (one NNN, one noise; policy + value + exploration + credit all
from the forward fluctuation, value credit through the same weight mirror as the actor)
and compares its learning curve against the Step-B version whose critic is an EXTERNAL
linear TD head on detached features.  If the unified curve tracks the external-critic
curve, the critic has been folded into the forward path with no loss (idea_rl.md §20.15-2).

    .venv/bin/python tmp/rl_cartpole_unified.py --seeds 0 1 --steps 40000 --coef 0.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from rl.unified import train_unified
from rl.train import train, Hypers

OUT = Path(__file__).resolve().parent / "out"


def binned(ep_end_steps, ep_returns, total, n=60):
    edges = np.linspace(0, total, n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(ep_end_steps, edges) - 1
    curve = np.full(n, np.nan)
    for b in range(n):
        vals = [ep_returns[i] for i in range(len(ep_returns)) if idx[i] == b]
        if vals:
            curve[b] = np.mean(vals)
    last = np.nan
    for b in range(n):
        if np.isnan(curve[b]):
            curve[b] = last
        else:
            last = curve[b]
    return centers, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--coef", type=float, default=0.1, help="value_body_coef for unified")
    ap.add_argument("--lr_actor", type=float, default=0.02)
    ap.add_argument("--lr_critic", type=float, default=0.02)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    results = {"unified": [], "external_critic": []}
    for seed in args.seeds:
        hp = Hypers(total_steps=args.steps, lr_actor=args.lr_actor, lr_critic=args.lr_critic)
        ru = train_unified(seed=seed, hp=hp, value_body_coef=args.coef, verbose=False)
        results["unified"].append(binned(ru.ep_end_steps, ru.ep_returns, args.steps))
        print(f"[unified seed{seed}] eps={len(ru.ep_returns)} "
              f"final~{np.nanmean(ru.ep_returns[-20:]):.0f} max={max(ru.ep_returns):.0f}")

        re = train("cov_jac", seed=seed, hp=hp, verbose=False)
        results["external_critic"].append(binned(re.ep_end_steps, re.ep_returns, args.steps))
        print(f"[external seed{seed}] eps={len(re.ep_returns)} "
              f"final~{np.nanmean(re.ep_returns[-20:]):.0f} max={max(re.ep_returns):.0f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.5))
    for name, runs in results.items():
        c = runs[0][0]
        Y = np.stack([y for _, y in runs])
        m, s = np.nanmean(Y, 0), np.nanstd(Y, 0)
        plt.plot(c, m, label=name)
        plt.fill_between(c, m - s, m + s, alpha=0.2)
    plt.xlabel("environment steps")
    plt.ylabel("episode return (mean +/- std over seeds)")
    plt.title(f"Unified forward-native vs external critic (coef={args.coef})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "rl_unified_curves.png", dpi=130)
    print(f"saved {OUT / 'rl_unified_curves.png'}")


if __name__ == "__main__":
    main()
