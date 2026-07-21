"""rl_cartpole_train -- Step B / M2-(a): learning curves per agent (idea_rl.md §20.6, §20.7).

Runs the online forward-fluctuation actor-critic on CartPole for each credit condition
(cov_jac / node_pert / true_transpose / backprop) over several seeds and plots return vs
env steps (mean +/- std).  This is the G2 learning half; the variance half is
rl_variance.py.

    .venv/bin/python tmp/rl_cartpole_train.py --agents cov_jac node_pert backprop \
        --seeds 0 1 2 --steps 60000 --H 64 --T 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from rl.train import train, Hypers

OUT = Path(__file__).resolve().parent / "out"


def binned_curve(ep_end_steps, ep_returns, total_steps, n_bins=60):
    """Average episodic return within each env-step bin (last-value hold when empty)."""
    edges = np.linspace(0, total_steps, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    curve = np.full(n_bins, np.nan)
    idx = np.digitize(ep_end_steps, edges) - 1
    for b in range(n_bins):
        vals = [ep_returns[i] for i in range(len(ep_returns)) if idx[i] == b]
        if vals:
            curve[b] = np.mean(vals)
    # forward-fill NaNs
    last = np.nan
    for b in range(n_bins):
        if np.isnan(curve[b]):
            curve[b] = last
        else:
            last = curve[b]
    return centers, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+",
                    default=["cov_jac", "node_pert", "backprop"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--lr_actor", type=float, default=0.025)
    ap.add_argument("--lr_critic", type=float, default=0.05)
    ap.add_argument("--opt", default="adam")
    ap.add_argument("--tag", default="m2a")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    results = {}
    for kind in args.agents:
        curves = []
        for seed in args.seeds:
            hp = Hypers(hidden=args.H, t=args.T, total_steps=args.steps,
                        lr_actor=args.lr_actor, lr_critic=args.lr_critic, opt=args.opt)
            res = train(kind, seed=seed, hp=hp, verbose=False)
            c, y = binned_curve(res.ep_end_steps, res.ep_returns, args.steps)
            curves.append(y)
            print(f"[{kind} seed{seed}] episodes={len(res.ep_returns)} "
                  f"final~{np.nanmean(res.ep_returns[-20:]):.0f} max={max(res.ep_returns):.0f}")
        results[kind] = (c, np.stack(curves))

    np.savez(OUT / f"rl_{args.tag}_curves.npz",
             **{k: v[1] for k, v in results.items()},
             steps=next(iter(results.values()))[0])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.5))
    for kind, (c, Y) in results.items():
        m, s = np.nanmean(Y, axis=0), np.nanstd(Y, axis=0)
        plt.plot(c, m, label=kind)
        plt.fill_between(c, m - s, m + s, alpha=0.2)
    plt.xlabel("environment steps")
    plt.ylabel("episode return (mean +/- std over seeds)")
    plt.title(f"CartPole -- forward-fluctuation actor-critic (H={args.H}, T={args.T})")
    plt.legend()
    plt.tight_layout()
    fig_path = OUT / f"rl_{args.tag}_curves.png"
    plt.savefig(fig_path, dpi=130)
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
