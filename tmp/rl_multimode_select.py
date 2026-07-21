"""rl_multimode_select -- reward-driven autonomous noise-field selection (§7.2, §9; Task #2).

Trains the selector + behavior body jointly on the two-target reach, where the field
prototypes carry NO pre-assigned meaning.  Shows that reward self-organizes a consistent
context -> field -> behavior mapping: each context is routed to a DIFFERENT field, and each
field's body behavior reaches that context's target.

    .venv/bin/python tmp/rl_multimode_select.py --episodes 5000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from rl.multimode_select import train_select, composed_behavior, _softmax
from rl.envs_multimode import MultiModeReach


def trajectories_for_field(policy, fields, o, n=10, horizon=40, seed=0):
    env = MultiModeReach(horizon=horizon, seed=200 + seed)
    policy.field = fields[o]
    trajs = []
    for e in range(n):
        env.reset(e % 2)
        xs = [env.x]
        obs_t = torch.tensor(env._obs()).unsqueeze(0)
        for _ in range(horizon):
            step = policy.rollout_step(obs_t, greedy=True)
            obs2, _, done = env.step(int(step.action.item()))
            xs.append(env.x)
            obs_t = torch.tensor(obs2).unsqueeze(0)
            if done:
                break
        trajs.append(xs)
    return trajs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--lr_sel", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT = Path(__file__).resolve().parent / "out"
    OUT.mkdir(exist_ok=True)

    policy, fields, env, theta, hist = train_select(
        seed=args.seed, H=args.H, sigma=args.sigma, episodes=args.episodes,
        lr_sel=args.lr_sel, verbose=True)
    print(f"\nfinal return(last200): {np.mean(hist[-200:]):.2f}")
    print("selector preferences theta (rows=context, cols=field):")
    print(np.round(theta, 2))

    print("\nAUTONOMOUS SELECTION -- selector chooses the field per context:")
    chosen = {}
    for ctx in (0, 1):
        o, m, s = composed_behavior(policy, fields, env, theta, ctx)
        chosen[ctx] = o
        ok = "OK" if abs(m - env.targets[ctx]) < 0.25 else "XX"
        print(f"  context {ctx} (target {env.targets[ctx]:+.0f}) -> selects P{o} "
              f"-> endpoint x = {m:+.2f} +/- {s:.2f}  [{ok}]")
    differentiated = chosen[0] != chosen[1]
    print(f"  contexts routed to different fields: {differentiated}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # selector softmax heatmap
    probs = np.stack([_softmax(theta[c]) for c in range(theta.shape[0])])
    im = axes[0].imshow(probs, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(theta.shape[1]))
    axes[0].set_xticklabels([f"P{k}" for k in range(theta.shape[1])])
    axes[0].set_yticks(range(theta.shape[0]))
    axes[0].set_yticklabels([f"ctx {c}" for c in range(theta.shape[0])])
    axes[0].set_title("learned selector  pi(field | context)")
    for c in range(theta.shape[0]):
        for k in range(theta.shape[1]):
            axes[0].text(k, c, f"{probs[c,k]:.2f}", ha="center", va="center",
                         color="w" if probs[c, k] < 0.6 else "k")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    # composed trajectories per context (using the selected field)
    for ctx, ax in zip((0, 1), axes[1:]):
        o = chosen[ctx]
        for xs in trajectories_for_field(policy, fields, o, seed=args.seed):
            ax.plot(xs, color=f"C{ctx}", alpha=0.5)
        ax.axhline(env.targets[ctx], color="k", ls="--", lw=1,
                   label=f"target {env.targets[ctx]:+.0f}")
        ax.axhline(env.targets[1 - ctx], color="gray", ls=":", lw=1)
        ax.set_ylim(-1.6, 1.6)
        ax.set_title(f"context {ctx} -> selector picks P{o}")
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
    axes[1].set_ylabel("position x")
    fig.suptitle("Reward self-organizes context -> noise-field -> behavior (autonomous option selection)")
    fig.tight_layout()
    fig.savefig(OUT / "rl_multimode_select.png", dpi=130)
    print(f"saved {OUT / 'rl_multimode_select.png'}")


if __name__ == "__main__":
    main()
