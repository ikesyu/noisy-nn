"""rl_multimode -- noise-field Sub-B: two behaviors multiplexed on ONE weight set,
addressed by the noise field (§7.2 / §14.2; RL version of front_comp's L1 result).

Trains the shared-weight policy on the hidden-regime two-target reach, then shows the
decisive test: FIX the noise field and the agent commits to that field's target; SWITCH
the field and, with the SAME weights, it seeks the other target.  The regime is never
observed -- only the field selects the behavior.

    .venv/bin/python tmp/rl_multimode.py --episodes 4000 --H 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from rl.multimode import train_multimode, behavior_under_field, _p  # noqa: F401
from rl.envs_multimode import MultiModeReach


def trajectories(policy, fields, field_idx, n=12, horizon=40, seed=0):
    env = MultiModeReach(horizon=horizon, seed=100 + seed)
    policy.field = fields[field_idx]
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
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT = Path(__file__).resolve().parent / "out"
    OUT.mkdir(exist_ok=True)

    policy, fields, env, hist = train_multimode(
        seed=args.seed, H=args.H, sigma=args.sigma, episodes=args.episodes, verbose=True)
    print(f"final return(last200): {np.mean(hist[-200:]):.2f}  (0 = target reached)")

    print("\nDECISIVE TEST -- fix the field, regime hidden:")
    for fi in (0, 1):
        m, s = behavior_under_field(policy, fields, env, fi)
        print(f"  field P_{fi} (target {env.targets[fi]:+.0f}) -> endpoint x = {m:+.2f} +/- {s:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for fi, ax in zip((0, 1), axes):
        for xs in trajectories(policy, fields, fi, seed=args.seed):
            ax.plot(xs, color="C0" if fi == 0 else "C1", alpha=0.5)
        ax.axhline(env.targets[fi], color="k", ls="--", lw=1, label=f"target {env.targets[fi]:+.0f}")
        ax.axhline(env.targets[1 - fi], color="gray", ls=":", lw=1)
        ax.set_title(f"field P_{fi} fixed  (regime hidden)")
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("position x")
    fig.suptitle("Same weights, switch the noise field -> switch the behavior (§14.2)")
    fig.tight_layout()
    fig.savefig(OUT / "rl_multimode.png", dpi=130)
    print(f"saved {OUT / 'rl_multimode.png'}")


if __name__ == "__main__":
    main()
