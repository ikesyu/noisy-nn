"""rl_cartpole_demo -- failure->success animation of the forward-fluctuation agent.

Trains the cov_jac agent (credit generated entirely inside the NNN forward fluctuation
path, no transposed-weight backward, no external RL algorithm), snapshots checkpoints
along the way, evaluates each greedily, selects an increasing progression, and renders
one greedy episode per selected checkpoint into a single captioned GIF:

    early  checkpoint -> pole falls almost immediately
    middle checkpoint -> balances briefly
    late   checkpoint -> balances for a long time (task success)

    .venv/bin/python tmp/rl_cartpole_demo.py                 # train + render
    .venv/bin/python tmp/rl_cartpole_demo.py --cache          # reuse saved checkpoints

Output: tmp/out/rl_cartpole_demo.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from rl.train import train, Hypers
from rl.unified import train_unified
from rl.policy import CartPolePolicy

OUT = Path(__file__).resolve().parent / "out"

# The "most natural for NNN" method (default): the fully unified forward-native
# actor-critic -- one NNN, one noise, where action sampling, exploration, local
# sensitivity, inter-layer credit for BOTH the policy and value heads, and eligibility
# all come from the single forward fluctuation.  No backprop, no external critic, no
# external RL algorithm (idea_rl.md §20.17, Task #1).
TITLES = {
    "unified": "single NNN, single noise: policy + value + exploration + credit\n"
               "all from the forward fluctuation  (no backprop, no external critic/RL)",
    "cov_jac": "NNN forward-fluctuation credit  (external TD critic; no backprop for policy)",
}


def build(state):
    policy = CartPolePolicy(hidden=int(state["hidden"]), t=int(state["t"]))
    policy.load_state_dict(state["net"])
    policy.eval()
    return policy, state["norm_mean"], state["norm_std"]


def act(policy, obs_raw, mean, std, greedy=True):
    x = torch.clamp((torch.tensor(obs_raw, dtype=torch.float32) - mean) / std, -5, 5)
    step = policy.rollout_step(x.unsqueeze(0), greedy=greedy)
    return int(step.action.item())


def run_episode(state, seed=0, max_len=500, render=False):
    import gymnasium as gym
    policy, mean, std = build(state)
    env = gym.make("CartPole-v1", render_mode="rgb_array" if render else None)
    obs, _ = env.reset(seed=seed)
    frames, ret = [], 0.0
    for _ in range(max_len):
        if render:
            frames.append(env.render())
        a = act(policy, obs, mean, std)
        obs, r, term, trunc, _ = env.step(a)
        ret += r
        if term or trunc:
            break
    env.close()
    return frames, ret


def eval_checkpoint(state, n_eval=4, max_len=500):
    return float(np.mean([run_episode(state, seed=1000 + i, max_len=max_len)[1]
                          for i in range(n_eval)]))


def select_ladder(rendered, targets):
    """rendered: list of (step, ret, frames) sorted by step.  Build a gradual
    failure->success ladder: always start with the first checkpoint (the failure
    baseline), then for each rising target take the EARLIEST checkpoint that first
    reaches it (a faithful 'first time it got this good' training story)."""
    chosen = [0]                                   # first checkpoint = failure baseline
    last_step = rendered[0][0]
    last_ret = rendered[0][1]
    for tgt in targets:
        for i, (step, ret, _) in enumerate(rendered):
            if step > last_step and ret >= tgt and ret > last_ret + 5:
                chosen.append(i)
                last_step, last_ret = step, ret
                break
    # dedupe preserving order
    out, seen = [], set()
    for i in chosen:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["unified", "cov_jac"], default="unified",
                    help="unified = most natural for NNN (default); cov_jac = external critic")
    ap.add_argument("--cache", action="store_true", help="reuse saved checkpoints")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt_every", type=int, default=2500)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    ckpt_path = OUT / f"rl_demo_ckpts_{args.method}.pt"

    if args.cache and ckpt_path.exists():
        checkpoints = torch.load(ckpt_path, weights_only=False)
        print(f"loaded {len(checkpoints)} checkpoints from {ckpt_path}")
    else:
        print(f"training '{args.method}' with checkpoints ...")
        if args.method == "unified":
            hp = Hypers(total_steps=args.steps, lr_actor=0.02, lr_critic=0.02)
            res = train_unified(seed=args.seed, hp=hp,
                                checkpoint_every=args.ckpt_every, verbose=True)
        else:
            hp = Hypers(total_steps=args.steps)
            res = train("cov_jac", seed=args.seed, hp=hp,
                        checkpoint_every=args.ckpt_every, verbose=True)
        checkpoints = res.checkpoints
        torch.save(checkpoints, ckpt_path)
        print(f"saved {len(checkpoints)} checkpoints -> {ckpt_path}")

    print("rendering every checkpoint (greedy, demo seed) ...")
    rendered = []
    for step, state in checkpoints:
        frames, ret = run_episode(state, seed=args.seed, max_len=500, render=True)
        rendered.append((step, ret, frames))
        print(f"  step {step:6d}  rendered greedy return {ret:6.1f}")

    chosen = select_ladder(rendered, targets=[40, 120, 300, 480])
    best = int(np.argmax([r for _, r, _ in rendered]))       # always end on the best run
    if best not in chosen and rendered[best][1] > rendered[chosen[-1]][1] + 5:
        chosen.append(best)
    print("selected ladder (step, return):",
          [(rendered[i][0], round(rendered[i][1], 1)) for i in chosen])

    segments = []
    for i in chosen:
        step, ret, frames = rendered[i]
        label = f"step {step:,}      greedy return = {int(ret)} / 500"
        segments.append((frames, label))

    render_gif(segments, OUT / "rl_cartpole_demo.gif", TITLES[args.method], fps=args.fps)


def render_gif(segments, path, title, fps=30, max_seg_frames=150, freeze=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    flat = []
    for frames, label in segments:
        fr = frames[:max_seg_frames]
        for k, f in enumerate(fr):
            flat.append((f, label, k + 1, len(fr)))
        for _ in range(freeze):                    # brief pause on the last frame
            flat.append((fr[-1], label, len(fr), len(fr)))

    h, w, _ = flat[0][0].shape
    fig, ax = plt.subplots(figsize=(w / 100 + 1.4, h / 100 + 1.3))
    ax.axis("off")
    ax.set_title(title, fontsize=9, family="monospace", pad=24)
    im = ax.imshow(flat[0][0])
    txt = ax.text(0.5, 1.01, "", transform=ax.transAxes, ha="center", va="bottom",
                  fontsize=11, family="monospace")

    def update(idx):
        f, label, k, n = flat[idx]
        im.set_data(f)
        txt.set_text(label)
        return im, txt

    anim = FuncAnimation(fig, update, frames=len(flat), interval=1000 / fps, blit=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {path}  ({len(flat)} frames)")


if __name__ == "__main__":
    main()
