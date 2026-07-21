"""rl_cartpole_swingup_demo -- learn and animate CartPole SWING-UP with an NNN RL.

The pole starts hanging DOWN; the agent must pump energy to swing it up and then balance
it near upright.  Solved here with the forward-fluctuation actor-critic (cov_jac credit,
no transposed-weight backprop) on a custom swing-up env (tmp/rl/envs_swingup.py).  The demo
renders one greedy episode per selected training checkpoint, showing the acquisition:
hanging -> partial swings -> full swing-up and balance.

Swing-up naturally has two regimes -- pump (far from upright) and catch/balance (near
upright) -- which is exactly the kind of behavioral-mode structure a noise-field option
(idea_rl.md §21) could carry; here we first solve it with a single policy.

    .venv/bin/python tmp/rl_cartpole_swingup_demo.py            # train (or reuse) + render
    .venv/bin/python tmp/rl_cartpole_swingup_demo.py --cache    # reuse tmp/out/swingup_run.pt
Output: tmp/out/rl_cartpole_swingup_demo.gif
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from rl.envs_swingup import CartPoleSwingUp
from rl.a2c_swingup import build_policy, _set_field

OUT = Path(__file__).resolve().parent / "out"
RUN = OUT / "swingup_a2c.pt"   # continuous NNN cov_jac actor + external GAE critic (方向1)


def run_episode(state, horizon=500, seed=0, collect=False):
    """Greedy continuous-force episode from the bottom; returns mean cos and (x, theta) traj.
    If the checkpoint carries a noise-field option, its field is set per step from cos(theta)
    (pump<->balance, 方向3a)."""
    policy, mean, std = build_policy(state)
    env = CartPoleSwingUp(horizon=horizon, random_start=False, seed=seed,
                          force_mag=state["force_mag"], x_threshold=4.0, continuous=True)
    obs, _ = env.reset(seed=seed)
    coss, traj = [], []
    for _ in range(horizon):
        if collect:
            traj.append((env.state[0], env.state[2]))       # (x, theta)
        if policy._opt_fields is not None:
            _set_field(policy, policy._opt_fields, float(obs[2]),
                       policy._gate_k, policy._gate_c)
        on = torch.clamp((torch.tensor(obs, dtype=torch.float32) - mean) / std, -5, 5)
        step = policy.rollout_step(on.unsqueeze(0), greedy=True)
        obs, r, term, trunc, _ = env.step(float(step.action.item()))
        coss.append(env.cos_theta())
        if term or trunc:
            break
    return float(np.mean(coss)), traj


class Renderer:
    """Persistent matplotlib cart-pole renderer -> RGB frames."""

    def __init__(self, x_thr=3.0, pole_len=1.0):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        self.plt = plt
        self.fig, self.ax = plt.subplots(figsize=(5.2, 3.7))
        self.fig.subplots_adjust(top=0.80, bottom=0.02, left=0.02, right=0.98)
        self.ax.set_xlim(-x_thr - 0.6, x_thr + 0.6)
        self.ax.set_ylim(-1.35, 1.55)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.ax.plot([-x_thr - 0.6, x_thr + 0.6], [0, 0], color="0.6", lw=1)   # track
        self.ax.plot([-x_thr, x_thr], [0, 0], "|", color="0.4", markersize=8)  # bounds
        self.ax.plot(0, pole_len, marker="*", color="gold", markersize=13, zorder=1)  # target
        self.cart = Rectangle((-0.2, -0.1), 0.4, 0.2, color="0.2", zorder=3)
        self.ax.add_patch(self.cart)
        (self.pole,) = self.ax.plot([], [], color="C1", lw=4, zorder=4)
        (self.bob,) = self.ax.plot([], [], "o", color="C3", markersize=8, zorder=5)
        self.title = self.ax.text(0.5, 1.14, "", transform=self.ax.transAxes,
                                  ha="center", va="bottom", fontsize=9, family="monospace")
        self.sub = self.ax.text(0.5, 1.03, "", transform=self.ax.transAxes,
                                ha="center", va="bottom", fontsize=9, family="monospace")
        self.pole_len = pole_len

    def frame(self, x, theta, title, sub):
        tipx, tipy = x + self.pole_len * math.sin(theta), self.pole_len * math.cos(theta)
        self.cart.set_xy((x - 0.2, -0.1))
        self.pole.set_data([x, tipx], [0.05, tipy + 0.05])
        self.bob.set_data([tipx], [tipy + 0.05])
        self.pole.set_color("C2" if math.cos(theta) > 0.9 else "C1")
        self.title.set_text(title)
        self.sub.set_text(sub)
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        return buf[:, :, :3].copy()


def select_ladder(evals):
    """evals: list of (step, mean_cos). Ladder of increasing performance ending on best."""
    best = int(np.argmax([c for _, c in evals]))
    chosen = [0]
    last = evals[0][1]
    for tgt in np.linspace(evals[0][1], evals[best][1], 5)[1:]:
        for i in range(chosen[-1] + 1, len(evals)):
            if evals[i][1] >= tgt - 1e-6 and evals[i][1] > last + 0.05 and i not in chosen:
                chosen.append(i)
                last = evals[i][1]
                break
    if best not in chosen:
        chosen.append(best)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=45)
    ap.add_argument("--run", default=str(RUN), help="checkpoint .pt (swingup_a2c or swingup_field)")
    ap.add_argument("--out", default="rl_cartpole_swingup_demo.gif")
    ap.add_argument("--title",
                    default="NNN cov_jac actor (no backprop) + GAE critic  |  CartPole swing-up")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    data = torch.load(args.run, weights_only=False)
    checkpoints = data["checkpoints"]
    print(f"loaded {len(checkpoints)} checkpoints from {args.run}")

    print("evaluating checkpoints (greedy mean cos) ...")
    evals = []
    for step, state in checkpoints:
        mc, _ = run_episode(state, horizon=args.horizon, seed=args.seed)
        evals.append((step, mc))
        print(f"  checkpoint {step:5d}  mean cos {mc:+.3f}")

    chosen = select_ladder(evals)
    print("ladder:", [(checkpoints[i][0], round(evals[i][1], 2)) for i in chosen])

    rend = Renderer(x_thr=4.0)
    flat = []
    for i in chosen:
        step, state = checkpoints[i]
        mc, traj = run_episode(state, horizon=args.horizon, seed=args.seed, collect=True)
        title = args.title
        sub = f"update {step}    mean cos(theta) = {mc:+.2f}   (+1 = upright & balanced)"
        for k, (x, th) in enumerate(traj[::2][:180]):
            flat.append((x, th, title, sub))
        for _ in range(18):
            flat.append((traj[-1][0], traj[-1][1], title, sub))

    print(f"rendering {len(flat)} frames ...")
    frames = [rend.frame(x, th, t, s) for (x, th, t, s) in flat]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, ax = plt.subplots(figsize=(frames[0].shape[1] / 100, frames[0].shape[0] / 100))
    ax.axis("off")
    im = ax.imshow(frames[0])

    def _upd(k):
        im.set_data(frames[k])
        return (im,)

    anim = FuncAnimation(fig, _upd, frames=len(frames),
                         interval=1000 / args.fps, blit=True)
    anim.save(OUT / args.out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"saved {OUT / args.out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
