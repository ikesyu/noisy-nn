"""rl_ppo_nowall_raster -- raster-plot video of BOTH NNN populations (actor + critic)
while the final no-stopper PPO v4 policy performs swing-up + balance.

For every step of one greedy from-bottom episode we record the T-averaged crossing
activity z of every hidden unit in the actor NNN (2 layers x 128) and the critic NNN
(2 layers x 64).  The video shows the cart-pole (left), both rasters filling in as the
episode unfolds (right, units sorted per layer by peak-activity time), and the aligned
behavior traces (cos(theta), normalized action, critic value V).

    .venv/bin/python tmp/rl_ppo_nowall_raster.py [--run tmp/out/swingup_ppo_nowall_s0.pt]
Output: tmp/out/rl_ppo_nowall_raster.gif   (animation)
        tmp/out/ppo_nowall_raster.png      (static full-episode figure)
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
from rl.a2c_swingup import build_policy
from rl.critic import NNNCritic

OUT = Path(__file__).resolve().parent / "out"
X_THR = 4.0


def record_episode(data, horizon=500, seed=0, ck_index=-1):
    state = data["checkpoints"][ck_index][1]
    policy, mean, std = build_policy(state)
    critic = NNNCritic(obs_dim=5, hidden=int(data["critic_hidden"]),
                       t=int(data["critic_t"]))
    critic.net.load_state_dict(data["critic_net"])
    critic.eval()
    env = CartPoleSwingUp(horizon=horizon, random_start=False, seed=seed,
                          force_mag=state["force_mag"], x_threshold=X_THR,
                          continuous=True, wall_mode="end",
                          wall_penalty=data.get("wall_penalty", 3.0),
                          x_barrier=data.get("x_barrier", 0.5))
    obs, _ = env.reset(seed=seed)
    rec = {"az": [[], []], "cz": [[], []], "cos": [], "x": [], "theta": [],
           "a": [], "v": [], "wall_hits": 0}
    for _ in range(horizon):
        on = torch.clamp((torch.tensor(obs, dtype=torch.float32) - mean) / std, -5, 5)
        step = policy.rollout_step(on.unsqueeze(0), greedy=True)
        v, cstep = critic.value_step(on.unsqueeze(0))
        for l in range(2):
            rec["az"][l].append(step.z[l][0].mean(dim=0).numpy())   # [H] mean over T
            rec["cz"][l].append(cstep.z[l][0].mean(dim=0).numpy())
        rec["a"].append(float(step.action.item()) / state["force_mag"])
        rec["v"].append(v)
        obs, r, te, tr, info = env.step(float(step.action.item()))
        rec["wall_hits"] += int(bool(info.get("wall", False)))
        rec["cos"].append(env.cos_theta())
        rec["x"].append(float(env.state[0]))
        rec["theta"].append(float(env.state[2]))
        if te or tr:
            break
    for k in ("az", "cz"):
        rec[k] = [np.stack(v, axis=1) for v in rec[k]]              # [H, steps]
    return rec


def sort_by_peak(z):
    return z[np.argsort(np.argmax(z, axis=1)), :]


def build_rasters(rec):
    """Per-network raster [units, steps], layers stacked, units sorted by peak time."""
    A = np.concatenate([sort_by_peak(z) for z in rec["az"]], axis=0)
    C = np.concatenate([sort_by_peak(z) for z in rec["cz"]], axis=0)
    return A, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(OUT / "swingup_ppo_nowall_s0.pt"))
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default="rl_ppo_nowall_raster.gif")
    ap.add_argument("--ck-index", type=int, default=-1)
    ap.add_argument("--torch-seed", type=int, default=0)
    ap.add_argument("--tries", type=int, default=1,
                    help="record N episodes (the NNN ensemble mean makes even greedy "
                         "episodes stochastic) and keep the one with the best hold")
    args = ap.parse_args()

    data = torch.load(args.run, weights_only=False)
    torch.manual_seed(args.torch_seed)
    best = None
    for k in range(args.tries):
        r = record_episode(data, horizon=args.horizon, seed=args.seed,
                           ck_index=args.ck_index)
        full = len(r["cos"]) == args.horizon
        tail = float((np.array(r["cos"])[-100:] > 0.9).mean()) if full else -1.0
        score = (tail, float(np.mean(r["cos"])))
        print(f"  try {k}: steps {len(r['cos'])}, walls {r['wall_hits']}, "
              f"last100_up {max(tail, 0.0):.3f}")
        if best is None or score > best[0]:
            best = (score, r)
    rec = best[1]
    n = len(rec["cos"])
    A, C = build_rasters(rec)
    tail_up = float((np.array(rec["cos"])[-100:] > 0.9).mean()) if n == args.horizon else 0.0
    print(f"episode: {n} steps, wall contacts {rec['wall_hits']}, "
          f"max|x| {max(abs(v) for v in rec['x']):.2f}, last100_up {tail_up:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.animation import FuncAnimation, PillowWriter
    def int_ticks(lo, hi, max_n=5):
        ticks = np.arange(math.ceil(lo), math.floor(hi) + 1)
        return ticks[::2] if len(ticks) > max_n else ticks

    plt.rcParams.update({
        "font.size":       13,
        "axes.titlesize":  14,
        "axes.labelsize":  13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })

    vA, vC = np.percentile(A, 98), np.percentile(C, 98)

    def setup(fig):
        gs = fig.add_gridspec(3, 2, width_ratios=[0.9, 1.35],
                              height_ratios=[1.35, 1.0, 0.75],
                              hspace=0.5, wspace=0.26,
                              left=0.075, right=0.97, top=0.95, bottom=0.1)
        axc = fig.add_subplot(gs[0, 0])                 # cart-pole
        axt = fig.add_subplot(gs[1:, 0])                # behavior traces
        axa = fig.add_subplot(gs[0:2, 1])               # actor raster
        axk = fig.add_subplot(gs[2, 1], sharex=axa)     # critic raster
        return axc, axt, axa, axk

    fig = plt.figure(figsize=(12.0, 7.0), dpi=85)
    axc, axt, axa, axk = setup(fig)

    # cart-pole
    axc.set_xlim(-X_THR - 0.8, X_THR + 0.8)
    axc.set_ylim(-1.45, 1.7)
    axc.set_aspect("equal")
    axc.axis("off")
    axc.plot([-X_THR - 0.6, X_THR + 0.6], [0, 0], color="0.6", lw=1)
    for sx in (-X_THR, X_THR):
        axc.plot([sx, sx], [-0.28, 0.28], color="#c23b3b", lw=3)
    axc.plot(0, 1.0, marker="*", color="goldenrod", markersize=11, zorder=1)
    cart = Rectangle((-0.2, -0.1), 0.4, 0.2, color="0.2", zorder=3)
    axc.add_patch(cart)
    (pole,) = axc.plot([], [], color="C1", lw=4, zorder=4)
    (bob,) = axc.plot([], [], "o", color="C3", markersize=7, zorder=5)
    head = axc.text(0.5, 1.02, "", transform=axc.transAxes, ha="center",
                    fontsize=14, family="monospace")

    # behavior traces (all dimensionless: cos(theta), a in [-1,1], V standardized)
    axt.set_xlim(0, n)
    axt.set_xticks([0, n // 2, n])
    axt.set_xlabel("episode step")
    axt.grid(alpha=0.25)
    axt.axhline(0.0, color="0.85", lw=0.8)
    (l_cos,) = axt.plot([], [], color="C2", lw=1.3, label="cos(theta)")
    (l_act,) = axt.plot([], [], color="C0", lw=0.9, label="action a")
    (l_val,) = axt.plot([], [], color="C4", lw=1.1, label="V (std. units)")
    lo = min(-1.1, np.min(rec["v"]) - 0.2)
    hi = max(1.1, np.max(rec["v"]) + 0.2)
    axt.set_ylim(lo, hi)
    axt.set_yticks(int_ticks(lo, hi))
    axt.legend(loc="lower right", ncol=1, framealpha=0.85)
    cur_t = axt.axvline(0, color="#c23b3b", lw=1.0)

    def raster(ax, Z, vmax, title, n_top):
        im = ax.imshow(np.full_like(Z, np.nan), aspect="auto", origin="lower",
                       cmap="Greys", vmin=0.0, vmax=vmax,
                       extent=(0, n, 0, Z.shape[0]), interpolation="nearest")
        ax.axhline(n_top, color="C0", lw=0.8, ls=":")
        ax.set_ylabel("unit")
        ax.set_title(title)
        ax.set_xticks([0, n // 2, n])
        ax.set_yticks([0, n_top, Z.shape[0]])
        cur = ax.axvline(0, color="#c23b3b", lw=1.0)
        return im, cur

    imA, curA = raster(axa, A, vA, "Actor NNN (128 + 128 units)", 128)
    imK, curK = raster(axk, C, vC, "Critic NNN (64 + 64 units)", 64)
    axk.set_xlabel("episode step")
    plt.setp(axa.get_xticklabels(), visible=False)

    ks = list(range(0, n, args.stride))
    if ks[-1] != n - 1:
        ks.append(n - 1)
    ks.extend([n - 1] * int(args.fps * 1.0))

    Amask = np.full_like(A, np.nan)
    Cmask = np.full_like(C, np.nan)

    def _upd(fi):
        k = ks[fi]
        x, th, c = rec["x"][k], rec["theta"][k], rec["cos"][k]
        tipx, tipy = x + math.sin(th), math.cos(th)
        cart.set_xy((x - 0.2, -0.1))
        pole.set_data([x, tipx], [0.05, tipy + 0.05])
        bob.set_data([tipx], [tipy + 0.05])
        pole.set_color("C2" if c > 0.9 else "C1")
        head.set_text(f"step {k + 1:3d}/{n}  cos {c:+.2f}  "
                      f"walls {rec['wall_hits']}")
        for line, key in ((l_cos, "cos"), (l_act, "a"), (l_val, "v")):
            line.set_data(range(k + 1), rec[key][:k + 1])
        Amask[:, :k + 1] = A[:, :k + 1]
        Cmask[:, :k + 1] = C[:, :k + 1]
        imA.set_data(Amask)
        imK.set_data(Cmask)
        for cur in (cur_t, curA, curK):
            cur.set_xdata([k, k])
        return ()

    anim = FuncAnimation(fig, _upd, frames=len(ks), interval=1000 / args.fps)
    out = OUT / args.out
    anim.save(out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"saved {out}  ({len(ks)} frames)")

    # static full-episode figure
    fig = plt.figure(figsize=(12.0, 7.0), dpi=120)
    axc, axt, axa, axk = setup(fig)
    axc.axis("off")
    axc.plot(rec["x"], np.cos(rec["theta"]), lw=0.8, color="0.4")
    axc.plot(rec["x"][0], math.cos(rec["theta"][0]), "o", color="C0", label="start")
    axc.plot(rec["x"][-1], math.cos(rec["theta"][-1]), "o", color="C2", label="end")
    for sx in (-X_THR, X_THR):
        axc.axvline(sx, color="#c23b3b", lw=2)
    axc.set_title("tip trajectory (x vs cos(theta)); red = stoppers")
    axc.legend(loc="lower right")
    axt.set_xlim(0, n)
    axt.set_xticks([0, n // 2, n])
    axt.set_yticks(int_ticks(lo, hi))
    axt.grid(alpha=0.25)
    axt.set_xlabel("episode step")
    axt.plot(rec["cos"], color="C2", lw=1.2, label="cos(theta)")
    axt.plot(rec["a"], color="C0", lw=0.8, label="action a")
    axt.plot(rec["v"], color="C4", lw=1.0, label="V (std. units)")
    axt.legend(loc="lower right")
    for ax, Z, vmax, title, ntop in ((axa, A, vA, "Actor NNN (128 + 128 units)", 128),
                                     (axk, C, vC, "Critic NNN (64 + 64 units)", 64)):
        ax.imshow(Z, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=vmax,
                  extent=(0, n, 0, Z.shape[0]), interpolation="nearest")
        ax.axhline(ntop, color="C0", lw=0.8, ls=":")
        ax.set_ylabel("unit")
        ax.set_title(title)
        ax.set_xticks([0, n // 2, n])
        ax.set_yticks([0, ntop, Z.shape[0]])
    axk.set_xlabel("episode step")
    png = OUT / "ppo_nowall_raster.png"
    fig.savefig(png)
    plt.close(fig)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
