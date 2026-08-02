"""rl_swingup_visualize_demo (旧 rl_ppo_nowall_demo) -- animate the ACQUISITION of no-stopper
swing-up from any swing-up training run (PPO or SAC, external or internal noise).

Left: cart-pole episode (greedy, from bottom) for a ladder of training checkpoints;
the stoppers at |x| = 4 are drawn in red -- touching one ends the episode.
Right-top: the training reward curve with a cursor at the current checkpoint.
Right-bottom: the current episode's cos(theta) trace growing in real time.

NOISE-SOURCE AGNOSTIC: playback is greedy (a = mu), and mu is the clean ensemble mean
in every noise mode, so checkpoints from the external-sigma_e drivers and the internal-
temperature drivers replay identically.  Checkpoints that carry a noise FIELD
(_snapshot's fields/gate/rho_mode -- e.g. outgate sigma fields, SAC v6.1 rho_T fields)
ARE field-dependent in mu; the field is re-applied per step here, mirroring
eval_strict / eval_from_bottom.

SAVE-FORMAT REQUIREMENTS (the .pt passed as --run):
  required  "checkpoints": [(step:int, snapshot:dict)]  -- snapshot from
            a2c_swingup._snapshot or a2c_nnncritic._snap (build_policy-compatible;
            fields/gate_k/gate_c/rho_mode optional inside)
  required  "stats": [{"upd"|"epi": int, "ret_step": float, ...}]  -- training curve
            (PPO drivers use "upd", SAC drivers use "epi"; both accepted)
  optional  "evals": [{"upd": int, "mean_cos", "last100_up", "wall_hits", ...}]
            -- used to pick the checkpoint ladder and annotate; without it the ladder
            falls back to first/last checkpoints
  optional  "wall_penalty", "x_barrier" -- terminal-wall env shaping for playback
All current drivers (rl_ppo_external_swingup, rl_ppo_itemp_swingup,
rl_sac_itemp_swingup) write this format.

    .venv/bin/python tmp/rl_swingup_visualize_demo.py [--run tmp/out/swingup_itemp_outconst_s0.pt]
Output: tmp/out/rl_swingup_demo.gif (override with --out)
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
X_THR = 4.0


def run_episode(state, horizon=500, seed=0, wall_penalty=3.0, x_barrier=0.5):
    """Greedy from-bottom episode on the terminal-wall env.  Returns trajectory dicts.
    If the checkpoint carries a noise field, it is re-applied per step (mu depends on
    it); field-less checkpoints run with the network defaults."""
    policy, mean, std = build_policy(state)
    env = CartPoleSwingUp(horizon=horizon, random_start=False, seed=seed,
                          force_mag=state["force_mag"], x_threshold=X_THR,
                          continuous=True, wall_mode="end",
                          wall_penalty=wall_penalty, x_barrier=x_barrier)
    obs, _ = env.reset(seed=seed)
    traj = []
    died = False
    for _ in range(horizon):
        if policy._opt_fields is not None:
            _set_field(policy, policy._opt_fields, float(obs[2]),
                       policy._gate_k, policy._gate_c,
                       policy._rho_mode, policy._sigma0, policy._h0)
        on = torch.clamp((torch.tensor(obs, dtype=torch.float32) - mean) / std, -5, 5)
        step = policy.rollout_step(on.unsqueeze(0), greedy=True)
        obs, r, te, tr, info = env.step(float(step.action.item()))
        traj.append({"x": float(env.state[0]), "theta": float(env.state[2]),
                     "cos": env.cos_theta(), "r": r})
        if te:
            died = True
            break
        if tr:
            break
    return traj, died


def pick_ladder(cks, evals):
    """Pick ~5 checkpoints showing the acquisition: early failure -> partial -> solved."""
    by_upd = {e["upd"]: e for e in evals}
    mcs = [by_upd[u]["mean_cos"] if u in by_upd else -1.0 for u, _ in cks]
    solved = [i for i, (u, _) in enumerate(cks)
              if u in by_upd and by_upd[u]["last100_up"] >= 1.0
              and by_upd[u]["wall_hits"] == 0]
    chosen = [0]
    lo, hi = mcs[0], max(mcs)
    for tgt in np.linspace(lo, hi, 5)[1:-1]:
        cand = [i for i in range(chosen[-1] + 1, len(cks)) if mcs[i] >= tgt]
        if cand and cand[0] not in chosen:
            chosen.append(cand[0])
    if solved:
        first, last = solved[0], len(cks) - 1
        for i in (first, last):
            if i not in chosen:
                chosen.append(i)
    elif (len(cks) - 1) not in chosen:
        chosen.append(len(cks) - 1)
    return sorted(set(chosen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(OUT / "swingup_itemp_outconst_s0.pt"))
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", default="rl_swingup_demo.gif")
    ap.add_argument("--torch-seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.torch_seed)   # greedy NNN episodes are still T-sample stochastic
    data = torch.load(args.run, weights_only=False)
    cks, stats, evals = data["checkpoints"], data["stats"], data.get("evals", [])
    wp, xb = data.get("wall_penalty", 3.0), data.get("x_barrier", 0.5)
    ladder = pick_ladder(cks, evals)
    print("ladder:", [cks[i][0] for i in ladder])

    episodes = []
    for i in ladder:
        upd, state = cks[i]
        traj, died = run_episode(state, horizon=args.horizon, seed=args.seed,
                                 wall_penalty=wp, x_barrier=xb)
        ev = next((e for e in evals if e["upd"] == upd), None)
        episodes.append((upd, traj, died, ev))
        print(f"  upd {upd:4d}  steps {len(traj):3d}  died_at_wall {died}  "
              f"mean cos {np.mean([s['cos'] for s in traj]):+.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.animation import FuncAnimation, PillowWriter

    plt.rcParams.update({
        "font.size":       13,
        "axes.titlesize":  14,
        "axes.labelsize":  13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })

    fig = plt.figure(figsize=(11.5, 4.8), dpi=90)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.65,
                          left=0.055, right=0.97, top=0.92, bottom=0.14)
    axc = fig.add_subplot(gs[:, 0])
    axr = fig.add_subplot(gs[0, 1])
    axe = fig.add_subplot(gs[1, 1])

    # cart-pole panel
    axc.set_xlim(-X_THR - 0.8, X_THR + 0.8)
    axc.set_ylim(-1.45, 1.65)
    axc.set_aspect("equal")
    axc.axis("off")
    axc.plot([-X_THR - 0.6, X_THR + 0.6], [0, 0], color="0.6", lw=1)
    for sx in (-X_THR, X_THR):                          # the stoppers: touching = failure
        axc.plot([sx, sx], [-0.28, 0.28], color="#c23b3b", lw=3)
    axc.text(-X_THR, -0.42, "stopper", color="#c23b3b", ha="center", fontsize=11)
    axc.text(X_THR, -0.42, "stopper", color="#c23b3b", ha="center", fontsize=11)
    axc.plot(0, 1.0, marker="*", color="goldenrod", markersize=12, zorder=1)
    cart = Rectangle((-0.2, -0.1), 0.4, 0.2, color="0.2", zorder=3)
    axc.add_patch(cart)
    (pole,) = axc.plot([], [], color="C1", lw=4, zorder=4)
    (bob,) = axc.plot([], [], "o", color="C3", markersize=7, zorder=5)
    head = axc.text(0.5, 1.06, "", transform=axc.transAxes, ha="center",
                    fontsize=13, family="monospace")

    # training-curve panel (cursor moves with the ladder)
    upds = [s.get("upd", s.get("epi")) for s in stats]   # PPO: "upd" / SAC: "epi"
    rets = [s["ret_step"] for s in stats]
    axr.plot(upds, rets, color="C0", lw=1.2)
    axr.set_xlabel("training update")
    axr.set_ylabel("reward / step")
    axr.xaxis.set_major_locator(plt.MaxNLocator(4))
    axr.yaxis.set_major_locator(plt.MaxNLocator(3))
    axr.grid(alpha=0.25)
    axr.set_title("training reward")
    cursor = axr.axvline(upds[0], color="#c23b3b", lw=1.4)

    # current-episode cos(theta) panel
    axe.set_xlim(0, args.horizon)
    axe.set_ylim(-1.15, 1.15)
    axe.axhspan(0.9, 1.15, color="C2", alpha=0.12, lw=0)
    axe.axhline(0.0, color="0.8", lw=0.8)
    axe.set_xlabel("episode step")
    axe.set_ylabel("cos(theta)")
    axe.set_xticks([0, args.horizon // 2, args.horizon])
    axe.set_yticks([-1, 0, 1])
    axe.grid(alpha=0.25)
    axe.set_title("current episode (shaded: upright)")
    (trace,) = axe.plot([], [], color="C2", lw=1.4)
    death = axe.text(0.98, 0.06, "", transform=axe.transAxes, ha="right",
                     color="#c23b3b", fontsize=11, family="monospace")

    frames = []                                          # (ep_idx, step_idx, hold)
    for e_i, (upd, traj, died, ev) in enumerate(episodes):
        idxs = list(range(0, len(traj), args.stride))
        if idxs[-1] != len(traj) - 1:
            idxs.append(len(traj) - 1)
        frames.extend((e_i, k) for k in idxs)
        frames.extend((e_i, len(traj) - 1) for _ in range(int(args.fps * 0.8)))

    def _upd(fk):
        e_i, k = frames[fk]
        upd, traj, died, ev = episodes[e_i]
        s = traj[k]
        x, th = s["x"], s["theta"]
        tipx, tipy = x + math.sin(th), math.cos(th)
        cart.set_xy((x - 0.2, -0.1))
        pole.set_data([x, tipx], [0.05, tipy + 0.05])
        bob.set_data([tipx], [tipy + 0.05])
        pole.set_color("C2" if s["cos"] > 0.9 else "C1")
        tail = ev["last100_up"] if ev else float("nan")
        head.set_text(f"update {upd:3d}   step {k + 1:3d}   cos {s['cos']:+.2f}\n"
                      f"eval: last100_up {tail:.2f}   wall contacts "
                      f"{ev['wall_hits'] if ev else '?'}")
        cursor.set_xdata([upd, upd])
        trace.set_data(range(k + 1), [t["cos"] for t in traj[:k + 1]])
        death.set_text("hit stopper -- episode over" if died and k == len(traj) - 1
                       else "")
        return ()

    anim = FuncAnimation(fig, _upd, frames=len(frames), interval=1000 / args.fps)
    out = OUT / args.out
    anim.save(out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"saved {out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
