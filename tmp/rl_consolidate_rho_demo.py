"""rl_consolidate_rho_demo -- animate the rho-OLAP skill-reuse curriculum (§23.7).

Renders how the composed policy of the rho/h-gated overlap arm acquires full
swing-up & balance: the frozen BALANCE skill lives on subnetwork A (phase 1),
PUMP is added on subnetwork B (phase 2, olap arm).  Each animation rung is one
greedy episode from the hanging start:

    rung 0 : phase-2 update 0  -- pump not yet learned (A frozen, B random)
    rung 1+: phase-2 checkpoints up to the best one -- full swing-up & balance

The field-mode banner shows the per-step rho gate g = sigma(6 cos(theta)):
PUMP mode (g<0.5) mobilises B plus the frozen A as a read-only vocabulary
(rho_A = rho_B = 1); BALANCE mode (g>0.5) drives rho_B into the dead zone
(h -> h0/rho sentinel), so B is EXACTLY silent and the top behaviour is the
phase-1 skill by construction (retention 1.000, drift 0.00e+00).

    .venv/bin/python tmp/rl_consolidate_rho_demo.py
Inputs : tmp/out/rl_consolidate_rho/p1_rho.pt, p2_olap.pt   (from rl_consolidate_rho.py)
Output : tmp/out/rl_consolidate_rho_demo.gif
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
from rl_consolidate_rho import arm_fields, OUT as EXP_OUT
from rl_cartpole_swingup_demo import Renderer

OUT = Path(__file__).resolve().parent / "out"


def init_state(p1):
    """Phase-2 update-0 snapshot: phase-1 body + olap gate, pump (B) untrained."""
    fields, _ = arm_fields("olap")
    return {"net": p1["body"], "norm_mean": p1["norm_mean"], "norm_std": p1["norm_std"],
            "hidden": p1["H"], "t": 64, "force_mag": p1["force_mag"], "n_layers": 2,
            "fields": fields, "gate_k": 6.0, "gate_c": 0.0,
            "rho_mode": True, "sigma0": 0.6, "h0": 0.15}


def run_episode(state, horizon=500, seed=0, collect=False):
    """Greedy episode from the bottom; returns (mean cos, last100_up, traj of (x, theta, g))."""
    policy, mean, std = build_policy(state)
    env = CartPoleSwingUp(horizon=horizon, random_start=False, seed=seed,
                          force_mag=state["force_mag"], x_threshold=4.0, continuous=True)
    obs, _ = env.reset(seed=seed)
    coss, traj = [], []
    for _ in range(horizon):
        g = 1.0 / (1.0 + math.exp(-state["gate_k"] * (float(obs[2]) - state["gate_c"])))
        if collect:
            traj.append((env.state[0], env.state[2], g))     # (x, theta, gate)
        _set_field(policy, policy._opt_fields, float(obs[2]), policy._gate_k,
                   policy._gate_c, policy._rho_mode, policy._sigma0, policy._h0)
        on = torch.clamp((torch.tensor(obs, dtype=torch.float32) - mean) / std, -5, 5)
        step = policy.rollout_step(on.unsqueeze(0), greedy=True)
        obs, r, term, trunc, _ = env.step(float(step.action.item()))
        coss.append(env.cos_theta())
        if term or trunc:
            break
    cs = np.array(coss)
    return float(cs.mean()), float((cs[-100:] > 0.9).mean()), traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=45)
    ap.add_argument("--out", default="rl_consolidate_rho_demo.gif")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    p1 = torch.load(f"{EXP_OUT}/p1_rho.pt", weights_only=False)
    p2 = torch.load(f"{EXP_OUT}/p2_olap.pt", weights_only=False)
    rungs = [("upd   0 (pump untrained)", init_state(p1))]

    print("evaluating olap checkpoints (greedy, from bottom) ...")
    evals = []
    for upd, st in p2["checkpoints"]:
        mc, tail, _ = run_episode(st, horizon=args.horizon, seed=args.seed)
        evals.append((upd, st, mc, tail))
        print(f"  upd {upd:4d}  mean cos {mc:+.3f}  last100_up {tail:.3f}")
    solved = next((e for e in evals if e[3] >= 1.0), None)
    best = max(evals, key=lambda e: e[2])
    if solved is not None:
        rungs.append((f"upd {solved[0]:3d} (first full balance)", solved[1]))
    if best[0] != (solved[0] if solved else None):
        rungs.append((f"upd {best[0]:3d} (best)", best[1]))

    rend = Renderer(x_thr=4.0)
    mode_txt = rend.ax.text(0.02, 0.97, "", transform=rend.ax.transAxes,
                            fontsize=9, family="monospace", va="top")
    title = "NNN rho-olap field option (S23.7): balance frozen on A + pump on B"

    flat = []
    for label, st in rungs:
        mc, tail, traj = run_episode(st, horizon=args.horizon, seed=args.seed, collect=True)
        sub = f"phase-2 {label}   cos={mc:+.2f}  last100_up={tail:.2f}"
        seq = traj[::2][:210]
        seq += [traj[-1]] * 18                                # hold the final pose
        flat.extend((x, th, g, title, sub) for x, th, g in seq)
        print(f"rung {label!r}: mean cos {mc:+.3f}  last100_up {tail:.3f}")

    print(f"rendering {len(flat)} frames ...")
    frames = []
    for x, th, g, t, s in flat:
        if g >= 0.5:
            mode_txt.set_text(f"field: BALANCE (A only, B silent)  g={g:.2f}")
            mode_txt.set_color("C2")
        else:
            mode_txt.set_text(f"field: PUMP (B + frozen-A vocab)   g={g:.2f}")
            mode_txt.set_color("C1")
        frames.append(rend.frame(x, th, t, s))

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

    anim = FuncAnimation(fig, _upd, frames=len(frames), interval=1000 / args.fps, blit=True)
    anim.save(OUT / args.out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"saved {OUT / args.out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
