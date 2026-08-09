"""E4: static paper figures for the closed-loop demo.

Two figures from the STANDARD demo animal (sector + sample + blended + circadian):

    fig_ethogram.png    one representative life, three day/night cycles.
                        Panels: day/night band + arbitration weights w(t)
                        (stacked), eating / homing events, speed |v|.
                        The GIF's story as a single printable figure.

    fig_diversity.png   diet and den usage over 3 seeds x 6 episodes:
                        per-food and per-den counts with evenness values
                        (C.9 metrics), showing the regrowth delay does its job.

Run from the repository root:
    .venv/bin/python tmp/neuromod_ethogram.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from neuromod import fields as F
from neuromod import protocol as P
from neuromod import world
import importlib.util

_s1 = importlib.util.spec_from_file_location(
    "neuromod_tolerance", Path(__file__).parent / "neuromod_tolerance.py")
T = importlib.util.module_from_spec(_s1)
_s1.loader.exec_module(T)

N_HIDDEN = 2
SEEDS = [7, 11, 23]


def one_life(net, nf, objects, seed, frames, device):
    predict = T.make_predict(net, device)
    sref = T.speed_ref_of(predict, objects, nf)
    params = world.LoopParams(speed_ref=sref, threat_seed=seed)
    state = world.initialize_demo_state(objects)
    state["threat_vels"] = world.make_threat_velocities(
        state["objects"], params.threat_speed, seed)
    rec_w, rec_speed, rec_eat, rec_home, rec_phase = [], [], [], [], []
    prev_goal, got_home = state["goal"], False
    torch.manual_seed(1000 + seed)
    for k in range(frames):
        rec = world.advance_frame(state, predict, nf, params,
                                  frame=k, n_frames=frames)
        rec_w.append(rec["weights"].copy())
        rec_speed.append(rec["speed"])
        rec_eat.append(bool(rec["ate"]))
        goal = state["goal"]
        home_event = False
        if goal == "shelter" and prev_goal == "food":
            got_home = False
        if goal == "shelter" and rec["inside_shelter"] and not got_home:
            got_home, home_event = True, True
        rec_home.append(home_event)
        rec_phase.append(world.circadian_phase(state["clock"],
                                               params.circadian_period))
        prev_goal = goal
    return (np.array(rec_w), np.array(rec_speed), np.array(rec_eat),
            np.array(rec_home), np.array(rec_phase), params.day_fraction)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-steps", type=int, default=40000)
    p.add_argument("--hidden-dim", type=int, default=144)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--frames", type=int, default=1260)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--grid-side", type=int, default=61)
    p.add_argument("--out-dir", default="tmp/out/sr_standard")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects = world.make_scripted_objects()
    nf = {k: v.to(device) for k, v in F.build_fields(
        world.CATEGORIES, args.hidden_dim, 0.8, 0.22, 0.15).items()}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- fig 1: ethogram (seed 7, one life) ----------
    t0 = time.time()
    net = T.train_base(7, args, objects, nf, device)
    print(f"[seed 7] trained ({time.time()-t0:.0f}s)")
    W, speed, ate, home, phase, day_frac = one_life(
        net, nf, objects, 7, args.frames, device)
    t = np.arange(args.frames)
    night = phase >= day_frac

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(3, 1, figsize=(10, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 0.8, 1.4]})
    colors = {"food": "tab:green", "threat": "tab:red", "shelter": "tab:blue"}
    axes[0].stackplot(t, W[:, 0], W[:, 1], W[:, 2],
                      colors=[colors[c] for c in world.CATEGORIES],
                      labels=[f"w_{c}" for c in world.CATEGORIES], alpha=0.85)
    axes[0].set_ylabel("arbitration w(t)")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8, loc="center left", ncol=3)
    ei = np.flatnonzero(ate)
    hi = np.flatnonzero(home)
    axes[1].scatter(ei, np.ones_like(ei), marker="v", color="tab:green",
                    s=45, label="ate")
    axes[1].scatter(hi, np.zeros_like(hi), marker="s", color="tab:blue",
                    s=45, label="reached den")
    axes[1].set_ylim(-0.6, 1.6)
    axes[1].set_yticks([])
    axes[1].legend(fontsize=8, loc="center left")
    axes[2].plot(t, speed, color="0.3", lw=1)
    axes[2].set_ylabel("speed |v|")
    axes[2].set_xlabel("frame")
    for ax in axes:
        ax.grid(alpha=0.25)
        for s0 in range(0, args.frames, 420):
            n0 = s0 + int(420 * day_frac)
            ax.axvspan(n0, min(s0 + 420, args.frames), color="0.85", alpha=0.5,
                       zorder=0)
    axes[0].set_title("Ethogram: one life, three day/night cycles "
                      "(grey = night)")
    fig.tight_layout()
    fig.savefig(out / "fig_ethogram.png", dpi=150)
    plt.close(fig)
    print("saved fig_ethogram.png")

    # ---------- fig 2: diversity over seeds x episodes ----------
    food_hits = np.zeros(objects["food"].shape[0])
    den_hits = np.zeros(objects["shelter"].shape[0])
    diet_es, den_es = [], []
    for seed in SEEDS:
        net = net if seed == 7 else T.train_base(seed, args, objects, nf, device)
        predict = T.make_predict(net, device)
        sref = T.speed_ref_of(predict, objects, nf)
        for e in range(args.episodes):
            torch.manual_seed(1000 + e)
            params = world.LoopParams(speed_ref=sref, threat_seed=seed + e)
            r = world.rollout(predict, nf, params, n_frames=args.frames,
                              objects=objects, seed=seed + e)
            food_hits += np.array(r["food_hits"])
            den_hits += np.array(r["den_hits"])
            diet_es.append(r["diet_evenness"])
            den_es.append(r["den_evenness"])
        print(f"[seed {seed}] episodes done")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    axes[0].bar(range(len(food_hits)), food_hits, color="tab:green",
                edgecolor="k")
    axes[0].set_xticks(range(len(food_hits)))
    axes[0].set_xticklabels([f"food {i}" for i in range(len(food_hits))])
    axes[0].set_ylabel("times eaten")
    axes[0].set_title(f"diet  (evenness {np.mean(diet_es):.2f})")
    axes[1].bar(range(len(den_hits)), den_hits, color="tab:blue",
                edgecolor="k")
    axes[1].set_xticks(range(len(den_hits)))
    axes[1].set_xticklabels([f"den {i}" for i in range(len(den_hits))])
    axes[1].set_ylabel("nights slept")
    axes[1].set_title(f"den use  (evenness {np.mean(den_es):.2f})")
    for ax in axes:
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle(f"Diversity over {len(SEEDS)} seeds x {args.episodes} episodes",
                 y=1.03)
    fig.tight_layout()
    fig.savefig(out / "fig_diversity.png", dpi=150, bbox_inches="tight")
    print("saved fig_diversity.png")


if __name__ == "__main__":
    main()
