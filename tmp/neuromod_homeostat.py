"""E7: can a TEACHER-FREE homeostatic rule provide tolerance on the decrease side?

E6 showed that three per-layer multiplicative gains are ENOUGH (capacity), but on
the tone-decrease side the working gains were found with the supervised task
loss.  Biology has no such teacher.  What it does have is firing-rate
homeostasis: each neuron senses its own long-run activity and scales its
synapses to bring that activity back to a set point (Turrigiano).

The rule tested here is the model counterpart:

    set point   nu*_k = unit k's mean crossing rate at the ADAPTED tone,
                measured from plain sensory experience (no labels)
    rule        adjust per-unit gains g1[144], g2[144] to minimise
                sum_k (nu_k - nu*_k)^2  at the NEW tone
    signals     sensory inputs (positions in the world) and the network's own
                activity statistics.  No teacher velocities anywhere.
    readout     g3 is fixed at 1: the speed calibration of the closed loop
                (speed_ref, itself measured without labels) absorbs any output
                magnitude change.

If foraging/homing recover to the reference level, the tolerance claim closes at
the LEARNING-RULE level on both sides.  Outputs a CSV next to the E6 one.

Run from the repository root:
    .venv/bin/python tmp/neuromod_homeostat.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from neuromod import fields as F
from neuromod import protocol as P
from neuromod import world
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "neuromod_tolerance", Path(__file__).parent / "neuromod_tolerance.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

N_HIDDEN = 2
BASE = 0.8
ALPHAS = [0.25, 0.5, 2.0, 4.0]
SEEDS = [7, 11, 23]


def crossing_rates_with_gains(adapter, x, field):
    """Per-unit mean crossing rates of both hidden layers, differentiable in g."""
    n = adapter.net
    a1 = adapter.g1 * n.fcs[0](x)
    a1 = n.sampled_layer(a1)
    z1 = n.gaussian_crossing[0](a1, field)
    a2 = adapter.g2 * n.fcs[1](z1)
    z2 = n.gaussian_crossing[1](a2, field)
    return z1.mean(dim=(0, 1)), z2.mean(dim=(0, 1))


def homeostat(adapter, nf_new, setpoints, objects, seed, steps, lr, device):
    """Gradient descent on the nu set-point loss.  No labels are ever used."""
    nu1_star, nu2_star = setpoints
    opt = torch.optim.Adam([adapter.g1, adapter.g2], lr=lr)
    rng = np.random.default_rng(seed + 900)
    field = F.blend_fields(nf_new, np.float32([1, 1, 1]) / 3, world.CATEGORIES)
    for step in range(steps):
        # Fresh sensory experience each step: random positions, random threat
        # placement -- the same world statistics the animal lives in.
        pos = rng.uniform(-1, 1, (256, 2)).astype(np.float32)
        obs = torch.tensor(world.encode_observations(pos, objects),
                          dtype=torch.float32, device=device)
        nu1, nu2 = crossing_rates_with_gains(adapter, obs, field)
        loss = ((nu1 - nu1_star) ** 2).sum() + ((nu2 - nu2_star) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            adapter.g1.clamp_(min=0.0)
            adapter.g2.clamp_(min=0.0)
    adapter.eval()
    return adapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-steps", type=int, default=40000)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--hidden-dim", type=int, default=144)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--frames", type=int, default=1260)
    p.add_argument("--grid-side", type=int, default=61)
    p.add_argument("--out", default="tmp/out/sr_standard/homeostat.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects = world.make_scripted_objects()
    alphas_states = world.alpha_states(None)
    positions = world.make_training_grid(args.grid_side)
    obs_grid = torch.tensor(world.encode_observations(positions, objects),
                            dtype=torch.float32, device=device)
    targets = {s: torch.tensor(world.make_behavior_targets(
                   positions, objects, alphas_states[s]),
                   dtype=torch.float32, device=device) for s in world.STATES}
    nf_base = {k: v.to(device) for k, v in F.build_fields(
        world.CATEGORIES, args.hidden_dim, BASE, 0.22, 0.15).items()}

    rows = []
    for seed in SEEDS:
        t0 = time.time()
        net = T.train_base(seed, args, objects, nf_base, device)
        pr = T.make_predict(net, device)
        sref0 = T.speed_ref_of(pr, objects, nf_base)
        f0, h0 = T.behaviour(pr, nf_base, objects, sref0, seed, args)
        print(f"[seed {seed}] base ({time.time()-t0:.0f}s) foods={f0:.2f} home={h0:.2f}")
        rows.append((seed, 1.0, f0, h0,
                     T.task_err(net, obs_grid, targets, nf_base)))

        # Set points: mean crossing rate per unit at the adapted tone, measured
        # from plain experience (uniform positions; the mixed field 1/3,1/3,1/3
        # as a neutral stand-in for the runtime blend).
        probe = torch.tensor(world.encode_observations(
            np.random.default_rng(0).uniform(-1, 1, (2048, 2)).astype(np.float32),
            objects), dtype=torch.float32, device=device)
        base_adapter = T.GainAdapter(net, True, args.hidden_dim).to(device)
        field0 = F.blend_fields(nf_base, np.float32([1, 1, 1]) / 3,
                                world.CATEGORIES)
        with torch.no_grad():
            nu1_star, nu2_star = crossing_rates_with_gains(
                base_adapter, probe, field0)

        for alpha in ALPHAS:
            nf_new = {k: v * alpha for k, v in nf_base.items()}
            ad = T.GainAdapter(net, True, args.hidden_dim).to(device)
            t1 = time.time()
            homeostat(ad, nf_new, (nu1_star, nu2_star), objects, seed,
                      args.steps, args.lr, device)
            pra = T.make_predict(ad, device)
            sra = T.speed_ref_of(pra, objects, nf_new)
            fl, hl = T.behaviour(pra, nf_new, objects, sra, seed, args)
            el = T.task_err(ad, obs_grid, targets, nf_new)
            print(f"  alpha={alpha:4.2f} homeostat ({time.time()-t1:.0f}s) "
                  f"foods={fl:5.2f} home={hl:.2f} err={el:.3f} "
                  f"g1~{float(ad.g1.detach().mean()):.2f} "
                  f"g2~{float(ad.g2.detach().mean()):.2f}")
            rows.append((seed, alpha, fl, hl, el))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("# E7 teacher-free nu-setpoint homeostat "
                "(docs/idea_neuromod.md 7.13/E7)\n")
        f.write(f"# base=0.8 steps={args.steps} lr={args.lr}; alpha=1 rows are "
                "the adapted reference\n")
        f.write("# columns: seed,alpha,foods_per_1k,night_home_rate,task_err\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.5f}\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
