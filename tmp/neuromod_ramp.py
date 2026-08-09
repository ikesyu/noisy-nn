"""E9: the same tone change breaks the animal when abrupt, passes unnoticed when slow.

The literal temporal form of the acute/chronic dissociation (7.13).  One
closed-loop life of 8400 frames (20 day/night cycles); the modulatory tone
c(t) rises from 1x to 4x either as a STEP or as a slow RAMP.  The teacher-free
homeostat of E7 (per-unit gains, nu set-point rule) runs ONLINE, fed only by
the observation the animal itself experiences each frame.

    schedules   step:  c = 1 until frame 2100, then 4.0
                ramp:  c = 1 until 2100, linear to 4.0 by 6300, hold
    conditions  rule on / rule off, per schedule
    prediction  step + rule off   : foraging drops to the acute 4x level and stays
                step + rule on    : transient collapse, then recovery
                ramp + rule on    : no visible dip -- tolerance in real time

Run from the repository root:
    .venv/bin/python tmp/neuromod_ramp.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from neuromod import fields as F
from neuromod import world
import importlib.util

_s1 = importlib.util.spec_from_file_location(
    "neuromod_tolerance", Path(__file__).parent / "neuromod_tolerance.py")
T = importlib.util.module_from_spec(_s1)
_s1.loader.exec_module(T)

N_HIDDEN = 2
BASE = 0.8
SEEDS = [int(x) for x in __import__("os").environ.get("E9_SEEDS","7 11 23").split()]
T_PRE, T_RAMP_END, T_TOTAL = 2100, 6300, 8400
C_HI = 4.0


def tone(frame: int, schedule: str) -> float:
    if frame < T_PRE:
        return 1.0
    if schedule == "step":
        return C_HI
    if frame >= T_RAMP_END:
        return C_HI
    return 1.0 + (C_HI - 1.0) * (frame - T_PRE) / (T_RAMP_END - T_PRE)


def life(net, nf_base, objects, seed, schedule, rule_on, args, device):
    """One continuous life with time-varying tone and (optionally) the online rule."""
    adapter = T.GainAdapter(net, True, args.hidden_dim).to(device)
    predict = T.make_predict(adapter, device)
    sref = T.speed_ref_of(predict, objects, nf_base)

    # nu set points from pre-shift experience (the animal's normal statistics)
    probe = torch.tensor(world.encode_observations(
        np.random.default_rng(0).uniform(-1, 1, (2048, 2)).astype(np.float32),
        objects), dtype=torch.float32, device=device)
    field0 = F.blend_fields(nf_base, np.float32([1, 1, 1]) / 3, world.CATEGORIES)
    with torch.no_grad():
        z1 = adapter.net.gaussian_crossing[0](
            adapter.net.sampled_layer(adapter.g1 * adapter.net.fcs[0](probe)), field0)
        nu1_star = z1.mean(dim=(0, 1)).detach()
        z2 = adapter.net.gaussian_crossing[1](
            adapter.g2 * adapter.net.fcs[1](z1), field0)
        nu2_star = z2.mean(dim=(0, 1)).detach()

    opt = torch.optim.Adam([adapter.g1, adapter.g2], lr=args.online_lr)
    params = world.LoopParams(speed_ref=sref, threat_seed=seed)
    state = world.initialize_demo_state(objects)
    state["threat_vels"] = world.make_threat_velocities(
        state["objects"], params.threat_speed, seed)

    # Homeostatic sensing.  Three properties had to match the biology before
    # this worked at all; each mismatch measurably destroyed the animal:
    #   1. the sensor integrates over TIME (never a single instant);
    #   2. the set point is the animal's OWN normal activity;
    #   3. the sensor's timescale is the LIFETIME, not the last few seconds --
    #      a reservoir sample of pre-shift experience, kept fixed afterwards.
    #      Sensing on only the recent (post-collapse) experience poisons the
    #      target: bad behaviour -> impoverished experience -> wrong rate
    #      error -> no recovery.  With the lifetime buffer the acute 4x step
    #      recovers fully (measured: foods 7.3/1k after recovery).
    # The sensing field uses the neutral 1/3-blend times the current tone, so
    # the target does not flutter with the behavioural state of the moment.
    BUF = 256
    buffer = [probe[i % probe.shape[0]].clone() for i in range(BUF)]
    seen = 0
    neutral = F.blend_fields(nf_base, np.float32([1, 1, 1]) / 3,
                             world.CATEGORIES)
    nu1_star = nu1_star.clone()      # placeholder; replaced by EMA below
    nu2_star = nu2_star.clone()
    ema_tau = 0.01

    def buffer_rates(field_now, grad):
        batch = torch.stack(buffer)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            a1 = adapter.net.sampled_layer(
                adapter.g1 * adapter.net.fcs[0](batch))
            z1 = adapter.net.gaussian_crossing[0](a1, field_now)
            nu1 = z1.mean(dim=(0, 1))
            z2 = adapter.net.gaussian_crossing[1](
                adapter.g2 * adapter.net.fcs[1](z1), field_now)
            nu2 = z2.mean(dim=(0, 1))
        return nu1, nu2

    eats, speeds, tones = [], [], []
    torch.manual_seed(1000 + seed)
    for k in range(T_TOTAL):
        c = tone(k, schedule)
        rec = world.advance_frame(state, predict, nf_base, params,
                                  concentration=c, frame=k, n_frames=T_TOTAL)
        eats.append(int(rec["ate"]))
        speeds.append(rec["speed"])
        tones.append(c)
        if k < T_PRE:
            # Reservoir-sample the pre-shift life into the lifetime buffer.
            obs_now = torch.tensor(world.encode_observation(
                state["pos"], state["objects"],
                food_strengths=state["food_strengths"]),
                dtype=torch.float32, device=device)
            seen += 1
            if seen <= BUF:
                buffer[seen - 1] = obs_now
            else:
                j = int(np.random.default_rng(k).integers(0, seen))
                if j < BUF:
                    buffer[j] = obs_now
            nu1, nu2 = buffer_rates(neutral * c, grad=False)
            nu1_star = (1 - ema_tau) * nu1_star + ema_tau * nu1
            nu2_star = (1 - ema_tau) * nu2_star + ema_tau * nu2
        elif rule_on:
            nu1, nu2 = buffer_rates(neutral * c, grad=True)
            loss = ((nu1 - nu1_star) ** 2).sum() + ((nu2 - nu2_star) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                adapter.g1.clamp_(min=0.0)
                adapter.g2.clamp_(min=0.0)
    return np.array(eats), np.array(speeds), np.array(tones)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-steps", type=int, default=40000)
    p.add_argument("--online-lr", type=float, default=0.02)
    p.add_argument("--hidden-dim", type=int, default=144)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--grid-side", type=int, default=61)
    p.add_argument("--episodes", type=int, default=1)   # unused; T API compat
    p.add_argument("--frames", type=int, default=1260)  # unused; T API compat
    p.add_argument("--out-dir", default="tmp/out/sr_standard")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects = world.make_scripted_objects()
    nf_base = {k: v.to(device) for k, v in F.build_fields(
        world.CATEGORIES, args.hidden_dim, BASE, 0.22, 0.15).items()}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        t0 = time.time()
        net = T.train_base(seed, args, objects, nf_base, device)
        print(f"[seed {seed}] base trained ({time.time()-t0:.0f}s)")
        for schedule in __import__("os").environ.get("E9_SCHED","step ramp").split():
            for rule_on in ([True] if __import__("os").environ.get("E9_RULEONLY") else [False, True]):
                t1 = time.time()
                eats, speeds, tones = life(net, nf_base, objects, seed,
                                           schedule, rule_on, args, device)
                tag = f"{schedule}_{'rule' if rule_on else 'norule'}_seed{seed}"
                np.savetxt(out / f"ramp_{tag}.csv",
                           np.stack([tones, eats, speeds], axis=1),
                           delimiter=",", header="tone,ate,speed", comments="# ")
                w = 840
                late = eats[-w:].sum() / w * 1000
                mid = eats[T_PRE:T_PRE + w].sum() / w * 1000
                print(f"  {tag} ({time.time()-t1:.0f}s): foods/1k "
                      f"post-shift={mid:.2f} final={late:.2f}")


if __name__ == "__main__":
    main()
