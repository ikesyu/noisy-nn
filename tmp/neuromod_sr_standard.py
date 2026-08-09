"""Comprehensive SR survey on the STANDARD benchmark (sector + sample + LoopParams).

Replaces the legacy vector-sensing SR scripts for the paper: every number here is
measured on the same closed loop the demo runs (docs/idea_neuromod.md, appendix C).
Two sweeps, both multi-seed:

    --sweep behavior   (E1)  nets trained at the standard field strength, then the
                       CONCENTRATION axis is swept at test time in the closed loop.
                       Per (seed, c): capability (separation/signal/task_err) under
                       the scaled fields AND full behavioural metrics from
                       `world.rollout` (foraging, night-home rate, diversity,
                       freezing).  speed_ref is calibrated ONCE per net at c=1 and
                       held fixed across the sweep -- recalibrating per c would
                       normalise away the low-concentration collapse, which is the
                       left arm of the U.

    --sweep train      (E2)  the rigorous SR statement: a FRESH net per training
                       level sigma_train (= peak recruited-unit std of the fields),
                       trained and evaluated at its own level.  Capability plus a
                       behavioural rollout at c=1.  h is a fixed global constant, so
                       report the gauge-invariant ratio h/sigma* with any optimum
                       (docs/idea_neuromod.md section 6).

Outputs: one CSV per (sweep, seed) under --out-dir, purely numeric rows with '#'
metadata (readable via numpy.loadtxt), plus a summary figure per sweep.

Run from the repository root, e.g.:

    .venv/bin/python tmp/neuromod_sr_standard.py --sweep behavior
    .venv/bin/python tmp/neuromod_sr_standard.py --sweep train
"""
from __future__ import annotations

import argparse
import io
import contextlib
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from neuromod import fields as F
from neuromod import protocol as P
from neuromod import world

N_HIDDEN = 2

# Concentration axis for --sweep behavior (test-time, log-ish spacing, wide on the
# high side per the old section-8(b) note).
CONCENTRATIONS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0]
# Training-level axis for --sweep train (sigma_train = base_std of the fields).
SIGMA_TRAIN = [0.1, 0.2, 0.4, 0.6, 0.8, 1.2, 1.8, 2.7, 4.0]

BEHAVIOR_SEEDS = [7, 11, 23, 31, 43]
TRAIN_SEEDS = [7, 11, 23]
# 2D sweep: adaptation level x acute concentration.  Two predictions to separate
# (docs/idea_neuromod.md 1.1a chain 3): the PEAK should track the adaptation
# level (tolerance; relative invariance) while the LEFT COLLAPSE should sit at a
# fixed ABSOLUTE test-time intensity sigma_test = c * sigma_train (the
# recruitment floor; absolute invariance).
GRID_SIGMA_TRAIN = [0.2, 0.4, 0.8, 1.6, 3.2]
GRID_CONCENTRATIONS = [0.05, 0.1, 0.2, 0.35, 0.7, 1.0, 1.5, 2.5, 4.0]
GRID_SEEDS = [7, 11, 23]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sweep", choices=("behavior", "train", "grid"), default="behavior")
    p.add_argument("--model", choices=("sample", "analytic"), default="sample",
                   help="analytic = mean-field control; the doc's claim is that "
                        "the low-dose collapse (the SR left arm) needs the "
                        "sample-level barrier [sample]")
    p.add_argument("--train-steps", type=int, default=40000)
    p.add_argument("--hidden-dim", type=int, default=144)
    p.add_argument("--episodes", type=int, default=6,
                   help="Closed-loop episodes per (seed, level) [6]")
    p.add_argument("--frames", type=int, default=1260,
                   help="Frames per episode; 1260 = three circadian cycles [1260]")
    p.add_argument("--base-std", type=float, default=0.8,
                   help="Standard field strength (behavior sweep trains here) [0.8]")
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--grid-side", type=int, default=61)
    p.add_argument("--out-dir", default="tmp/out/sr_standard")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Override the seed list")
    return p.parse_args()


def build_data(grid_side: int, objects, device):
    """Observation grid and pure-state targets for `protocol.capability`."""
    alphas = world.alpha_states(None)
    positions = world.make_training_grid(grid_side)
    obs = torch.tensor(world.encode_observations(positions, objects),
                       dtype=torch.float32, device=device)
    targets = {
        s: torch.tensor(world.make_behavior_targets(positions, objects, alphas[s]),
                        dtype=torch.float32, device=device)
        for s in world.STATES
    }
    return obs, targets


def train_net(seed, base_std, args, objects, nf, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = P.build_network(args.hidden_dim, n_hidden=N_HIDDEN, base_std=base_std,
                          kind=args.model, t=args.samples, crossing_h=args.crossing_h,
                          in_dim=world.obs_dim()).to(device)
    pool = np.random.default_rng(seed).uniform(
        -1.0, 1.0, size=(16384, 2)).astype(np.float32)
    with contextlib.redirect_stdout(io.StringIO()):
        P.train_blended(net, pool, objects, world.alpha_states(None), nf,
                        N_HIDDEN, args.train_steps, bs=512, lr=3e-4, seed=seed,
                        verbose=False)
    net.eval()
    return net


def make_predict(net, device):
    def predict(o, f):
        with torch.no_grad():
            return P.evaluate_vector_field(
                net, torch.as_tensor(o, dtype=torch.float32, device=device),
                f, N_HIDDEN).cpu().numpy()
    return predict


def measure_speed_ref(predict, objects, nf):
    """This animal's normal |v| at c=1 (30th percentile; see the driver)."""
    pos = np.random.default_rng(0).uniform(-1, 1, size=(4096, 2)).astype(np.float32)
    obs = world.encode_observations(pos, objects)
    mags = []
    for w in (np.float32([1, 0, 0]), np.float32([0, 1, 0]), np.float32([0, 0, 1]),
              np.float32([.5, .5, 0]), np.float32([0, .5, .5])):
        mags.append(np.linalg.norm(
            predict(obs, F.blend_fields(nf, w, world.CATEGORIES)), axis=1))
    return float(np.percentile(np.concatenate(mags), 30))


BEH_KEYS = ["foods_per_1k", "night_home_rate", "shelter_frac", "mean_speed",
            "stall_frac", "wall_frac", "contact_frac", "d_threat_min",
            "diet_evenness", "den_evenness", "path_len"]


def behaviour_at(predict, nf, objects, speed_ref, concentration, seed, args):
    """Mean behavioural metrics over episodes at one concentration."""
    rows = []
    for e in range(args.episodes):
        # Same per-episode seeds across concentrations: every level faces the
        # same threat paths and the same forward-noise stream.
        torch.manual_seed(1000 + e)
        params = world.LoopParams(speed_ref=speed_ref, threat_seed=seed + e)
        rows.append(world.rollout(predict, nf, params, n_frames=args.frames,
                                  concentration=concentration, objects=objects,
                                  seed=seed + e))
    out = {k: float(np.mean([r[k] for r in rows])) for k in BEH_KEYS}
    out["d_threat_min"] = float(np.min([r["d_threat_min"] for r in rows]))
    return out


def capability_at(net, obs, targets, nf, concentration):
    scaled = {k: v * float(concentration) for k, v in nf.items()}
    return P.capability(net, obs, targets, scaled, world.CATEGORIES, world.STATES,
                        world.STATE_TO_FIELD, N_HIDDEN)


def write_csv(path: Path, header_lines, colnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        f.write("# columns: " + ",".join(colnames) + "\n")
        for row in rows:
            f.write(",".join(f"{v:.6g}" for v in row) + "\n")
    print(f"saved {path}")


def sweep_behavior(args, device):
    objects = world.make_scripted_objects()
    obs, targets = build_data(args.grid_side, objects, device)
    nf = {k: v.to(device) for k, v in F.build_fields(
        world.CATEGORIES, args.hidden_dim, args.base_std, 0.22, 0.15).items()}
    seeds = args.seeds or BEHAVIOR_SEEDS
    cols = ["concentration", "separation", "signal", "task_err"] + BEH_KEYS
    for seed in seeds:
        t0 = time.time()
        net = train_net(seed, args.base_std, args, objects, nf, device)
        predict = make_predict(net, device)
        sref = measure_speed_ref(predict, objects, nf)
        print(f"[seed {seed}] trained {args.train_steps} steps in "
              f"{time.time()-t0:.0f}s   speed_ref={sref:.3f}")
        rows = []
        for c in CONCENTRATIONS:
            sep, sig, err = capability_at(net, obs, targets, nf, c)
            beh = behaviour_at(predict, nf, objects, sref, c, seed, args)
            rows.append([c, sep, sig, err] + [beh[k] for k in BEH_KEYS])
            print(f"  c={c:5.2f}  signal={sig:.3f} err={err:.3f}  "
                  f"foods={beh['foods_per_1k']:5.2f} home={beh['night_home_rate']:.2f} "
                  f"speed={beh['mean_speed']:.3f} stall={beh['stall_frac']:.3f}")
        write_csv(Path(args.out_dir) / f"behavior_{args.model}_seed{seed}.csv"
                  if args.model != "sample" else
                  Path(args.out_dir) / f"behavior_seed{seed}.csv",
                  [f"E1 behaviour-level SR sweep (standard benchmark, sector+sample)",
                   f"seed={seed} base_std={args.base_std} h={args.crossing_h} "
                   f"T={args.samples} train_steps={args.train_steps}",
                   f"episodes={args.episodes} frames={args.frames} "
                   f"speed_ref={sref:.4f} (fixed across the sweep)",
                   f"h/sigma at c: h / (c * {args.base_std})"],
                  cols, rows)


def sweep_train(args, device):
    objects = world.make_scripted_objects()
    obs, targets = build_data(args.grid_side, objects, device)
    seeds = args.seeds or TRAIN_SEEDS
    cols = ["sigma_train", "separation", "signal", "task_err"] + BEH_KEYS
    for seed in seeds:
        rows = []
        for st in SIGMA_TRAIN:
            t0 = time.time()
            nf = {k: v.to(device) for k, v in F.build_fields(
                world.CATEGORIES, args.hidden_dim, float(st), 0.22, 0.15).items()}
            net = train_net(seed, float(st), args, objects, nf, device)
            predict = make_predict(net, device)
            sep, sig, err = capability_at(net, obs, targets, nf, 1.0)
            sref = measure_speed_ref(predict, objects, nf)
            beh = behaviour_at(predict, nf, objects, sref, 1.0, seed, args)
            rows.append([st, sep, sig, err] + [beh[k] for k in BEH_KEYS])
            print(f"[seed {seed}] sigma_train={st:4.2f} ({time.time()-t0:.0f}s)  "
                  f"signal={sig:.3f} err={err:.3f}  "
                  f"foods={beh['foods_per_1k']:5.2f} home={beh['night_home_rate']:.2f}")
        write_csv(Path(args.out_dir) / f"train_{args.model}_seed{seed}.csv"
                  if args.model != "sample" else
                  Path(args.out_dir) / f"train_seed{seed}.csv",
                  [f"E2 training-level SR sweep (standard benchmark, sector+sample)",
                   f"seed={seed} h={args.crossing_h} T={args.samples} "
                   f"train_steps={args.train_steps}",
                   f"episodes={args.episodes} frames={args.frames}; "
                   f"behaviour measured at c=1 with per-net speed_ref",
                   f"gauge ratio at any optimum: h/sigma* = {args.crossing_h}/sigma*"],
                  cols, rows)


def sweep_grid(args, device):
    objects = world.make_scripted_objects()
    obs, targets = build_data(args.grid_side, objects, device)
    seeds = args.seeds or GRID_SEEDS
    cols = ["sigma_train", "concentration", "sigma_test", "separation", "signal",
            "task_err"] + BEH_KEYS
    for seed in seeds:
        rows = []
        for st in GRID_SIGMA_TRAIN:
            t0 = time.time()
            nf = {k: v.to(device) for k, v in F.build_fields(
                world.CATEGORIES, args.hidden_dim, float(st), 0.22, 0.15).items()}
            net = train_net(seed, float(st), args, objects, nf, device)
            predict = make_predict(net, device)
            sref = measure_speed_ref(predict, objects, nf)
            print(f"[seed {seed}] trained at sigma_train={st:4.2f} "
                  f"({time.time()-t0:.0f}s)  speed_ref={sref:.3f}")
            for c in GRID_CONCENTRATIONS:
                sep, sig, err = capability_at(net, obs, targets, nf, c)
                beh = behaviour_at(predict, nf, objects, sref, c, seed, args)
                rows.append([st, c, st * c, sep, sig, err]
                            + [beh[k] for k in BEH_KEYS])
                print(f"    c={c:5.2f} (s_test={st*c:5.2f})  signal={sig:.3f} "
                      f"err={err:.3f}  foods={beh['foods_per_1k']:5.2f} "
                      f"home={beh['night_home_rate']:.2f}")
        write_csv(Path(args.out_dir) / f"grid_{args.model}_seed{seed}.csv"
                  if args.model != "sample" else
                  Path(args.out_dir) / f"grid_seed{seed}.csv",
                  [f"2D sweep: adaptation level x acute concentration "
                   f"(standard benchmark, {args.model})",
                   f"seed={seed} h={args.crossing_h} T={args.samples} "
                   f"train_steps={args.train_steps} episodes={args.episodes} "
                   f"frames={args.frames}",
                   "speed_ref calibrated once per net at c=1, fixed across c"],
                  cols, rows)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"sweep={args.sweep}  device={device}  sensing={world.SENSING} "
          f"obs_dim={world.obs_dim()}")
    if args.sweep == "behavior":
        sweep_behavior(args, device)
    elif args.sweep == "grid":
        sweep_grid(args, device)
    else:
        sweep_train(args, device)


if __name__ == "__main__":
    main()
