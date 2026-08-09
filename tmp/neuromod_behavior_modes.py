"""Neuromodulator-like noise fields recruit functional subnetworks from ONE
shared weight set, producing Foraging, Avoidance, or Sheltering.

This is the reference driver for the standard benchmark.  The problem, the
fields, the protocol, and the drawing live in `tmp/neuromod/`; this file only
wires them together, so a new challenge script under `tmp/` can import the
package and vary only its own question.

    world     scene, sensing, behaviour targets, closed-loop drives
    fields    noise fields on the unit sheet + crossing-rate participation
    protocol  how the benchmark is trained and scored
    viz       panels, animation, file saves

Run from the repository root:

    .venv/bin/python tmp/neuromod_behavior_modes.py       # endless live animation
    .venv/bin/python tmp/neuromod_behavior_modes.py --demo-mode pure-panels
    .venv/bin/python tmp/neuromod_behavior_modes.py --save --anim-frames 600
                                                    # write files, 600-frame gif
    .venv/bin/python tmp/neuromod_behavior_modes.py --field-radius 0.18 --epochs 800

Context: `docs/idea_neuromod.md`.  Symbols: `docs/idea_core.md`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from neuromod import fields as F
from neuromod import protocol as P
from neuromod import viz, world


# The wall-consistent (soft-reflection) teacher moved into the benchmark
# definition: see `world.make_behavior_targets` (Stage 1).

# Single source of truth for every closed-loop default.  The argparse defaults
# below are READ FROM this instance, so `world.LoopParams()` and this driver can
# never disagree again (they used to: shelter_radius 0.08 vs 0.12, threat_range
# 0.4 vs 0.5, refuge_range 0 vs 0.3, risk_hunger 0 vs 0.9, ...).
_LP = world.LoopParams()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("\n\n")[0],
    )
    # Model
    p.add_argument("--model", choices=("analytic", "sample"), default="sample",
                   help="sample = the mechanism, real noise injection with a "
                        "crossing threshold h>0, and the only level at which SR "
                        "exists; analytic = the mean field, no subthreshold "
                        "barrier -- use it only when you specifically want the "
                        "closed form [sample]")
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda. The sample model runs T stochastic "
                        "forward passes per input (~65x the analytic cost: 47 vs "
                        "0.7 ms/step here), so it is only practical on a GPU -- "
                        "40k training steps take ~31 min on CPU and ~1.5 min on "
                        "this RTX 5080 [auto]")
    p.add_argument("--sensing", choices=("sector", "vector"), default="sector",
                   help="sector = K angular sectors x (food,threat,shelter,wall) "
                        "bounded proximities (Stage 1: shared substrate, walls "
                        "perceivable); vector = original 6D nearest-object "
                        "relative vectors [sector]")
    p.add_argument("--hidden-dim", type=int, default=144,
                   help="Hidden units per layer; must be a perfect square. "
                        "144 for sector sensing -- the shared-substrate task is "
                        "genuinely nonlinear and underfits at 64 (the 6D vector "
                        "code, in contrast, makes the teacher near-LINEAR, which "
                        "is exactly the sec4 triviality critique) [144]")
    p.add_argument("--sectors", type=int, default=8,
                   help="Angular sectors K for sector sensing [8]")
    p.add_argument("--hidden-layers", type=int, default=2,
                   help="Hidden layers. 1 is where sigma-only recruitment gates "
                        "exactly (no sigma=0 leak) [2]")
    p.add_argument("--samples", type=int, default=64,
                   help="Stochastic forward samples T, sample model only [64]")
    p.add_argument("--crossing-h", type=float, default=0.2,
                   help="Crossing threshold h, sample model only [0.2]")
    # Noise field
    p.add_argument("--base-std", type=float, default=0.8,
                   help="Peak noise std of a recruited unit [0.8]")
    p.add_argument("--field-radius", type=float, default=0.28,
                   help="Ring radius of the bump centres on the unit sheet. All "
                        "pairs overlap equally; smaller = more sharing [0.28]")
    p.add_argument("--sigma", type=float, default=0.22,
                   help="Gaussian WIDTH of a bump on the sheet (not the noise "
                        "strength) [0.22]")
    p.add_argument("--theta", type=float, default=0.15,
                   help="Intensity cut that carves out the field support [0.15]")
    # Task
    p.add_argument("--alpha-mix", type=float, default=None,
                   help="Off-drive weight as a fraction of the dominant one. "
                        "Default keeps the near-one-hot targets; ~0.3 makes the "
                        "three behaviours genuinely compete for the same inputs")
    p.add_argument("--target-gamma", type=float, default=2.0,
                   help="Speed saturation gain in the tanh target rule [2.0]")
    p.add_argument("--wall-margin", type=float, default=0.15,
                   help="(G) width of the boundary band where the outward "
                        "component of the TRAINING targets is smoothly "
                        "reflected, so the learned policy veers off walls; "
                        "0 restores the wall-agnostic targets [0.15]")
    p.add_argument("--wall-kappa", type=float, default=1.0,
                   help="(G) reflection strength at the wall: the outward "
                        "target component becomes -kappa times itself there. "
                        "0 = pure damping, which measurably WORSENS wall "
                        "dwell time (viscous band); keep near 1 [1.0]")
    p.add_argument("--grid-side", type=int, default=61,
                   help="Training grid resolution per axis.  Sector sensing "
                        "varies faster in space than the 6D vector code (angles "
                        "sweep quickly near objects), so it needs the denser "
                        "grid; 31 sufficed for vector [61]")
    # Training
    p.add_argument("--train-mode", choices=("blended", "pure"), default="blended",
                   help="blended = sample the RUNTIME distribution (field blends "
                        "on the goal<->threat manifold x random threat "
                        "placements); required for natural closed-loop "
                        "behaviour with sector sensing (sec7.12b). pure = the "
                        "original three-corner protocol; the only mode that "
                        "supports the sec7.11 zero-shot interpolation claim "
                        "[blended]")
    p.add_argument("--train-steps", type=int, default=40000,
                   help="Adam steps for --train-mode blended [40000]")
    p.add_argument("--train-batch", type=int, default=512,
                   help="Minibatch size for --train-mode blended [512]")
    p.add_argument("--train-points", type=int, default=16384,
                   help="Random-position pool size for --train-mode blended; "
                        "fixed grids overfit the sector code between grid "
                        "points [16384]")
    p.add_argument("--epochs", type=int, default=5000,
                   help="Training epochs, --train-mode pure only [5000]")
    p.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate [3e-4]")
    p.add_argument("--train-chunk", type=int, default=0,
                   help="Minibatch size, --train-mode pure only; 0 = full batch [0]")
    p.add_argument("--seed", type=int, default=7, help="Random seed [7]")
    # Scene
    p.add_argument("--dynamic-objects", action="store_true",
                   help="Slowly move objects during the animation (demo only)")
    p.add_argument("--object-speed", type=float, default=0.002)
    # Closed loop
    p.add_argument("--demo-mode",
                   choices=("scripted", "cycle", "pure-panels", "fields"),
                   default="scripted",
                   help="scripted = reactive closed loop; cycle = smooth blend "
                        "sweep; pure-panels / fields = static diagnostics")
    p.add_argument("--speed-gain", type=float, default=_LP.speed_gain,
                   help="Agent speed scale [0.9]")
    p.add_argument("--anim-frames", type=int, default=360,
                   help="Length (frames) of the SAVED animation with --save; "
                        "the live window (no --save) runs endlessly [360]")
    p.add_argument("--eat-radius", type=float, default=_LP.eat_radius)
    # arrival radius raised from 0.08 to sit just inside the DRAWN shelter
    # circle (0.125): with 0.08 the agent spent tens of frames visually inside
    # the shelter yet not counted as arrived (deceleration law is slowest there)
    p.add_argument("--shelter-radius", type=float, default=_LP.shelter_radius)
    p.add_argument("--circadian-period", type=int, default=None,
                   help="Frames per full day+night cycle, --goal-mode circadian "
                        "[420]")
    p.add_argument("--day-fraction", type=float, default=None,
                   help="Share of the cycle spent foraging; the rest is night, "
                        "and the night must cover TRAVEL to a shelter plus "
                        "sleep, not just sleep [0.62]")
    p.add_argument("--refuge-range", type=float, default=_LP.refuge_range,
                   help="Damp threat urgency by shelter proximity while "
                        "shelter-bound, so the agent dashes in instead of "
                        "hovering at the doorstep when a threat loiters near "
                        "the entrance; 0 restores the original arbitration "
                        "[0.30]")
    p.add_argument("--risk-hunger", type=float, default=_LP.risk_hunger,
                   help="Starvation-predation trade-off: damp threat urgency "
                        "by up to this factor once hunger passes 0.6 "
                        "(smoothstep), so a blocked foraging phase ends with "
                        "a dart at the food instead of dragging on; 0 "
                        "disables.  This is the LIVELY end of the trade-off and "
                        "is the shipped default: with --panic-range 0.22 it "
                        "measures the best foraging (8.9-9.6 foods/1k over four "
                        "nets) and sheltering (0.19-0.22), no stalls, and low "
                        "wall dwell -- but threat contact is reduced, NOT "
                        "removed (0.09-0.19%% of frames, closest approach "
                        "0.021-0.040).  For zero measured contact use "
                        "--risk-hunger 0.6 --panic-range 0.34, which costs "
                        "about 20%% of the foraging [0.9]")
    p.add_argument("--panic-range", type=float, default=_LP.panic_range,
                   help="Radius inside which the courage (--risk-hunger) and "
                        "refuge (--refuge-range) dampings are VETOED, so "
                        "avoidance is never switched off at point-blank range. "
                        "This is what separates 'dares to approach a distant "
                        "threat' from 'walks into one'; 0 restores the old "
                        "all-or-nothing courage [0.22]")
    p.add_argument("--turn-rate", type=float, default=_LP.turn_rate,
                   help="Max heading change per frame in radians.  0.65 "
                        "measured wall dwell AND stalls at exactly zero; 0.20 "
                        "leaves 1.5%% wall dwell [0.65]")
    p.add_argument("--speed-ref", type=float, default=None,
                   help="This animal's normal |v|, which rescales 'learned' "
                        "speed mode (see world.step_agent). Default: measured "
                        "automatically from the trained net; pass a number to "
                        "override or 0 to disable")
    p.add_argument("--hunger-rate", type=float, default=_LP.hunger_rate)
    p.add_argument("--eat-amount", type=float, default=_LP.eat_amount)
    p.add_argument("--rest-frames", type=int, default=_LP.rest_frames)
    p.add_argument("--food-regrow-frames", type=int, default=_LP.food_regrow_frames,
                   help="Frames an eaten food stays gone before it regrows.  0 "
                        "restores the old rule, which respawned it as soon as "
                        "the animal was 0.25 away -- one patch could then be "
                        "harvested over and over and the diet collapsed onto it "
                        "(one of three foods eaten 1 time against 200 and 183, "
                        "and one den used on 105 nights against 5).  260 "
                        "measures diet evenness 0.86 and den evenness 0.94 "
                        "with the night-home rate unchanged [260]")
    p.add_argument("--threat-gain", type=float, default=_LP.threat_gain)
    # threat-range / neuromod-smoothing raised from 0.40 / 0.12: at the old
    # values the urgency ramp plus the field cross-fade lag (~8 frames) let the
    # agent graze right past a threat before the flee field took over.  0.50 /
    # 0.20 nearly doubles the minimum threat distance (0.15 -> 0.27) and cuts
    # d<0.3 close calls 3.3% -> 1.0% on the wallG network; pushing further
    # (range 0.6, gain 2.3) collapses foraging and re-inflates wall dwell.
    p.add_argument("--threat-range", type=float, default=_LP.threat_range)
    p.add_argument("--neuromod-smoothing", type=float, default=_LP.neuromod_smoothing)
    p.add_argument("--threat-motion", choices=("moving", "static"), default=_LP.threat_motion)
    p.add_argument("--threat-speed", type=float, default=_LP.threat_speed)
    p.add_argument("--velocity-smoothing", type=float, default=_LP.velocity_smoothing)
    # Output
    p.add_argument("--save", action="store_true",
                   help="Write figures/animation to --out-dir instead of showing")
    p.add_argument("--out-dir", default=None, help="Output directory [tmp/out]")
    p.add_argument("--tag", default="neuromod",
                   help="Filename prefix for saved output [neuromod]")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--show-reference", action="store_true",
                   help="Overlay the analytic target vector field as a diagnostic")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    world.set_sensing(args.sensing)
    world.set_sectors(args.sectors)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    def out(name, ext):
        return viz.resolve_out(f"{args.tag}_{name}.{ext}", args.out_dir) \
            if args.save else None

    # --- Scene ---
    objects = world.make_scripted_objects()
    velocities = world.make_object_velocities(rng, objects, args.object_speed)
    print("Scene: " + ", ".join(f"{k}={objects[k].shape[0]}" for k in world.CATEGORIES))

    # --- Data ---
    alphas = world.alpha_states(args.alpha_mix)
    positions = world.make_training_grid(args.grid_side)
    obs_np = world.encode_observations(positions, objects)
    obs = torch.tensor(obs_np, dtype=torch.float32)
    targets = {
        s: torch.tensor(world.make_behavior_targets(
            positions, objects, alphas[s], gamma=args.target_gamma,
            wall_margin=args.wall_margin, wall_kappa=args.wall_kappa),
            dtype=torch.float32)
        for s in world.STATES
    }

    # --- Noise fields ---
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
        else args.device)
    noise_fields = {k: v.to(device) for k, v in F.build_fields(
        world.CATEGORIES, args.hidden_dim, args.base_std, args.sigma, args.theta,
        radius=args.field_radius).items()}
    state_fields = {s: noise_fields[world.STATE_TO_FIELD[s]] for s in world.STATES}

    # --- Model ---
    n_hidden = args.hidden_layers
    net = P.build_network(args.hidden_dim, n_hidden=n_hidden, base_std=args.base_std,
                        kind=args.model, t=args.samples, crossing_h=args.crossing_h,
                        in_dim=world.obs_dim()).to(device)
    if args.model == "sample" and device.type == "cpu":
        print("WARNING: the sample model on CPU is ~65x slower than analytic; "
              f"{args.train_steps} steps will take roughly "
              f"{args.train_steps * 47e-3 / 60:.0f} min.  Use --device cuda.")
    print(f"\nModel : {args.model}  sensing={args.sensing}  "
          f"structure=[{world.obs_dim()}, "
          + ", ".join([str(args.hidden_dim)] * n_hidden) + ", 2]")
    if args.train_mode == "blended":
        train_desc = (f"blended steps={args.train_steps} bs={args.train_batch} "
                      f"pool={args.train_points}")
    else:
        train_desc = f"pure epochs={args.epochs} grid={args.grid_side}x{args.grid_side}"
    print(f"Train : {train_desc}  lr={args.lr}  "
          f"alpha_mix={args.alpha_mix if args.alpha_mix is not None else 'one-hot'}  "
          f"wall_margin={args.wall_margin}  wall_kappa={args.wall_kappa}")
    print("        same obs, same weights, different noise fields.\n")

    # --- Train ---
    if args.train_mode == "blended":
        rng_pool = np.random.default_rng(args.seed)
        pool = rng_pool.uniform(-1.0, 1.0,
                                size=(args.train_points, 2)).astype(np.float32)
        history = P.train_blended(net, pool, objects, alphas, noise_fields,
                                  n_hidden, args.train_steps, bs=args.train_batch,
                                  lr=args.lr, seed=args.seed,
                                  gamma=args.target_gamma,
                                  wall_margin=args.wall_margin,
                                  wall_kappa=args.wall_kappa)
    else:
        history = P.train(net, obs, targets, state_fields, world.STATES, n_hidden,
                          args.epochs, args.lr, chunk=args.train_chunk)

    print("\nFinal per-state MSE:")
    for state, value in P.final_losses(net, obs, targets, state_fields,
                                       world.STATES, n_hidden).items():
        print(f"  {state:15s}: {value:.5f}")

    # --- Diagnostics: does the network use the field, and who participates? ---
    print("\nNoise-field separation (mean ||y_i - y_j|| over "
          f"{obs.shape[0]} observations; near zero => field ignored):")
    for pair, value in P.field_separation(net, obs, noise_fields,
                                          world.CATEGORIES, n_hidden).items():
        print(f"  {pair:20s} {value:.4f}")

    sample_idx = torch.randperm(obs.shape[0])[:min(256, obs.shape[0])]
    report = F.overlap_report(net, obs[sample_idx], noise_fields, world.CATEGORIES,
                              n_hidden)
    F.print_overlap_report(report)

    # --- Visualise ---
    net.eval()

    def predict(obs_array, field):
        with torch.no_grad():
            return P.evaluate_vector_field(
                net, torch.as_tensor(obs_array, dtype=torch.float32, device=device),
                field, n_hidden).cpu().numpy()

    # --- This animal's normal |v| (see world.step_agent) ---
    # `speed_gain` was picked for cruise mode where |v| is 1 by construction; a
    # trained net's typical |v| is well below that, so without this the animal
    # crawls.  Measuring it beats hard-coding one: it differs per model kind,
    # and the sample model's magnitude in particular depends on h/sigma.
    speed_ref = args.speed_ref
    if speed_ref is None:
        probe_pos = np.random.default_rng(0).uniform(
            -1.0, 1.0, size=(4096, 2)).astype(np.float32)
        probe_obs = world.encode_observations(probe_pos, objects)
        mags = []
        for w in (np.float32([1, 0, 0]), np.float32([0, 1, 0]), np.float32([0, 0, 1]),
                  np.float32([.5, .5, 0]), np.float32([0, .5, .5])):
            mags.append(np.linalg.norm(
                predict(probe_obs, F.blend_fields(noise_fields, w, world.CATEGORIES)),
                axis=1))
        # The 30th percentile, not the median.  speed_ref is a saturation point
        # -- the animal runs at full `speed_gain` wherever |v| >= speed_ref and
        # proportionally slower below it -- so calibrating to the median leaves
        # it below full pace half the time.  That slack is what lets a wandering
        # threat close on it: measured contact roughly doubles at the median
        # (0.82) versus 0.65 on the same networks.
        speed_ref = float(np.percentile(np.concatenate(mags), 30))
        print(f"\nMeasured speed_ref (30th pct |v| over the arena): {speed_ref:.4f}")
    elif speed_ref == 0:
        speed_ref = None

    if args.demo_mode == "fields":
        viz.field_sheet(noise_fields, args.hidden_dim, save_path=out("fields", "png"))
        return
    if args.demo_mode == "pure-panels":
        viz.pure_panels(predict, objects, noise_fields,
                        save_path=out("pure_panels", "png"))
        return

    if args.save:
        viz.loss_figure(history, save_path=out("loss", "png"))
        viz.pure_panels(predict, objects, noise_fields,
                        save_path=out("pure_panels", "png"))

    # ONE LoopParams for the animation -- the same object a headless
    # `world.rollout` would take, so what you watch is what gets measured.
    params = world.LoopParams(
        eat_radius=args.eat_radius, shelter_radius=args.shelter_radius,
        speed_gain=args.speed_gain,
        speed_ref=speed_ref, turn_rate=args.turn_rate,
        velocity_smoothing=args.velocity_smoothing,
        hunger_rate=args.hunger_rate,
        eat_amount=args.eat_amount, rest_frames=args.rest_frames,
        food_regrow_frames=args.food_regrow_frames,
        threat_gain=args.threat_gain, threat_range=args.threat_range,
        neuromod_smoothing=args.neuromod_smoothing,
        refuge_range=args.refuge_range, risk_hunger=args.risk_hunger,
        panic_range=args.panic_range,
        circadian_period=(args.circadian_period or world.CIRCADIAN_PERIOD),
        day_fraction=(args.day_fraction if args.day_fraction is not None
                      else world.DAY_FRACTION),
        threat_motion=args.threat_motion, threat_speed=args.threat_speed,
        threat_seed=args.seed)

    viz.animate(predict, objects, noise_fields, history, alphas,
                params=params, demo_mode=args.demo_mode,
                anim_frames=args.anim_frames,
                dynamic=args.dynamic_objects, velocities=velocities,
                show_reference=args.show_reference, target_gamma=args.target_gamma,
                hidden_dim=args.hidden_dim,
                save_path=out(args.demo_mode, "gif"), fps=args.fps)


if __name__ == "__main__":
    main()
