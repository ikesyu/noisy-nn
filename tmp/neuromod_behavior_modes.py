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

    .venv/bin/python tmp/neuromod_behavior_modes.py
    .venv/bin/python tmp/neuromod_behavior_modes.py --demo-mode pure-panels
    .venv/bin/python tmp/neuromod_behavior_modes.py --save          # write files
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("\n\n")[0],
    )
    # Model
    p.add_argument("--model", choices=("analytic", "sample"), default="analytic",
                   help="analytic = mean field (no SR barrier); sample = mechanism "
                        "with crossing threshold h>0 [analytic]")
    p.add_argument("--hidden-dim", type=int, default=64,
                   help="Hidden units per layer; must be a perfect square [64]")
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
    p.add_argument("--corner-centers", action="store_true",
                   help="Reproduce the original asymmetric right-triangle layout "
                        "(food-threat then share almost nothing)")
    # Task
    p.add_argument("--alpha-mix", type=float, default=None,
                   help="Off-drive weight as a fraction of the dominant one. "
                        "Default keeps the near-one-hot targets; ~0.3 makes the "
                        "three behaviours genuinely compete for the same inputs")
    p.add_argument("--target-gamma", type=float, default=2.0,
                   help="Speed saturation gain in the tanh target rule [2.0]")
    p.add_argument("--grid-side", type=int, default=31,
                   help="Training grid resolution per axis [31]")
    # Training
    p.add_argument("--epochs", type=int, default=3000, help="Training epochs [3000]")
    p.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate [3e-4]")
    p.add_argument("--train-chunk", type=int, default=0,
                   help="Minibatch size; 0 = full batch [0]")
    p.add_argument("--seed", type=int, default=7, help="Random seed [7]")
    # Scene
    p.add_argument("--layout", choices=("scripted", "random"), default="scripted")
    p.add_argument("--n-food", type=int, default=5)
    p.add_argument("--n-threat", type=int, default=3)
    p.add_argument("--n-shelter", type=int, default=2)
    p.add_argument("--dynamic-objects", action="store_true",
                   help="Slowly move objects during the animation (demo only)")
    p.add_argument("--object-speed", type=float, default=0.002)
    # Closed loop
    p.add_argument("--demo-mode",
                   choices=("scripted", "cycle", "pure-panels", "fields"),
                   default="scripted",
                   help="scripted = reactive closed loop; cycle = smooth blend "
                        "sweep; pure-panels / fields = static diagnostics")
    p.add_argument("--speed-mode", choices=("learned", "cruise"), default="learned",
                   help="learned keeps the network's output magnitude, so a "
                        "silent network freezes; cruise normalises it away "
                        "(original look) [learned]")
    p.add_argument("--speed-gain", type=float, default=0.9,
                   help="Agent speed scale [0.9]")
    p.add_argument("--anim-frames", type=int, default=360)
    p.add_argument("--eat-radius", type=float, default=0.10)
    p.add_argument("--shelter-radius", type=float, default=0.08)
    p.add_argument("--food-respawn", action="store_true")
    p.add_argument("--hunger-rate", type=float, default=0.006)
    p.add_argument("--need-rate", type=float, default=0.006)
    p.add_argument("--eat-amount", type=float, default=0.6)
    p.add_argument("--rest-frames", type=int, default=50)
    p.add_argument("--threat-gain", type=float, default=1.7)
    p.add_argument("--threat-range", type=float, default=0.40)
    p.add_argument("--neuromod-smoothing", type=float, default=0.12)
    p.add_argument("--threat-motion", choices=("moving", "static"), default="moving")
    p.add_argument("--threat-speed", type=float, default=0.01)
    p.add_argument("--velocity-smoothing", type=float, default=0.2)
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    def out(name, ext):
        return viz.resolve_out(f"{args.tag}_{name}.{ext}", args.out_dir) \
            if args.save else None

    # --- Scene ---
    objects = (world.make_scripted_objects() if args.layout == "scripted"
               else world.make_objects(rng, args.n_food, args.n_threat, args.n_shelter))
    velocities = world.make_object_velocities(rng, objects, args.object_speed)
    print(f"Layout '{args.layout}': "
          + ", ".join(f"{k}={objects[k].shape[0]}" for k in world.CATEGORIES))

    # --- Data ---
    alphas = world.alpha_states(args.alpha_mix)
    positions = world.make_training_grid(args.grid_side)
    obs_np = world.encode_observations(positions, objects)
    obs = torch.tensor(obs_np, dtype=torch.float32)
    targets = {
        s: torch.tensor(world.make_mixed_behavior_targets(
            obs_np, alphas[s], gamma=args.target_gamma), dtype=torch.float32)
        for s in world.STATES
    }

    # --- Noise fields ---
    centers = None
    if args.corner_centers:
        centers = {"food": (0.25, 0.75), "threat": (0.75, 0.25), "shelter": (0.25, 0.25)}
    noise_fields = F.build_fields(world.CATEGORIES, args.hidden_dim, args.base_std,
                                  args.sigma, args.theta, radius=args.field_radius,
                                  centers=centers)
    state_fields = {s: noise_fields[world.STATE_TO_FIELD[s]] for s in world.STATES}

    # --- Model ---
    n_hidden = args.hidden_layers
    net = P.build_network(args.hidden_dim, n_hidden=n_hidden, base_std=args.base_std,
                        kind=args.model, t=args.samples, crossing_h=args.crossing_h)
    print(f"\nModel : {args.model}  structure=[6, "
          + ", ".join([str(args.hidden_dim)] * n_hidden) + ", 2]")
    print(f"Train : epochs={args.epochs}  lr={args.lr}  "
          f"grid={args.grid_side}x{args.grid_side}  "
          f"alpha_mix={args.alpha_mix if args.alpha_mix is not None else 'one-hot'}")
    print("        same obs, same weights, different noise fields.\n")

    # --- Train ---
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
    def predict(obs_array, field):
        with torch.no_grad():
            return P.evaluate_vector_field(
                net, torch.as_tensor(obs_array, dtype=torch.float32), field,
                n_hidden).numpy()

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

    viz.animate(predict, objects, noise_fields, history, alphas,
                demo_mode=args.demo_mode, layout=args.layout,
                anim_frames=args.anim_frames, eat_radius=args.eat_radius,
                shelter_radius=args.shelter_radius, food_respawn=args.food_respawn,
                speed_mode=args.speed_mode, speed_gain=args.speed_gain,
                hunger_rate=args.hunger_rate, need_rate=args.need_rate,
                eat_amount=args.eat_amount, rest_frames=args.rest_frames,
                threat_gain=args.threat_gain, threat_range=args.threat_range,
                neuromod_smoothing=args.neuromod_smoothing,
                threat_motion=args.threat_motion, threat_speed=args.threat_speed,
                seed=args.seed, dynamic=args.dynamic_objects, velocities=velocities,
                show_reference=args.show_reference, target_gamma=args.target_gamma,
                velocity_smoothing=args.velocity_smoothing,
                hidden_dim=args.hidden_dim,
                save_path=out(args.demo_mode, "gif"), fps=args.fps)


if __name__ == "__main__":
    main()
