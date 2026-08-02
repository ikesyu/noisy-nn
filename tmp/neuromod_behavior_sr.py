"""neuromod_behavior_sr -- stochastic resonance in BEHAVIOUR, not in a regression.

The SR curve in `neuromod_sr_curve.py` plots task error, a machine-learning
quantity.  This script asks the version a neuroscientist reads: with one animal and
one set of weights, does the amount of food it actually gets peak at an intermediate
neuromodulator concentration?  Too little and the network falls below threshold and
the animal freezes; too much and the crossing saturates, the output stops depending
on the input, and the animal wanders.  That is Yerkes-Dodson drawn as behaviour.

Two things make this honest rather than decorative:

    speed_mode="learned"   the closed loop uses the network's output MAGNITUDE, so a
                           silent network produces a stationary animal.  Normalising
                           the heading away (the "cruise" mode) hides the low-noise
                           collapse, which is the left arm of the inverted U.
    --model sample         the mean field has no subthreshold barrier, so behavioural
                           SR cannot exist in it (`docs/idea_neuromod.md` section 6).
                           `--model analytic` is available as the predicted null.

Sweep semantics, and the reason the default is the acute one:

    --sweep test    ONE animal, trained at --base-std, then run at each concentration.
                    This is the acute manipulation the animation shows, and the same
                    logical move as the drug in section 7.2.  It CANNOT carry the SR
                    claim: an animal does worse away from the concentration it grew up
                    at whatever the model, so this sweep yields an inverted U by
                    construction, mean field included.  Use it for the animation.
    --sweep train   a fresh animal trained AT each concentration, then scored there.
                    Every point is an animal at home, so an interior optimum means a
                    genuinely best concentration to live at.  This is the claim, and
                    it costs one training per point.

    .venv/bin/python tmp/neuromod_behavior_sr.py --epochs 60 --grid-side 9 \
        --samples 16 --concentrations 5 --episodes 2 --frames 300   # plumbing, ~3 min
    .venv/bin/python tmp/neuromod_behavior_sr.py --save                # default
    .venv/bin/python tmp/neuromod_behavior_sr.py --sweep train --save  # rigorous curve
    .venv/bin/python tmp/neuromod_behavior_sr.py --model analytic --save \
        --tag behavior_sr_analytic                                     # the null

Output: tmp/out/<tag>_curve.png, tmp/out/<tag>_slider.gif, tmp/out/<tag>.csv
Context: `docs/idea_neuromod.md` section 7.1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = Path(__file__).resolve().parent
for _p in (str(ROOT), str(TMP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from neuromod import fields as F
from neuromod import protocol as P
from neuromod import viz, world

N_HIDDEN = 2

# Behavioural measures worth reporting.  `foods_per_1k` is the headline: it is what
# the animal achieves, it needs the whole loop to work, and it is legible without
# knowing anything about the model.
METRICS = ("foods_per_1k", "mean_speed", "shelter_frac", "path_len", "close_frac")


def build_data(args):
    objects = world.make_scripted_objects()
    alphas = world.alpha_states(args.alpha_mix)
    obs_np = world.encode_observations(
        world.make_training_grid(args.grid_side), objects)
    obs = torch.tensor(obs_np, dtype=torch.float32)
    targets = {s: torch.tensor(
                   world.make_mixed_behavior_targets(obs_np, alphas[s],
                                                     gamma=args.target_gamma),
                   dtype=torch.float32)
               for s in world.STATES}
    return objects, obs, targets


def make_fields(peak_std: float, args):
    return F.build_fields(world.CATEGORIES, args.hidden_dim, float(peak_std),
                          args.sigma, args.theta, radius=args.field_radius)


def save_net(net, path: Path, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "model": args.model,
                "hidden_dim": args.hidden_dim, "samples": args.samples,
                "crossing_h": args.crossing_h, "base_std": args.base_std,
                "epochs": args.epochs, "grid_side": args.grid_side}, path)
    print(f"saved {path}")


def load_net(path: Path, args):
    """Reload a trained animal so the animation can be retuned without retraining."""
    blob = torch.load(path, weights_only=False)
    for key in ("model", "hidden_dim", "crossing_h"):
        if blob.get(key) != getattr(args, key if key != "model" else "model"):
            print(f"  WARNING: checkpoint {key}={blob.get(key)} differs from "
                  f"the current setting; using the checkpoint's network anyway")
    net = P.build_network(blob["hidden_dim"], n_hidden=N_HIDDEN,
                          base_std=blob["base_std"], kind=blob["model"],
                          t=blob["samples"], crossing_h=blob["crossing_h"])
    net.load_state_dict(blob["state_dict"])
    print(f"loaded {path}  (model={blob['model']}, epochs={blob['epochs']})")
    return net, make_fields(blob["base_std"], args)


def train_at(peak_std: float, seed: int, obs, targets, args):
    torch.manual_seed(seed)
    np.random.seed(seed)
    fields = make_fields(peak_std, args)
    state_fields = {s: fields[world.STATE_TO_FIELD[s]] for s in world.STATES}
    net = P.build_network(args.hidden_dim, n_hidden=N_HIDDEN, base_std=float(peak_std),
                          kind=args.model, t=args.samples, crossing_h=args.crossing_h)
    P.train(net, obs, targets, state_fields, world.STATES, N_HIDDEN,
            args.epochs, args.lr, verbose=False)
    return net, fields


def make_predict(net):
    def predict(obs_array, field):
        with torch.no_grad():
            return P.evaluate_vector_field(
                net, torch.as_tensor(obs_array, dtype=torch.float32), field,
                N_HIDDEN).numpy()
    return predict


def loop_params(args, threat_seed: int) -> world.LoopParams:
    return world.LoopParams(
        eat_radius=args.eat_radius, shelter_radius=args.shelter_radius,
        food_respawn=True, speed_mode=args.speed_mode, speed_gain=args.speed_gain,
        velocity_smoothing=args.velocity_smoothing, dt=args.dt,
        hunger_rate=args.hunger_rate, need_rate=args.need_rate,
        eat_amount=args.eat_amount, rest_frames=args.rest_frames,
        threat_gain=args.threat_gain, threat_range=args.threat_range,
        neuromod_smoothing=args.neuromod_smoothing,
        threat_motion=args.threat_motion, threat_speed=args.threat_speed,
        threat_seed=threat_seed, speed_ref=getattr(args, 'speed_ref', None))


def measure(predict, fields, objects, concentration: float, args) -> list[dict]:
    """Several episodes at one concentration; episodes differ in threat wandering."""
    out = []
    for e in range(args.episodes):
        params = loop_params(args, threat_seed=args.seed0 + e)
        m = world.rollout(predict, fields, params, n_frames=args.frames,
                          concentration=concentration, objects=objects,
                          seed=args.seed0 + e)
        m["episode"] = e
        out.append(m)
    return out


def sweep(objects, obs, targets, concentrations, args):
    """Behavioural measures over concentration, by whichever sweep was asked for."""
    rows = []
    net_for_anim, fields_for_anim = None, None

    if args.sweep == "test":
        if args.load_net:
            net, fields = load_net(Path(args.load_net), args)
        else:
            net, fields = train_at(args.base_std, args.seed0, obs, targets, args)
            if args.net_path:
                save_net(net, Path(args.net_path), args)
        predict = make_predict(net)
        net_for_anim, fields_for_anim = net, fields
        for c in concentrations:
            for m in measure(predict, fields, objects, c, args):
                rows.append({"concentration": float(c), "seed": args.seed0, **m})
            print(f"  concentration={c:6.3f}  "
                  + "  ".join(f"{k}={np.mean([r[k] for r in rows if np.isclose(r['concentration'], c)]):.3f}"
                              for k in ("foods_per_1k", "mean_speed")), flush=True)
    else:
        for c in concentrations:
            for k in range(args.seeds):
                seed = args.seed0 + k
                net, fields = train_at(args.base_std * c, seed, obs, targets, args)
                predict = make_predict(net)
                # Each animal is scored at the concentration it grew up at, so the
                # field is already at that level: probe at scale 1.
                for m in measure(predict, fields, objects, 1.0, args):
                    rows.append({"concentration": float(c), "seed": seed, **m})
                if net_for_anim is None or np.isclose(c, 1.0):
                    net_for_anim, fields_for_anim = net, make_fields(args.base_std, args)
            print(f"  concentration={c:6.3f}  "
                  + "  ".join(f"{k}={np.mean([r[k] for r in rows if np.isclose(r['concentration'], c)]):.3f}"
                              for k in ("foods_per_1k", "mean_speed")), flush=True)
    return rows, net_for_anim, fields_for_anim


def aggregate(rows, concentrations):
    out = []
    for c in concentrations:
        cell = [r for r in rows if np.isclose(r["concentration"], c)]
        agg = {"concentration": float(c), "n": len(cell)}
        for key in METRICS:
            values = np.array([r[key] for r in cell], dtype=float)
            agg[key] = float(values.mean())
            agg[key + "_std"] = float(values.std())
        out.append(agg)
    return out


def summarise(agg, args) -> None:
    x = np.array([a["concentration"] for a in agg])
    y = np.array([a[args.metric] for a in agg])
    e = np.array([a[args.metric + "_std"] for a in agg])
    best = int(np.argmax(y))
    interior = 0 < best < len(y) - 1

    # An argmax on a noisy curve lands in the interior most of the time, so the peak
    # has to clear both ends by more than the episode-to-episode spread before it
    # counts.  The quantity that actually separates the mechanism from the mean field
    # is the LOW-end collapse: without a subthreshold barrier there is nothing to
    # collapse, and the animal keeps foraging however little noise it has.
    margin = max(float(e[best]), float(e[0]), float(e[-1]))
    clears = y[best] - max(y[0], y[-1]) > margin
    lo_ratio = y[0] / y[best] if y[best] > 0 else float("nan")
    hi_ratio = y[-1] / y[best] if y[best] > 0 else float("nan")

    print(f"\n{args.metric} over concentration ({args.sweep} sweep, "
          f"model={args.model}, speed_mode={args.speed_mode}):")
    print(f"  peak {y[best]:.3f} +-{e[best]:.3f} at concentration {x[best]:.3f}"
          + (f"  (h/sigma = {args.crossing_h / (args.base_std * x[best]):.3f})"
             if args.model != "analytic" else ""))
    print(f"  ends {y[0]:.3f} +-{e[0]:.3f} / {y[-1]:.3f} +-{e[-1]:.3f}"
          f"   (low end = {lo_ratio:.2f} of peak, high end = {hi_ratio:.2f})")
    if not (interior and clears):
        print("  No interior optimum beyond the episode spread.")
        if args.sweep == "train" and args.model == "analytic":
            print("  This is the expected null: the mean field has no subthreshold "
                  "barrier, so low concentration costs it nothing and the curve "
                  "stays flat. Compare the low-end ratio against the sample run.")
        else:
            print("  Check that --speed-mode is 'learned', raise --episodes to "
                  "shrink the spread, and widen the concentration range.")
    elif args.sweep == "test":
        print("  Interior optimum, but this sweep CANNOT support an SR claim.")
        print("  An animal trained at one concentration and run at another does "
              "worse away from home for ANY model, mean field included, so the "
              "test sweep produces an inverted U by construction (the confound "
              "documented in docs/idea_neuromod.md section 6). This curve is the "
              "acute manipulation the animation illustrates; run --sweep train "
              "with --model sample, against --model analytic, for the claim.")
    else:
        print("  Interior optimum on the TRAINING-level sweep, i.e. not the "
              "away-from-home confound: every animal here grew up at its own "
              "concentration.")
        if args.seeds == 1:
            print("  BUT one network per concentration cannot establish this. The "
                  "spread above is episode-to-episode within a single animal, so it "
                  "does not cover the variation between animals. Re-run with "
                  "--seeds 3 before calling it a result.")
        print("  The claim also needs the cross-model contrast: compare the low- and "
              "high-end ratios against --model analytic. A mean field with no "
              "subthreshold barrier should stay flatter, and if it does not, the "
              "behavioural readout is simply too coarse to see the barrier.")
        if args.model == "analytic":
            print("  NOTE: this is the mean field, which has no subthreshold "
                  "barrier. An interior optimum here would undercut the contrast, "
                  "so compare its depth against the sample run.")
    speeds = np.array([a["mean_speed"] for a in agg])
    print(f"  mean speed runs {speeds[0]:.3f} -> {speeds[best]:.3f} -> "
          f"{speeds[-1]:.3f} (the low end is the freezing that carries the "
          f"left arm)")


def plot_curve(agg, args, save_path=None):
    viz.use_headless(save_path)
    import matplotlib.pyplot as plt
    plt.rcParams.update(viz.FONT)

    x = np.array([a["concentration"] for a in agg])
    fig, (ax_m, ax_s) = plt.subplots(2, 1, figsize=(8.5, 8.0), sharex=True)

    y = np.array([a[args.metric] for a in agg])
    e = np.array([a[args.metric + "_std"] for a in agg])
    ax_m.plot(x, y, "-o", color="0.2", lw=1.8)
    ax_m.fill_between(x, y - e, y + e, color="0.2", alpha=0.18)
    best = int(np.argmax(y))
    ax_m.axvline(x[best], color="#c23b3b", lw=1.4, ls=":")
    ax_m.set_ylabel(args.metric.replace("_", " "))
    ax_m.set_xscale("log")
    ax_m.grid(alpha=0.3)

    for key, colour in (("mean_speed", "tab:blue"), ("shelter_frac", "tab:green")):
        yy = np.array([a[key] for a in agg])
        ee = np.array([a[key + "_std"] for a in agg])
        ax_s.plot(x, yy, "-o", lw=1.6, color=colour, label=key.replace("_", " "))
        ax_s.fill_between(x, yy - ee, yy + ee, color=colour, alpha=0.18)
    ax_s.axvline(x[best], color="#c23b3b", lw=1.4, ls=":")
    ax_s.set_xlabel("neuromodulator concentration  (field scale)")
    ax_s.set_ylabel("supporting measures")
    ax_s.set_xscale("log")
    ax_s.grid(alpha=0.3)
    ax_s.legend()

    fig.tight_layout()
    return viz.save_or_show(fig, save_path)


def make_schedule(concentrations, args):
    """Open at the optimum, drop to starvation, recover, overshoot, recover.

    Starting at the working concentration matters for reading the demo: the viewer
    needs to see what competent behaviour looks like BEFORE the two failures, or the
    freezing and the flailing have nothing to be worse than.  Each return to the
    optimum is held long enough to eat again.
    """
    lo, hi = float(concentrations[0]), float(concentrations[-1])
    mid = float(np.sqrt(lo * hi)) if args.slider_mid is None else float(args.slider_mid)
    hold, ramp, dwell = args.slider_hold, args.slider_ramp, args.slider_dwell
    return np.concatenate([
        np.full(dwell, mid),               # this is the animal working
        np.geomspace(mid, lo, ramp),       # starve it of noise
        np.full(hold, lo),                 # frozen
        np.geomspace(lo, mid, ramp),
        np.full(dwell, mid),               # working again
        np.geomspace(mid, hi, ramp),       # drown it in noise
        np.full(hold, hi),                 # flailing
        np.geomspace(hi, mid, ramp),
        np.full(dwell, mid),               # working again
    ])


def write_csv(path: Path, rows, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w") as f:
        f.write(f"# neuromod_behavior_sr  sweep={args.sweep} model={args.model} "
                f"speed_mode={args.speed_mode} base_std={args.base_std} "
                f"frames={args.frames} episodes={args.episodes} epochs={args.epochs}\n")
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(f"{row[k]}" for k in keys) + "\n")
    print(f"saved {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("\n\n")[0])
    p.add_argument("--sweep", choices=("test", "train"), default="test",
                   help="test = one animal dosed acutely (what the animation shows); "
                        "train = a fresh animal per concentration (rigorous) [test]")
    p.add_argument("--model", choices=("sample", "analytic"), default="sample",
                   help="sample is the mechanism; analytic is the predicted null "
                        "(no subthreshold barrier) [sample]")
    p.add_argument("--speed-mode", choices=("learned", "cruise"), default="learned",
                   help="learned keeps the output magnitude, so silence freezes the "
                        "animal. 'cruise' hides the low-noise arm [learned]")
    p.add_argument("--metric", default="foods_per_1k",
                   help="Headline behavioural measure [foods_per_1k]")
    # Concentration axis
    p.add_argument("--c-min", type=float, default=0.15)
    p.add_argument("--c-max", type=float, default=6.0)
    p.add_argument("--concentrations", type=int, default=11)
    # Episodes
    p.add_argument("--frames", type=int, default=1500,
                   help="Closed-loop frames per episode [1500]")
    p.add_argument("--episodes", type=int, default=4,
                   help="Episodes per concentration (threat wandering differs) [4]")
    p.add_argument("--seeds", type=int, default=1,
                   help="Networks per concentration, train sweep only [1]")
    p.add_argument("--seed0", type=int, default=0)
    # Model / training
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=2500,
                   help="Epochs per network. The analytic model needs "
                        "~2500 to drive task MSE to ~0; below that the "
                        "animal cannot forage and the demo looks broken "
                        "for reasons unrelated to noise [2500]")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grid-side", type=int, default=15)
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--base-std", type=float, default=1.0,
                   help="Peak recruited-unit std the animal is trained at [1.0]")
    p.add_argument("--sigma", type=float, default=0.22)
    p.add_argument("--theta", type=float, default=0.15)
    p.add_argument("--field-radius", type=float, default=0.28)
    p.add_argument("--alpha-mix", type=float, default=None)
    p.add_argument("--target-gamma", type=float, default=2.0)
    # Closed loop
    p.add_argument("--speed-gain", type=float, default=2.7,
                   help="Agent speed scale. Larger than the cruise demo's 0.9 on "
                        "purpose: in learned mode the step is proportional to |v|, "
                        "and in the closed loop |v| sits well below its saturated "
                        "value because the agent spends its time near objects, "
                        "where the learned deceleration law applies. At 0.9 the "
                        "animal covers a third of the ground and forages too "
                        "rarely to watch [2.7]")
    p.add_argument("--speed-ref", type=float, default=None,
                   help="Override the measured normal |v| used to scale "
                        "learned-mode speed; None measures it")
    p.add_argument("--velocity-smoothing", type=float, default=0.2)
    p.add_argument("--dt", type=float, default=0.04)
    p.add_argument("--eat-radius", type=float, default=0.10)
    p.add_argument("--shelter-radius", type=float, default=0.08)
    p.add_argument("--hunger-rate", type=float, default=0.006)
    p.add_argument("--need-rate", type=float, default=0.006)
    p.add_argument("--eat-amount", type=float, default=0.6)
    p.add_argument("--rest-frames", type=int, default=50)
    p.add_argument("--threat-gain", type=float, default=1.7)
    p.add_argument("--threat-range", type=float, default=0.40)
    p.add_argument("--neuromod-smoothing", type=float, default=0.12)
    p.add_argument("--threat-motion", choices=("moving", "static"), default="moving")
    p.add_argument("--threat-speed", type=float, default=0.01)
    # Animation
    p.add_argument("--no-anim", action="store_true", help="Curve only, no slider GIF")
    p.add_argument("--slider-hold", type=int, default=90,
                   help="Frames held at each extreme [90]")
    p.add_argument("--slider-dwell", type=int, default=260,
                   help="Frames held at the optimum, long enough to "
                        "actually forage [260]")
    p.add_argument("--slider-ramp", type=int, default=120)
    p.add_argument("--slider-mid", type=float, default=None,
                   help="Concentration to dwell at; default is the measured peak")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--stride", type=int, default=3,
                   help="Dynamics steps per drawn frame; the episode is "
                        "unchanged, the file just gets smaller [3]")
    # Output
    p.add_argument("--save", action="store_true")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tag", default="behavior_sr")
    p.add_argument("--net-path", default=None,
                   help="Save the trained network here (test sweep). Training is the "
                        "expensive step; keeping it lets the animation be retuned "
                        "for free")
    p.add_argument("--load-net", default=None,
                   help="Reuse a network saved by --net-path instead of training")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    concentrations = np.geomspace(args.c_min, args.c_max, args.concentrations)
    objects, obs, targets = build_data(args)

    n_nets = 1 if args.sweep == "test" else args.concentrations * args.seeds
    print(f"sweep={args.sweep}  model={args.model}  speed_mode={args.speed_mode}  "
          f"concentrations={args.concentrations} in "
          f"[{args.c_min}, {args.c_max}]  episodes={args.episodes} x {args.frames} "
          f"frames  ({n_nets} networks to train)")

    rows, net, fields = sweep(objects, obs, targets, concentrations, args)
    agg = aggregate(rows, concentrations)
    summarise(agg, args)

    def out(name, ext):
        return viz.resolve_out(f"{args.tag}_{name}.{ext}", args.out_dir) \
            if args.save else None

    if args.save:
        write_csv(viz.resolve_out(f"{args.tag}.csv", args.out_dir), rows, args)
    plot_curve(agg, args, save_path=out("curve", "png"))

    if args.no_anim:
        return
    x = np.array([a["concentration"] for a in agg])
    y = np.array([a[args.metric] for a in agg])
    e = np.array([a[args.metric + "_std"] for a in agg])
    if args.slider_mid is None:
        args.slider_mid = float(x[int(np.argmax(y))])
    schedule = make_schedule(concentrations, args)
    viz.concentration_slider(
        make_predict(net), objects, fields, loop_params(args, args.seed0),
        schedule, (x, y, e), metric_label=args.metric.replace("_", " "),
        hidden_dim=args.hidden_dim, save_path=out("slider", "gif"),
        fps=args.fps, stride=args.stride)


if __name__ == "__main__":
    main()
