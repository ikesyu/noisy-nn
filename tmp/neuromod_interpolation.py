"""neuromod_interpolation -- does an interpolated field produce an interpolated behaviour?

This is the L4 experiment (`docs/idea_neuromod.md` sections 1.1 and 8(c)), the last
gap in the neuromodulation line.  The network is trained on THREE discrete states
only (`protocol.train` ever sees the three pure fields), and then asked, at
inference time, what it does under fields it has never seen.

The two competing readings of the same demo:

    SWITCH        the field is a discrete selector.  An intermediate field selects
                  one of the trained behaviours, so the output snaps: it stays on
                  behaviour A, jumps once, and stays on behaviour B.
    CONTINUOUS    the field is a control axis.  An intermediate field produces the
                  corresponding intermediate behaviour, which was never trained.

For a pair of categories (a, b) and a mixing coefficient lambda:

    field    F(lam) = (1 - lam) F_a + lam F_b        (given to the network)
    target   t(lam) from the drive weights (1 - lam) alpha_a + lam alpha_b
                                                     (never shown to the network)

t(lam) is NOT the linear interpolation of t(0) and t(1): the target map bounds the
velocity through tanh, so hitting t(lam) is a real prediction, not an average of two
things the network already knows.

Metrics, and what each hypothesis predicts:

    M1 err(lam)     MSE(y(lam), t(lam)).  CONTINUOUS: stays near the endpoint
                    (trained) error.  SWITCH: rises in the interior.
    M2 lam_hat(lam) argmin_mu MSE(y(lam), t(mu)), the mixture the output actually
                    implies.  CONTINUOUS: monotone and continuous.  SWITCH: a
                    staircase taking essentially two values.
    M3 smoothness   largest single step of y along lam, as a fraction of the total
                    path.  SWITCH: near 1.  CONTINUOUS: near 1 / n_lambda.

M2 is the headline because it does not require the correspondence to be LINEAR: a
monotone but warped dial still satisfies L4, and lam_hat measures exactly that.

The verdict rule is fixed by the flag defaults and is not to be tuned after seeing
the curves (`docs/idea_neuromod.md` section 7.9.1 warns against picking the metric
that gives the wanted answer).

Why `--model sample` is the default here, unlike the other drivers: in the
mean-field model the field enters smoothly through sigma, so continuity is nearly
true by construction and the experiment would be uninformative.  At sample level
the field decides WHICH units cross the band, and blending two fields could just as
well superpose two conflicting behaviours as produce a graded one.  analytic is
kept only as a fast wiring check.

    .venv/bin/python tmp/neuromod_interpolation.py --model analytic --epochs 200
                                                                  # wiring check
    .venv/bin/python tmp/neuromod_interpolation.py --seeds 3 --alpha-mix 0.6 --save
                                                                  # the real run

Output: tmp/out/<tag>.png, tmp/out/<tag>.csv, and the trained nets at --net-path.
Context: `docs/idea_neuromod.md` section 8(c).
"""
from __future__ import annotations

import argparse
import itertools
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

# The recorded results of this script were produced with the legacy 6D vector
# sensing; the module default is now the sector code (the standard benchmark),
# so pin the old encoding explicitly.
world.set_sensing("vector")

FIELD_TO_STATE = {v: k for k, v in world.STATE_TO_FIELD.items()}


# ============================================================
# Data, training, evaluation
# ============================================================

def build_data(args, mix):
    """Scripted scene -> observations, drive weights, and the three pure targets.

    `mix` raises the off-drives.  Under the default near-one-hot weights the three
    behaviours read nearly disjoint slices of the 6D input, so an interpolated
    target decomposes almost trivially and L4 becomes cheap to satisfy; 0.6 makes
    the drives genuinely compete (`docs/idea_neuromod.md` sections 4 and 7.3).
    """
    objects = world.make_scripted_objects()
    alphas = world.alpha_states(mix)
    obs_np = world.encode_observations(
        world.make_training_grid(args.grid_side), objects)
    obs = torch.tensor(obs_np, dtype=torch.float32)
    targets = {s: torch.tensor(
                   world.make_mixed_behavior_targets(obs_np, alphas[s],
                                                     gamma=args.target_gamma),
                   dtype=torch.float32)
               for s in world.STATES}
    return obs_np, obs, alphas, targets


def make_net(args):
    return P.build_network(args.hidden_dim, n_hidden=args.hidden_layers,
                           base_std=args.base_std, kind=args.model,
                           t=args.samples, crossing_h=args.crossing_h)


def build_fields(args):
    return F.build_fields(world.CATEGORIES, args.hidden_dim, args.base_std,
                          args.sigma, args.theta, radius=args.field_radius)


def train_one(seed: int, obs, targets, args):
    """One trained network.  Training only ever sees the three PURE fields."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    flds = build_fields(args)
    state_fields = {s: flds[world.STATE_TO_FIELD[s]] for s in world.STATES}
    net = make_net(args)
    P.train(net, obs, targets, state_fields, world.STATES, args.hidden_layers,
            args.epochs, args.lr, verbose=args.verbose)
    return net, flds


def get_net(seed: int, obs, targets, args, cache: dict):
    """Train, or restore from the checkpoint cache.  Sample training is ~1 s/epoch."""
    key = f"seed{seed}"
    flds = build_fields(args)
    if key in cache:
        net = make_net(args)
        net.load_state_dict(cache[key])
        print(f"  seed {seed}: loaded from checkpoint", flush=True)
        return net, flds
    net, flds = train_one(seed, obs, targets, args)
    cache[key] = net.state_dict()
    return net, flds


def set_eval_samples(net, t: int):
    """Raise T for evaluation.  Cheap: only TRAINING is sample-count bound."""
    if hasattr(net, "sampled_layer"):
        net.sampled_layer.numT = t


def predict(net, obs, field, args) -> np.ndarray:
    with torch.no_grad():
        return P.evaluate_vector_field(net, obs, field, args.hidden_layers).cpu().numpy()


def mse(y: np.ndarray, t: np.ndarray) -> float:
    return float(np.mean((y - t) ** 2))


def rms_distance(y: np.ndarray, z: np.ndarray) -> float:
    """Per-point RMS distance between two output fields."""
    return float(np.sqrt(np.mean(np.sum((y - z) ** 2, axis=1))))


# ============================================================
# One pair, one seed: the lambda sweep
# ============================================================

def blend_weights(a: str, b: str, lam: float) -> np.ndarray:
    w = np.zeros(len(world.CATEGORIES), dtype=np.float32)
    w[world.CATEGORIES.index(a)] = 1.0 - lam
    w[world.CATEGORIES.index(b)] = lam
    return w


def blend_field(flds, a: str, b: str, lam: float, rule: str) -> torch.Tensor:
    """The interpolated field.  The rule is not a free parameter, it is the question.

        linear    sigma(lam) = (1 - lam) sigma_a + lam sigma_b.  The obvious reading
                  of "interpolate the field", but it LOWERS sigma on every unit that
                  belongs to only one of the two supports, so at lam = 0.5 the whole
                  network runs at a different h/sigma than it was trained at.  That
                  is not a gauge transformation, it is a real perturbation
                  (`docs/idea_core.md` section 4.3), and it confounds "is the field a
                  continuous control axis" with "does the network survive being
                  pushed off its operating point".
        variance  sigma(lam)^2 = (1 - lam) sigma_a^2 + lam sigma_b^2.  Interpolates
                  the noise POWER, which is what the crossing statistics see, and
                  keeps the intermediate closer to the trained operating point.

    Both are reported; `linear` is the pre-registered one.
    """
    if rule == "linear":
        return F.blend_fields(flds, blend_weights(a, b, lam), world.CATEGORIES)
    if rule == "variance":
        power = ((1.0 - lam) * flds[a] ** 2 + lam * flds[b] ** 2)
        return torch.sqrt(power)
    raise ValueError(f"unknown blend rule {rule!r}")


def target_at(obs_np, alphas, a: str, b: str, lam: float, gamma: float) -> np.ndarray:
    alpha = ((1.0 - lam) * alphas[FIELD_TO_STATE[a]]
             + lam * alphas[FIELD_TO_STATE[b]])
    return world.make_mixed_behavior_targets(obs_np, alpha, gamma=gamma)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if np.allclose(y, y[0]):
        return 0.0
    return float(spearmanr(x, y).statistic)


def sweep_pair(net, flds, obs, obs_np, alphas, args, a: str, b: str) -> dict:
    """The lambda sweep for one pair, with every metric this experiment reports."""
    lams = np.linspace(0.0, 1.0, args.n_lambda)
    mus = np.linspace(0.0, 1.0, args.n_mu)
    t_mu = np.stack([target_at(obs_np, alphas, a, b, mu, args.target_gamma)
                     for mu in mus])

    ys, ys2, nus = [], [], []
    for lam in lams:
        field = blend_field(flds, a, b, lam, args.blend)
        ys.append(predict(net, obs, field, args))
        # Participation under the blended field.  If this dips in the interior, the
        # network is being evaluated off the operating point it was trained at, and
        # any interior error is partly that rather than a failure to interpolate.
        nus.append(float(np.mean(
            F.crossing_rates(net, obs, field, args.hidden_layers)[0])))
        # A second, independent draw at the SAME lambda.  Without it the smoothness
        # metric M3 is not interpretable: pure sampling jitter also produces many
        # small equal steps and would pass the test for the wrong reason.
        ys2.append(predict(net, obs, field, args))
    ys = np.stack(ys)

    # Against the exact interpolated target, not the nearest point of the mu grid.
    err_interp = np.array([mse(ys[i], target_at(obs_np, alphas, a, b, lam,
                                                args.target_gamma))
                           for i, lam in enumerate(lams)])
    err_a = np.array([mse(y, t_mu[0]) for y in ys])
    err_b = np.array([mse(y, t_mu[-1]) for y in ys])
    err_switch = np.minimum(err_a, err_b)

    grid_err = np.array([[mse(y, t) for t in t_mu] for y in ys])
    lam_hat = mus[np.argmin(grid_err, axis=1)]
    # Post-hoc diagnostics, added after M1 failed on the first sample run.  They
    # do not enter the verdict; they separate three ways M1 can fail.
    #   err_best_mu  the error at the mixture the output actually implies.  If this
    #                is also high, the output is the wrong SHAPE, not just at the
    #                wrong point of the axis.
    #   err_blend    what a network that merely mixed its two trained outputs would
    #                score.  t(lam) is tanh-bounded and therefore NOT the linear
    #                interpolation of t(0), t(1), so this reference is not zero: it
    #                is the accuracy ceiling of pure mixing.
    err_best_mu = grid_err.min(axis=1)
    err_blend = np.array([
        mse((1.0 - lam) * ys[0] + lam * ys[-1],
            target_at(obs_np, alphas, a, b, lam, args.target_gamma))
        for lam in lams])

    steps = np.array([rms_distance(ys[i + 1], ys[i]) for i in range(len(lams) - 1)])
    total = float(steps.sum())
    jitter = float(np.mean([rms_distance(y, y2) for y, y2 in zip(ys, ys2)]))
    # Endpoint distance is the jitter-robust scale: it is large, so the noise that
    # inflates every consecutive step adds negligibly in quadrature here.  Consecutive
    # steps cannot be used for this -- at low T they are almost entirely jitter, which
    # would make the dial look FINER the noisier it gets.
    endpoint_dist = rms_distance(ys[-1], ys[0])

    return {
        "lams": lams, "err_interp": err_interp, "err_a": err_a, "err_b": err_b,
        "err_switch": err_switch, "lam_hat": lam_hat,
        "err_best_mu": err_best_mu, "err_blend": err_blend,
        "nu_mean": np.array(nus),
        "spearman": spearman(lams, lam_hat),
        "max_step_frac": float(steps.max() / total) if total > 0 else 1.0,
        "path_ratio": (total / rms_distance(ys[-1], ys[0])
                       if rms_distance(ys[-1], ys[0]) > 0 else np.inf),
        "step_over_jitter": (float(steps.mean() / jitter) if jitter > 0 else np.inf),
        "jitter": jitter,
        "resolution": (float(jitter / endpoint_dist) if endpoint_dist > 0 else np.inf),
        "interior_gain": float(np.mean(
            err_interp[1:-1] < err_switch[1:-1])) if len(lams) > 2 else 0.0,
    }


# ============================================================
# Aggregation and the verdict
# ============================================================

SCALARS = ("spearman", "max_step_frac", "path_ratio", "step_over_jitter",
           "jitter", "interior_gain", "resolution")
CURVES = ("err_interp", "err_a", "err_b", "err_switch", "lam_hat",
          "err_best_mu", "err_blend", "nu_mean")


def aggregate(per_seed: list[dict]) -> dict:
    """Mean (and std) over seeds of every curve and scalar."""
    out = {"lams": per_seed[0]["lams"], "n_seeds": len(per_seed)}
    for key in CURVES:
        stack = np.stack([r[key] for r in per_seed])
        out[key] = stack.mean(axis=0)
        out[key + "_std"] = stack.std(axis=0)
    for key in SCALARS:
        values = np.array([r[key] for r in per_seed], dtype=float)
        finite = values[np.isfinite(values)]
        out[key] = float(values.mean())
        out[key + "_std"] = float(finite.std()) if finite.size else 0.0
    return out


def jump_threshold(args) -> float:
    """M3's threshold, as a multiple of the step a uniform traversal would take.

    The largest-step fraction is not comparable across sweep resolutions: a
    perfectly uniform traversal already gives 1 / (n_lambda - 1), so a fixed
    absolute threshold silently gets stricter as the sweep gets coarser.  The
    registered criterion is `--max-jump-k` = 3 uniform steps, which at the
    registered n_lambda = 21 is exactly the 0.15 fixed in the plan.
    """
    return args.max_jump_k / max(1, args.n_lambda - 1)


def verdict(agg: dict, args) -> tuple[bool, str]:
    """The pre-registered rule.  Do not tune these after seeing the curves."""
    err = agg["err_interp"]
    endpoint = float(max(err[0], err[-1]))
    c1 = bool(np.all(err <= args.err_factor * endpoint))
    c2 = bool(agg["spearman"] >= args.min_spearman)
    c3 = bool(agg["max_step_frac"] <= jump_threshold(args))

    worst = float(err.max())
    lines = [
        f"    M1 interior error   {'PASS' if c1 else 'FAIL'}  "
        f"worst {worst:.4f} vs {args.err_factor:g}x endpoint "
        f"{args.err_factor * endpoint:.4f}",
        f"    M2 implied mixture  {'PASS' if c2 else 'FAIL'}  "
        f"Spearman {agg['spearman']:.3f} (>= {args.min_spearman:g}), "
        f"{len(np.unique(np.round(agg['lam_hat'], 3)))} distinct values",
        f"    M3 smoothness       {'PASS' if c3 else 'FAIL'}  "
        f"largest step {agg['max_step_frac']:.3f} of the path "
        f"(<= {jump_threshold(args):.3f} = {args.max_jump_k:g} uniform steps; "
        f"a switch gives ~1.0)",
    ]
    if agg["step_over_jitter"] < args.min_step_over_jitter:
        lines.append(
            f"    !! M3 UNINFORMATIVE: mean step is only "
            f"{agg['step_over_jitter']:.2f}x the sampling jitter. Raise "
            f"--eval-samples until this clears {args.min_step_over_jitter:g}.")
        c3 = False
    interior = slice(1, -1)
    lines.append(
        f"    diagnostic: the interpolated target beats the nearer endpoint target "
        f"at {100 * agg['interior_gain']:.0f}% of interior lambdas; "
        f"path/endpoint distance {agg['path_ratio']:.2f}")
    lines.append(
        f"    post-hoc:   interior error {err[interior].max():.4f} worst, "
        f"{err[interior].mean():.4f} mean; at the implied mixture "
        f"{agg['err_best_mu'][interior].mean():.4f}; a pure mixer would score "
        f"{agg['err_blend'][interior].mean():.4f}; endpoint {endpoint:.4f}")
    # How finely the dial can be read at THIS T: the output travels `endpoint_dist`
    # across the whole axis while one evaluation is uncertain by `jitter`, so
    # lambda steps below jitter / endpoint_dist are not resolvable.  This is the
    # operationally meaningful number -- raising T is a MEASUREMENT choice, but an
    # agent that acts on one forward pass lives at its training T.
    resolution = agg["resolution"]
    if np.isfinite(resolution) and resolution > 0:
        lines.append(
            f"    dial resolution at T={args.eval_samples}: delta-lambda below "
            f"{resolution:.3f} is lost in the single-pass noise, i.e. about "
            f"{max(1, int(round(1.0 / resolution)))} distinguishable settings "
            f"across the axis")
    nu = agg["nu_mean"]
    lines.append(
        f"    operating point: mean crossing rate nu = {nu[0]:.4f} / "
        f"{nu[interior].min():.4f} (interior min) / {nu[-1]:.4f}, i.e. the interior "
        f"runs at {100 * nu[interior].min() / max(nu[0], nu[-1]):.0f}% of the "
        f"endpoint participation under the '{args.blend}' blend")

    ok = c1 and c2 and c3
    head = ("CONTINUOUS: the untrained intermediate fields produce the "
            "corresponding intermediate behaviours." if ok else
            "NOT ESTABLISHED: at least one pre-registered criterion failed "
            "(see below).")
    return ok, head + "\n" + "\n".join(lines)


# ============================================================
# Figure and CSV
# ============================================================

def plot_interpolation(results: dict, args, save_path=None):
    """Two rows: the recovery curves (M1) and the implied mixture (M2)."""
    viz.use_headless(save_path)
    import matplotlib.pyplot as plt
    plt.rcParams.update(viz.FONT)

    pairs = list(results.keys())
    fig, axes = plt.subplots(2, len(pairs), figsize=(4.7 * len(pairs), 7.6))
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:
        axes = axes.T

    for col, pair in enumerate(pairs):
        a, b = pair.split("|")
        agg = results[pair]
        lams = agg["lams"]
        ax = axes[0, col]
        ax.plot(lams, agg["err_interp"], "-o", ms=3.5, lw=1.8, color="C3",
                label="vs interpolated target")
        if agg["n_seeds"] > 1:
            ax.fill_between(lams, agg["err_interp"] - agg["err_interp_std"],
                            agg["err_interp"] + agg["err_interp_std"],
                            color="C3", alpha=0.2)
        ax.plot(lams, agg["err_switch"], "--", lw=1.5, color="0.35",
                label="vs nearer trained target")
        ax.plot(lams, agg["err_blend"], "-.", lw=1.4, color="C1",
                label="pure mixer reference")
        ax.axhline(max(agg["err_interp"][0], agg["err_interp"][-1]),
                   color="C0", lw=1.2, ls=":", label="endpoint (trained) error")
        ax.set_title(f"{world.FIELD_EPISODE[a]} -> {world.FIELD_EPISODE[b]}")
        ax.set_xlabel(r"field mixture $\lambda$")
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel("task error")
            ax.legend(fontsize=10)

        ax = axes[1, col]
        ax.plot([0, 1], [0, 1], color="0.6", lw=1.2, ls=":", label="identity")
        ax.step(lams, (lams > 0.5).astype(float), where="mid", color="0.35",
                lw=1.4, ls="--", label="discrete switch")
        ax.plot(lams, agg["lam_hat"], "-o", ms=3.5, lw=1.8, color="C2",
                label="measured")
        if agg["n_seeds"] > 1:
            ax.fill_between(lams, agg["lam_hat"] - agg["lam_hat_std"],
                            agg["lam_hat"] + agg["lam_hat_std"],
                            color="C2", alpha=0.2)
        ax.set_xlabel(r"field mixture $\lambda$")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.text(0.03, 0.95, f"Spearman {agg['spearman']:.3f}",
                transform=ax.transAxes, va="top", fontsize=11)
        if col == 0:
            ax.set_ylabel(r"implied mixture $\hat\lambda$")
            ax.legend(fontsize=10, loc="lower right")

    fig.tight_layout()
    return viz.save_or_show(fig, save_path)


def write_csv(path: Path, rows, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w") as f:
        f.write(f"# neuromod_interpolation  model={args.model} "
                f"hidden_layers={args.hidden_layers} epochs={args.epochs} "
                f"seeds={args.seeds} alpha_mix={args.alpha_mix} "
                f"eval_samples={args.eval_samples}\n")
        f.write("# training saw the three PURE fields only; every lambda in (0,1) "
                "is an untrained field\n")
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(f"{row[k]}" for k in keys) + "\n")
    print(f"saved {path}")


# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("\n\n")[0])
    p.add_argument("--model", choices=("analytic", "sample"), default="sample",
                   help="sample is the mechanism and the only informative level "
                        "here; analytic makes continuity nearly true by "
                        "construction, so use it for wiring checks only [sample]")
    p.add_argument("--hidden-layers", type=int, default=1,
                   help="1 keeps sigma-only recruitment exact (no sigma=0 leak) [1]")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=2500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grid-side", type=int, default=21)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--samples", type=int, default=48,
                   help="T during TRAINING; this is the one that costs time [48]")
    p.add_argument("--eval-samples", type=int, default=256,
                   help="T during evaluation. Evaluation is cheap, and too small a "
                        "value leaves sampling jitter large enough to make the "
                        "smoothness metric meaningless [256]")
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--base-std", type=float, default=0.8)
    p.add_argument("--sigma", type=float, default=0.22)
    p.add_argument("--theta", type=float, default=0.15)
    p.add_argument("--field-radius", type=float, default=0.28)
    p.add_argument("--alpha-mix", type=float, default=0.6,
                   help="Off-drive weight. 0.6 makes the three behaviours compete "
                        "for the same inputs, as the L2 figure does; pass a "
                        "negative value for the original near-one-hot targets [0.6]")
    p.add_argument("--target-gamma", type=float, default=2.0)
    p.add_argument("--blend", choices=("linear", "variance"), default="linear",
                   help="How two fields are interpolated. 'linear' is the "
                        "pre-registered rule but lowers every unit's sigma in the "
                        "interior; 'variance' interpolates noise power and stays "
                        "closer to the trained h/sigma operating point [linear]")
    p.add_argument("--n-lambda", type=int, default=21,
                   help="Interpolation points per pair [21]")
    p.add_argument("--n-mu", type=int, default=101,
                   help="Resolution of the implied-mixture search [101]")
    p.add_argument("--err-factor", type=float, default=1.5,
                   help="M1 passes if no interior error exceeds this multiple of "
                        "the endpoint error [1.5]")
    p.add_argument("--min-spearman", type=float, default=0.9,
                   help="M2 passes above this rank correlation [0.9]")
    p.add_argument("--max-jump-k", type=float, default=3.0,
                   help="M3 passes if the largest single step is below this many "
                        "uniform steps, i.e. K / (n_lambda - 1) of the path. At the "
                        "registered n_lambda = 21 this is the planned 0.15 [3.0]")
    p.add_argument("--min-step-over-jitter", type=float, default=2.0,
                   help="Below this, M3 cannot be distinguished from sampling "
                        "noise and is reported as uninformative [2.0]")
    p.add_argument("--net-path", default=None,
                   help="Checkpoint file for the trained nets; reused if present")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--save", action="store_true")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tag", default="interp")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mix = None if args.alpha_mix is not None and args.alpha_mix < 0 else args.alpha_mix
    args.alpha_mix = mix
    obs_np, obs, alphas, targets = build_data(args, mix)

    cache = {}
    net_path = Path(args.net_path) if args.net_path else None
    if net_path and net_path.exists():
        cache = torch.load(net_path, weights_only=False)

    label = "one-hot" if mix is None else f"{mix:g}"
    print(f"model={args.model}  hidden_layers={args.hidden_layers}  "
          f"epochs={args.epochs}  seeds={args.seeds}  alpha_mix={label}\n"
          f"training sees the 3 pure fields only; "
          f"{args.n_lambda - 2} interior lambdas per pair are untrained")

    pairs = [f"{a}|{b}" for a, b in itertools.combinations(world.CATEGORIES, 2)]
    per_pair = {pair: [] for pair in pairs}
    rows = []

    for k in range(args.seeds):
        seed = args.seed0 + k
        net, flds = get_net(seed, obs, targets, args, cache)
        losses = P.final_losses(net, obs, targets,
                               {s: flds[world.STATE_TO_FIELD[s]] for s in world.STATES},
                               world.STATES, args.hidden_layers)
        print(f"  seed {seed}: trained losses  "
              + "  ".join(f"{s.split('_')[0]}={v:.4f}" for s, v in losses.items()),
              flush=True)
        set_eval_samples(net, args.eval_samples)
        for pair in pairs:
            a, b = pair.split("|")
            res = sweep_pair(net, flds, obs, obs_np, alphas, args, a, b)
            per_pair[pair].append(res)
            for i, lam in enumerate(res["lams"]):
                rows.append({
                    "seed": seed, "pair": pair, "lam": f"{lam:.4f}",
                    "err_interp": f"{res['err_interp'][i]:.6f}",
                    "err_a": f"{res['err_a'][i]:.6f}",
                    "err_b": f"{res['err_b'][i]:.6f}",
                    "lam_hat": f"{res['lam_hat'][i]:.4f}",
                })

    if net_path:
        net_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, net_path)
        print(f"saved {net_path}")

    results = {pair: aggregate(per_pair[pair]) for pair in pairs}
    n_ok = 0
    for pair in pairs:
        a, b = pair.split("|")
        print(f"\n=== {pair}   ({world.FIELD_EPISODE[a]} -> "
              f"{world.FIELD_EPISODE[b]}) ===")
        ok, text = verdict(results[pair], args)
        n_ok += int(ok)
        print("  " + text)

    print(f"\n=== L4 over {len(pairs)} pairs: {n_ok}/{len(pairs)} continuous ===")

    if args.save:
        write_csv(viz.resolve_out(f"{args.tag}.csv", args.out_dir), rows, args)
        plot_interpolation(results, args,
                           save_path=viz.resolve_out(f"{args.tag}.png", args.out_dir))
    else:
        plot_interpolation(results, args)


if __name__ == "__main__":
    main()
