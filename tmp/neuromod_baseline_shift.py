"""neuromod_baseline_shift -- the SAME drug helps or hurts depending on baseline.

The signature prediction of an inverted U is not the U itself but what it implies
about an intervention: a fixed dose moves a low-baseline system UP the left arm
(improvement) and a high-baseline system DOWN the right arm (impairment).  In
human pharmacology this is one of the most robust findings about dopamine and
working memory: the same drug improves low-baseline individuals and impairs
high-baseline ones (Cools & D'Esposito 2011).  A look-up key cannot produce it,
because a key has no optimum for a dose to sit on either side of.

Design (this is the point of the script, so it is worth stating plainly):

    baseline = a TRAIT.  Each network is trained at its own sigma_base, i.e. it is
               adapted to the noise level it lives at.  This is the training-level
               sweep, and it is what makes the comparison fair: every baseline is a
               network that works as well as it can at that baseline.
    drug     = an ACUTE perturbation.  Weights are frozen and only the noise field
               is scaled, sigma -> gain * sigma, with the crossing threshold h held
               fixed.  Holding h fixed matters: scaling sigma and h together is a
               gauge transformation and would change nothing at all
               (`docs/idea_core.md` section 4.6), so the drug has to move sigma alone.

Predicted contrast, which doubles as the control: the sign flip requires an
interior optimum, so it should appear for `--model sample` (real noise injection
with a threshold) and NOT for `--model analytic` (mean field, no subthreshold
barrier, optimum at the low end).  Running both is the honest version of the claim.

Cost: one network is trained per (baseline, seed), and the sample model runs at
roughly 1 s/epoch almost independently of grid size and T (the cost is per-step
overhead, not tensor size).  So epochs x baselines x seeds is the wall clock in
seconds; budget accordingly and run the big ones in the background.

    .venv/bin/python tmp/neuromod_baseline_shift.py --epochs 40 --baselines 4 \
        --grid-side 9 --samples 16                                   # plumbing check, ~2 min
    .venv/bin/python tmp/neuromod_baseline_shift.py --save           # default, ~35 min
    .venv/bin/python tmp/neuromod_baseline_shift.py --epochs 1000 --grid-side 21 \
        --baselines 9 --seeds 3 --save                               # paper version, hours
    .venv/bin/python tmp/neuromod_baseline_shift.py --model analytic --save \
        --tag baseline_analytic                                      # the null control (fast)

Output: tmp/out/<tag>_curve.png and tmp/out/<tag>.csv
Context: `docs/idea_neuromod.md` section 7.2.
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


def build_data(grid_side: int, gamma: float, alpha_mix):
    """Scripted scene -> grid observations and the three per-state targets."""
    objects = world.make_scripted_objects()
    alphas = world.alpha_states(alpha_mix)
    obs_np = world.encode_observations(world.make_training_grid(grid_side), objects)
    obs = torch.tensor(obs_np, dtype=torch.float32)
    targets = {s: torch.tensor(
                   world.make_mixed_behavior_targets(obs_np, alphas[s], gamma=gamma),
                   dtype=torch.float32)
               for s in world.STATES}
    return obs, targets


def score(net, obs, targets, peak_std, args):
    """Benchmark scores with the fields rebuilt at a given peak recruited-unit std."""
    fields = F.build_fields(world.CATEGORIES, args.hidden_dim, float(peak_std),
                            args.sigma, args.theta, radius=args.field_radius)
    return P.capability(net, obs, targets, fields, world.CATEGORIES, world.STATES,
                        world.STATE_TO_FIELD, N_HIDDEN)


def run_one(sigma_base: float, seed: int, obs, targets, args):
    """Train a network at its own baseline, then dose it at every gain and re-score.

    Returns one record per dose.  Training happens once, because the drug is an
    acute perturbation of a frozen network, so the dose axis is nearly free: sweep
    it rather than re-running the whole experiment per dose.  `d_err` is the drug's
    effect on task error, so NEGATIVE means the drug helped.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    fields = F.build_fields(world.CATEGORIES, args.hidden_dim, float(sigma_base),
                            args.sigma, args.theta, radius=args.field_radius)
    state_fields = {s: fields[world.STATE_TO_FIELD[s]] for s in world.STATES}
    net = P.build_network(args.hidden_dim, n_hidden=N_HIDDEN,
                          base_std=float(sigma_base), kind=args.model,
                          t=args.samples, crossing_h=args.crossing_h)
    P.train(net, obs, targets, state_fields, world.STATES, N_HIDDEN,
            args.epochs, args.lr, verbose=False)

    sep_b, sig_b, err_b = score(net, obs, targets, sigma_base, args)
    out = []
    for gain in args.drug_gains:
        sep_d, sig_d, err_d = score(net, obs, targets, sigma_base * gain, args)
        out.append({
            "seed": seed,
            "gain": float(gain),
            "sigma_base": float(sigma_base),
            "sigma_drug": float(sigma_base * gain),
            "h_over_sigma_base": args.crossing_h / float(sigma_base),
            "err_base": err_b, "err_drug": err_d, "d_err": err_d - err_b,
            "signal_base": sig_b, "signal_drug": sig_d, "d_signal": sig_d - sig_b,
            "sep_base": sep_b, "sep_drug": sep_d,
        })
    return out


AGG_KEYS = ("err_base", "err_drug", "d_err", "signal_base", "signal_drug",
            "d_signal", "sep_base")


def aggregate(records, baselines, gain):
    """Mean and std over seeds for each baseline at one dose, in `baselines` order."""
    out = []
    for sigma in baselines:
        cell = [r for r in records
                if np.isclose(r["sigma_base"], sigma) and np.isclose(r["gain"], gain)]
        agg = {"sigma_base": float(sigma), "gain": float(gain), "n": len(cell),
               "h_over_sigma_base": cell[0]["h_over_sigma_base"]}
        for key in AGG_KEYS:
            values = np.array([c[key] for c in cell], dtype=float)
            agg[key] = float(values.mean())
            agg[key + "_std"] = float(values.std())
        out.append(agg)
    return out


def crossover(agg):
    """Baseline where the drug's effect changes sign, by linear interpolation.

    Returns None when the effect keeps one sign across the whole range, which is
    what the mean-field control is expected to do.
    """
    sigma = np.array([a["sigma_base"] for a in agg])
    d_err = np.array([a["d_err"] for a in agg])
    for i in range(len(d_err) - 1):
        if d_err[i] < 0.0 <= d_err[i + 1]:
            w = -d_err[i] / (d_err[i + 1] - d_err[i])
            return float(sigma[i] + w * (sigma[i + 1] - sigma[i]))
    return None


def write_csv(path: Path, rows, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w") as f:
        f.write(f"# neuromod_baseline_shift  model={args.model} "
                f"drug_gains={args.drug_gains} h={args.crossing_h} "
                f"epochs={args.epochs} grid_side={args.grid_side} "
                f"hidden={args.hidden_dim} seeds={args.seeds}\n")
        f.write("# d_err < 0 means the drug IMPROVED the task\n")
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(f"{row[k]}" for k in keys) + "\n")
    print(f"saved {path}")


def plot(per_gain, args, save_path=None):
    """Two panels: where each baseline sits on the U, and the effect's sign flip.

    `per_gain` maps dose -> aggregated rows.  With several doses the lower panel
    becomes one line per dose, which shows how the crossover moves with the dose.
    """
    viz.use_headless(save_path)
    import matplotlib.pyplot as plt
    plt.rcParams.update(viz.FONT)

    gains = sorted(per_gain)
    ref = per_gain[gains[0]]
    sigma = np.array([a["sigma_base"] for a in ref])
    err_b = np.array([a["err_base"] for a in ref])
    err_b_std = np.array([a["err_base_std"] for a in ref])

    fig, (ax_u, ax_d) = plt.subplots(2, 1, figsize=(8.5, 8.4), sharex=True)

    # Panel A: the trained-level curve, with the acute dose drawn as an arrow.
    # Arrow tails sit ON the curve (each net at its own baseline); the heads are
    # OFF it, because the drugged point is the same frozen net evaluated elsewhere.
    ax_u.plot(sigma, err_b, "-o", color="0.25", lw=1.6,
              label="trained at its own baseline")
    if args.seeds > 1:
        ax_u.fill_between(sigma, err_b - err_b_std, err_b + err_b_std,
                          color="0.25", alpha=0.18)
    show = per_gain[gains[-1]]
    err_d = np.array([a["err_drug"] for a in show])
    d_err = np.array([a["d_err"] for a in show])
    for x, y0, y1 in zip(sigma, err_b, err_d):
        ax_u.annotate("", xy=(x, y1), xytext=(x, y0),
                      arrowprops=dict(arrowstyle="->", lw=2.0,
                                      color="tab:blue" if y1 < y0 else "tab:red"))
    ax_u.scatter(sigma, err_d, s=28, zorder=5,
                 color=["tab:blue" if d < 0 else "tab:red" for d in d_err],
                 label=f"after drug (sigma x {gains[-1]:g})")
    ax_u.set_ylabel("task error")
    ax_u.legend(loc="upper center")
    ax_u.grid(alpha=0.3)

    # Panel B: the effect itself.  A zero crossing is the whole claim.
    if len(gains) == 1:
        d_err = np.array([a["d_err"] for a in ref])
        d_std = np.array([a["d_err_std"] for a in ref])
        width = 0.6 * float(np.min(np.diff(sigma))) if len(sigma) > 1 else 0.2
        ax_d.bar(sigma, -d_err, width=width, alpha=0.85,
                 color=["tab:blue" if d < 0 else "tab:red" for d in d_err])
        if args.seeds > 1:
            ax_d.errorbar(sigma, -d_err, yerr=d_std, fmt="none", ecolor="0.3",
                          capsize=3)
    else:
        for gain in gains:
            rows = per_gain[gain]
            improve = -np.array([a["d_err"] for a in rows])
            std = np.array([a["d_err_std"] for a in rows])
            line, = ax_d.plot(sigma, improve, "-o", lw=1.6,
                              label=f"sigma x {gain:g}")
            if args.seeds > 1:
                ax_d.fill_between(sigma, improve - std, improve + std,
                                  color=line.get_color(), alpha=0.18)
        ax_d.legend(title="dose", ncol=2)
    ax_d.axhline(0.0, color="0.2", lw=1.2)
    ax_d.set_xlabel(r"baseline noise level  $\sigma_{\rm base}$"
                    f"   (h={args.crossing_h} fixed)")
    ax_d.set_ylabel("improvement from drug\n(> 0 = helped)")
    ax_d.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    return viz.save_or_show(fig, save_path)


def summarise(per_gain, args) -> None:
    print(f"\nh={args.crossing_h} fixed, model={args.model}, seeds={args.seeds}")
    any_flip = False
    for gain in sorted(per_gain):
        agg = per_gain[gain]
        helped = [a["sigma_base"] for a in agg if a["d_err"] < 0]
        hurt = [a["sigma_base"] for a in agg if a["d_err"] > 0]
        cross = crossover(agg)
        print(f"  dose sigma x {gain:g}:")
        print("    helped at: "
              + (", ".join(f"{s:.2f}" for s in helped) if helped else "(none)"))
        print("    hurt   at: "
              + (", ".join(f"{s:.2f}" for s in hurt) if hurt else "(none)"))
        if cross is not None:
            any_flip = True
            print(f"    SIGN FLIP at sigma_base ~ {cross:.3f} "
                  f"(h/sigma = {args.crossing_h / cross:.3f})")
        else:
            direction = "helped" if helped else "hurt"
            print(f"    no sign flip: {direction} at every baseline")
    if any_flip:
        print("  A look-up key cannot produce a sign flip; it requires an interior "
              "optimum.")
    else:
        print("  No flip at any dose. Expected for --model analytic (the mean field "
              "has no interior optimum); for --model sample, widen --s-min/--s-max.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("\n\n")[0])
    p.add_argument("--model", choices=("analytic", "sample"), default="sample",
                   help="sample = mechanism (interior optimum, so a flip can exist); "
                        "analytic = mean field, the predicted null [sample]")
    p.add_argument("--drug-gains", default="1.4",
                   help="Comma-separated acute doses as multipliers on sigma, h held "
                        "fixed. Doses are probed on the SAME trained network, so "
                        "extra doses are nearly free [1.4]")
    p.add_argument("--s-min", type=float, default=0.3, help="Lowest baseline [0.3]")
    p.add_argument("--s-max", type=float, default=2.2, help="Highest baseline [2.2]")
    p.add_argument("--baselines", type=int, default=7,
                   help="Number of baselines spanning s-min..s-max [7]")
    p.add_argument("--seeds", type=int, default=1,
                   help="Networks per baseline; >1 gives error bars [1]")
    p.add_argument("--epochs", type=int, default=300,
                   help="Epochs per network. The sample model costs about 1 s/epoch, "
                        "so this times baselines times seeds is the wall clock [300]")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grid-side", type=int, default=13,
                   help="Training grid resolution per axis [13]")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--samples", type=int, default=24,
                   help="Stochastic forward samples T, sample model only [24]")
    p.add_argument("--crossing-h", type=float, default=0.2,
                   help="Crossing threshold h; held FIXED across the sweep, which is "
                        "what makes sigma a real degree of freedom [0.2]")
    p.add_argument("--sigma", type=float, default=0.22,
                   help="Bump WIDTH on the unit sheet (not the noise strength) [0.22]")
    p.add_argument("--theta", type=float, default=0.15)
    p.add_argument("--field-radius", type=float, default=0.28)
    p.add_argument("--alpha-mix", type=float, default=None)
    p.add_argument("--target-gamma", type=float, default=2.0)
    p.add_argument("--seed0", type=int, default=0, help="First seed [0]")
    p.add_argument("--save", action="store_true",
                   help="Write the figure and CSV instead of showing the figure")
    p.add_argument("--out-dir", default=None, help="Output directory [tmp/out]")
    p.add_argument("--tag", default="baseline_shift")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.drug_gains = [float(g) for g in str(args.drug_gains).split(",") if g.strip()]
    baselines = np.linspace(args.s_min, args.s_max, args.baselines)
    obs, targets = build_data(args.grid_side, args.target_gamma, args.alpha_mix)

    n_nets = args.baselines * args.seeds
    print(f"model={args.model}  baselines={args.baselines} in "
          f"[{args.s_min}, {args.s_max}]  seeds={args.seeds}  epochs={args.epochs}  "
          f"doses={args.drug_gains}  ({n_nets} networks to train, "
          f"~{n_nets * args.epochs / 60:.0f} min for the sample model)")

    records = []
    for sigma_base in baselines:
        for k in range(args.seeds):
            cells = run_one(sigma_base, args.seed0 + k, obs, targets, args)
            records.extend(cells)
            effects = "  ".join(
                f"x{c['gain']:g}:{c['d_err']:+.4f}" for c in cells)
            print(f"  sigma_base={sigma_base:5.2f} seed={args.seed0 + k}  "
                  f"err={cells[0]['err_base']:.4f}   d_err  {effects}", flush=True)

    per_gain = {g: aggregate(records, baselines, g) for g in args.drug_gains}
    summarise(per_gain, args)

    if args.save:
        write_csv(viz.resolve_out(f"{args.tag}.csv", args.out_dir), records, args)
        plot(per_gain, args,
             save_path=viz.resolve_out(f"{args.tag}_curve.png", args.out_dir))
    else:
        plot(per_gain, args)


if __name__ == "__main__":
    main()
