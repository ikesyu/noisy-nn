"""
examples/sr_separation_curve.py

Stochastic-resonance (SR) signature of neuromodulator-like noise-field recruitment
==================================================================================

This is a diagnostic companion to `neuromodulated_behavior_modes.py`.  It asks a
single question:

    Is the noise field a FUNCTIONAL RESOURCE with an optimal strength -- i.e. does
    the recruited computation show the classic SR inverted-U as a function of noise
    magnitude -- rather than a mere symbolic "key" that would work at any magnitude?

We reuse the trained shared-weight NNN from the behaviour demo: ONE network is
trained under the three neuromodulatory noise fields (food / threat / shelter) at a
fixed base noise level.  Then, at TEST time, we scale every noise field by a common
factor and sweep the peak recruited-unit std

    s = peak std of the recruited hidden units   (a proxy for neuromodulator
                                                   concentration)

from ~0 up to several times the trained level, and measure how the recruited output
behaves.  We report three curves vs s:

  * separation(s)  -- mean pairwise distance between the three fields' output vector
                      fields (candidate 2): "how distinctly do the fields recruit
                      different behaviours".
  * signal(s)      -- the input-driven part of the output (its variation across
                      observations after removing the per-field mean): "how much
                      structured, stimulus-locked signal the recruited subnetwork
                      carries".  This is the quantity SR is expected to make
                      unimodal: 0 when the units are detached (s->0) and 0 again when
                      the crossing activation saturates to a constant (s large).
  * task-error(s)  -- MSE of the recruited field against the trained target
                      behaviour (lower = better); expected to dip near the optimum.

Interpretation / paper story
----------------------------
A look-up key has no optimal magnitude.  An SR-based functional resource does: too
little noise leaves the recruited subnetwork sub-threshold (detached), too much
swamps the stimulus-locked signal, and an intermediate level is best.  A clear
interior peak in signal(s) (and separation(s)) is the SR monomodal curve.

Caveat: the network here is `SimpleNNNAnalytic` (the deterministic EXPECTATION of
the noisy crossing) and it is trained at one base level, so task-error(s) is
minimised near the trained level by construction.  separation(s)/signal(s) are not
the training objective, so their optimum is a more honest SR signal.  The rigorous
follow-ups (see neuromodulated_behavior_modes notes) are: (a) repeat with
`SimpleNNNSample` (real injected noise), and (b) a training-level sweep.

Model variants (--model)
------------------------
  analytic  : the deterministic expectation of the noisy crossing (default, fast;
              its optimum is a mean-field effect and has no crossing threshold h).
  sample    : REAL injected Gaussian noise averaged over t samples, with a crossing
              threshold h>0 -- the genuine SR mechanism (sub-threshold signal is only
              transmitted with the help of noise).  Slower and a little noisier.
  statistic : a moment-based estimate of the sampled response (also uses h, t).
The chosen model is trained AND probed as itself, so the curve is self-consistent.

Two sweeps (--sweep)
--------------------
  test   : train ONE net at --base-std, then sweep the TEST-time noise strength
           (fast; but task-error is minimised near the trained level by construction).
  train  : the rigorous version -- train a FRESH net at each noise std and score it
           AT that std.  Every point is trained-and-evaluated at its own level, so an
           interior optimum shows the OPTIMAL TRAINING noise is genuinely interior
           (no train/test confound).  Costs one training run per level.
           Use it with --model sample: the real-noise threshold gives a genuine
           interior optimum (sub-threshold signals need noise to be transmitted),
           whereas --model analytic has no low-noise barrier (the net rescales its
           weights), so its training-level optimum sits at the LOW end.

Run
---
    python examples/sr_separation_curve.py
    python examples/sr_separation_curve.py --sweep train --train-steps 13
    python examples/sr_separation_curve.py --model sample --epochs 1200 --samples 64
    python examples/sr_separation_curve.py --sweep train --model sample --grid-side 21 \
        --epochs 1000 --train-steps 9

    python examples/sr_separation_curve.py --sweep train --model sample --grid-side 21 --epochs 1000 --train-steps 9   # 厳密なSR（内点最適）
    python examples/sr_separation_curve.py --sweep train --model analytic                                              # 対照（端点最適＝SRなし）
    python examples/sr_separation_curve.py                                                                             # 従来のテスト時スイープ（既定）
    （記憶にも要点を記録しました。）補足の TODO 候補：高σ側の減衰は穏やかなので、より明瞭な逆U字の裾を出すには σ の上限を広げる（例：〜5）と良いです。また Analytic と E[Sample] の重ね描きで「平均場 vs 機構」を1枚にする案も、必要になれば追加できます。

Data export
-----------
Pass --save PATH to write the swept curve to a CSV (metadata in leading '#' comments;
columns x, separation, signal, task_err), and --no-show for headless batch runs.
Read back with numpy.loadtxt(PATH, delimiter=',').  These CSVs are the inputs to the
paper figures (see docs/recipe_sr.md for which runs make which plot).

Requires only numpy, torch, matplotlib.  Without --save, no files are written.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Reuse the behaviour-demo module (its top level only defines things; main() is
# guarded by __main__, so importing it is side-effect free).
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
import neuromodulated_behavior_modes as demo


def build_model(kind: str, hidden: int, base_std: float, h: float, t: int):
    """Instantiate one of the three NNN variants; all accept `stds` in forward().

    analytic  : deterministic EXPECTATION of the noisy crossing (no threshold h).
    sample    : real injected Gaussian noise, averaged over t samples, with a
                crossing threshold h -- the genuine stochastic-resonance mechanism
                (a sub-threshold signal is only transmitted with the help of noise).
    statistic : moment-based estimate of the sampled response (also uses h, t).
    """
    structure = [6, hidden, hidden, 2]
    if kind == "analytic":
        return demo.model.SimpleNNNAnalytic(structure=structure, std=base_std)
    if kind == "sample":
        return demo.model.SimpleNNNSample(structure=structure, std=base_std, h=h, t=t)
    if kind == "statistic":
        return demo.model.SimpleNNNStatistic(structure=structure, std=base_std, h=h, t=t)
    raise ValueError(f"unknown model kind '{kind}'")


def build_data(grid_side: int, gamma: float, device: torch.device):
    """Scripted scene -> grid observations and the three per-state target fields."""
    objects = demo.make_scripted_objects()
    positions = demo.make_training_grid(grid_side)
    obs_np = demo.encode_observations(positions, objects)                 # [N, 6]
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
    targets = {s: torch.tensor(
                   demo.make_mixed_behavior_targets(obs_np, demo.ALPHA_STATES[s],
                                                    gamma=gamma),
                   dtype=torch.float32, device=device)
               for s in demo.STATES}
    return obs, targets


def make_fields(peak_std: float, sigma: float, theta: float, device: torch.device):
    """The three category noise fields at a given peak recruited-unit std."""
    return {c: demo.make_noise_field(demo.FIELD_CENTERS[c], float(peak_std), sigma,
                                     theta).to(device)
            for c in demo.CATEGORIES}


def measure_capability(net: nn.Module, obs: torch.Tensor, targets: dict,
                       fields: dict, device: torch.device):
    """Return (separation, signal, task_err) of `net` under the given noise fields.

    separation = mean pairwise ||y_i - y_j|| between the three fields' output fields;
    signal     = mean stimulus-locked variation ||y - mean_obs(y)|| (input-driven);
    task_err   = mean over states of MSE(recruited output, trained target).
    """
    cats = demo.CATEGORIES
    with torch.no_grad():
        y = {c: demo.evaluate_vector_field(net, obs, fields[c]).cpu().numpy()
             for c in cats}                                               # each [N, 2]
    separation = (np.linalg.norm(y["food"] - y["threat"], axis=1).mean()
                  + np.linalg.norm(y["food"] - y["shelter"], axis=1).mean()
                  + np.linalg.norm(y["threat"] - y["shelter"], axis=1).mean()) / 3.0
    signal = np.mean([np.linalg.norm(y[c] - y[c].mean(axis=0, keepdims=True),
                                     axis=1).mean() for c in cats])
    task_err = np.mean([np.mean((y[demo.STATE_TO_FIELD[s]] - targets[s].cpu().numpy())
                                ** 2) for s in demo.STATES])
    return float(separation), float(signal), float(task_err)


def build_and_train(kind: str, epochs: int, lr: float, base_std: float, sigma: float,
                    theta: float, grid_side: int, gamma: float, hidden: int,
                    h: float, t: int, device: torch.device):
    """Build the scripted scene, train the shared-weight NNN under the three fields.

    The chosen model is trained AND probed as itself (no cross-model weight transfer),
    so the SR curve is self-consistent.  Returns (net, obs, targets, base_fields).
    """
    obs, targets = build_data(grid_side, gamma, device)
    base_fields = make_fields(base_std, sigma, theta, device)
    state_fields = {s: base_fields[demo.STATE_TO_FIELD[s]] for s in demo.STATES}

    net = build_model(kind, hidden, base_std, h, t).to(device)
    extra = f", h={h}, t={t}" if kind != "analytic" else ""
    print(f"Training {kind} NNN ({epochs} epochs) under the three fields at base "
          f"std={base_std}{extra} ...")
    if kind != "analytic":
        print("  (sample/statistic inject real noise -> slower & a bit noisier; "
              "reduce --epochs / --grid-side if needed)")
    demo.train(net, obs, targets, state_fields, epochs, lr)
    return net, obs, targets, base_fields


def sr_scan(net: nn.Module, obs: torch.Tensor, targets: dict,
            s_values: np.ndarray, sigma: float, theta: float,
            device: torch.device):
    """Sweep the peak recruited-unit std s and measure separation / signal / error.

    For each s we rebuild each category's noise field with peak std = s (same active
    unit set, only the magnitude changes), evaluate the trained net, and compute:
      separation[k] = mean pairwise ||y_i - y_j|| over observations,
      signal[k]     = mean over fields of ||y - mean_obs(y)|| (stimulus-locked part),
      task_err[k]   = mean over states of MSE(recruited output, trained target).
    """
    separation = np.zeros_like(s_values)
    signal = np.zeros_like(s_values)
    task_err = np.zeros_like(s_values)
    for k, s in enumerate(s_values):
        fields_s = make_fields(s, sigma, theta, device)
        separation[k], signal[k], task_err[k] = measure_capability(
            net, obs, targets, fields_s, device)
    return separation, signal, task_err


def train_level_sweep(kind: str, sigma_values: np.ndarray, epochs: int, lr: float,
                      sigma: float, theta: float, grid_side: int, gamma: float,
                      hidden: int, h: float, t: int, device: torch.device):
    """TRAINING-LEVEL sweep: train a FRESH net at each std and score it AT that std.

    This removes the test-time confound (one net evaluated away from where it was
    trained): every point is a network that was both trained and evaluated at its own
    noise level sigma_train, so a clear interior optimum in signal/separation is a
    genuine "optimal noise level for the recruited computation" -- the rigorous SR
    statement.  Sub-threshold (sigma->0) the units are detached and cannot be trained;
    over-strong noise saturates the crossing to a constant, so the task cannot be fit.
    """
    obs, targets = build_data(grid_side, gamma, device)
    separation = np.zeros_like(sigma_values)
    signal = np.zeros_like(sigma_values)
    task_err = np.zeros_like(sigma_values)

    for i, st in enumerate(sigma_values):
        fields = make_fields(st, sigma, theta, device)
        state_fields = {s: fields[demo.STATE_TO_FIELD[s]] for s in demo.STATES}
        net = build_model(kind, hidden, float(st), h, t).to(device)
        with contextlib.redirect_stdout(io.StringIO()):        # hush per-epoch logs
            demo.train(net, obs, targets, state_fields, epochs, lr)
        separation[i], signal[i], task_err[i] = measure_capability(
            net, obs, targets, fields, device)
        print(f"  sigma_train={st:5.2f} -> signal={signal[i]:.3f}  "
              f"separation={separation[i]:.3f}  task_err={task_err[i]:.4f}")
    return separation, signal, task_err


def plot_curves(x_values, separation, signal, task_err, model_kind, xlabel,
                trained_level=None, sweep="test"):
    """Plot the SR curves; annotate the interior optimum (and trained level, test)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # separation and signal share the left axis (higher = more/structured output)
    ax.plot(x_values, separation, "-o", ms=3, color="tab:blue",
            label="field separation  (candidate 2)")
    ax.plot(x_values, signal, "-s", ms=3, color="tab:green",
            label="stimulus-locked signal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("output magnitude  (separation / signal)")
    ax.grid(alpha=0.3)

    # task error on a twin axis (lower = better)
    ax2 = ax.twinx()
    ax2.plot(x_values, task_err, "--^", ms=3, color="tab:red", alpha=0.7,
             label="task error (MSE)")
    ax2.set_ylabel("task error (MSE)  [lower = better]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # mark the SR optimum (peak of the stimulus-locked signal)
    x_opt = x_values[int(np.argmax(signal))]
    ax.axvline(x_opt, color="tab:green", ls=":", lw=1.5,
               label=f"signal optimum  = {x_opt:.2f}")
    if trained_level is not None:                      # only meaningful for test sweep
        ax.axvline(trained_level, color="0.4", ls="-.", lw=1.2,
                   label=f"trained level  ({trained_level:.2f})")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)

    if sweep == "train":
        sub = ("TRAINING-LEVEL sweep: each net is trained AND evaluated at its own "
               "noise std\nan interior optimum => a genuinely optimal noise level "
               "(no train/test confound)")
    else:
        sub = ("test-time noise sweep on one trained net\nan interior optimum in "
               "signal/separation => the noise field is a functional resource, not a "
               "symbolic key")
    fig.suptitle(f"Stochastic-resonance signature of noise-field recruitment "
                 f"({model_kind} NNN)\n{sub}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    plt.show()


def save_curves(path: str, x_values, separation, signal, task_err, args, x_name: str):
    """Write the swept curve to a CSV: metadata in '#' comments, then numeric rows.

    Every non-data line (including the column names) is a '#' comment, so the data is
    purely numeric and reads back with just:
        data = numpy.loadtxt(path, delimiter=',')      # columns below, in order
        pandas.read_csv(path, comment='#', header=None,
                        names=['x', 'separation', 'signal', 'task_err'])
    Column order: <x_name>, separation, signal, task_err  (x_name is 'sigma_train' for
    the --sweep train run and 'test_s' for the --sweep test run).
    """
    import os
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    k = int(np.argmax(signal))
    interior = 0 < k < len(x_values) - 1
    with open(path, "w") as f:
        f.write("# SR sweep data  (examples/sr_separation_curve.py)\n")
        f.write(f"# model={args.model} sweep={args.sweep} x_variable={x_name}\n")
        f.write(f"# base_std={args.base_std} sigma={args.sigma} theta={args.theta} "
                f"target_gamma={args.target_gamma}\n")
        f.write(f"# grid_side={args.grid_side} hidden_dim={args.hidden_dim} "
                f"epochs={args.epochs} lr={args.lr}\n")
        f.write(f"# crossing_h={args.crossing_h} samples={args.samples} seed={args.seed}\n")
        if args.sweep == "train":
            f.write(f"# train_steps={args.train_steps} s_min={args.s_min} "
                    f"s_max={args.s_max}\n")
        else:
            f.write(f"# s_min={args.s_min} s_max={args.s_max} s_steps={args.s_steps}\n")
        f.write(f"# signal_optimum={x_values[k]:.6g} signal_peak={signal[k]:.6g} "
                f"interior={interior}\n")
        f.write(f"# columns: {x_name},separation,signal,task_err\n")
        for i in range(len(x_values)):
            f.write(f"{x_values[i]:.6g},{separation[i]:.6g},{signal[i]:.6g},"
                    f"{task_err[i]:.6g}\n")
    print(f"  saved curve data -> {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Stochastic-resonance (inverted-U) curve of neuromodulator-like "
                    "noise-field recruitment, using the trained behaviour-demo NNN.")
    p.add_argument("--sweep", choices=("test", "train"), default="test",
                   help="'test' sweep test-time noise on ONE trained net (default), or "
                        "'train' train a FRESH net at each std and score it there "
                        "(rigorous: shows the OPTIMAL TRAINING std is interior)")
    p.add_argument("--train-steps", type=int, default=13,
                   help="Number of training levels for --sweep train [13]")
    p.add_argument("--model", choices=("analytic", "sample", "statistic"),
                   default="analytic",
                   help="NNN variant: 'analytic' expectation (fast, default), "
                        "'sample' real injected noise (genuine SR), or 'statistic'")
    p.add_argument("--crossing-h", type=float, default=0.2,
                   help="Crossing threshold h for sample/statistic (SR needs h>0) [0.2]")
    p.add_argument("--samples",   type=int,   default=64,
                   help="Monte-Carlo samples t for sample/statistic [64]")
    p.add_argument("--epochs",    type=int,   default=3000, help="Training epochs [3000]")
    p.add_argument("--lr",        type=float, default=3e-4, help="Adam lr [3e-4]")
    p.add_argument("--grid-side", type=int,   default=31,   help="Training grid per axis [31]")
    p.add_argument("--hidden-dim", type=int,  default=64,   help="Hidden units (8x8 sheet) [64]")
    p.add_argument("--base-std",  type=float, default=0.8,  help="Trained peak noise std [0.8]")
    p.add_argument("--sigma",     type=float, default=0.22, help="Noise-bump width on the sheet [0.22]")
    p.add_argument("--theta",     type=float, default=0.15, help="Intensity truncation threshold [0.15]")
    p.add_argument("--target-gamma", type=float, default=2.0, help="tanh target speed gain [2.0]")
    p.add_argument("--s-min",     type=float, default=0.0,  help="Min noise strength to scan [0.0]")
    p.add_argument("--s-max",     type=float, default=3.0,  help="Max noise strength to scan [3.0]")
    p.add_argument("--s-steps",   type=int,   default=41,   help="Number of scan points [41]")
    p.add_argument("--seed",      type=int,   default=7,    help="Random seed [7]")
    # Output.
    p.add_argument("--save",      type=str,   default=None,
                   help="Write the swept curve data to this CSV path (metadata in "
                        "leading '#' comment lines) for later paper plotting")
    p.add_argument("--no-show",   action="store_true",
                   help="Do not open the matplotlib window (use with --save for "
                        "headless batch runs)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.hidden_dim != demo.GRID_SIZE * demo.GRID_SIZE:
        raise ValueError(f"--hidden-dim must be {demo.GRID_SIZE ** 2}")
    device = torch.device("cpu")

    print("Model variants selectable via --model: analytic | sample | statistic")
    print(f"  selected model: --model {args.model}   |   sweep: --sweep {args.sweep}")

    if args.sweep == "train":
        # rigorous training-level sweep: one fresh net per noise level.
        lo = max(args.s_min, 0.1)   # cannot train at std~0 (units detached, no grad)
        x_values = np.linspace(lo, args.s_max, args.train_steps, dtype=np.float64)
        print(f"\nTraining {args.train_steps} fresh {args.model} nets, one per std in "
              f"[{lo:.2f}, {args.s_max:.2f}] ({args.epochs} epochs each) ...")
        if args.model == "analytic":
            print("  NOTE: the analytic model has no real threshold barrier, so it can"
                  " rescale weights to work at any low std -> its training-level"
                  " optimum sits at the LOW end (no genuine SR). Use --model sample"
                  " for the interior SR optimum.")
        else:
            print("  (sample/statistic are slow; consider --grid-side 21 "
                  "--epochs 1000 --train-steps 9)")
        separation, signal, task_err = train_level_sweep(
            args.model, x_values, args.epochs, args.lr, args.sigma, args.theta,
            args.grid_side, args.target_gamma, args.hidden_dim,
            args.crossing_h, args.samples, device)
        xlabel = "TRAINING noise level  sigma_train  =  peak recruited-unit std"
        trained_level = None
    else:
        net, obs, targets, _ = build_and_train(
            args.model, args.epochs, args.lr, args.base_std, args.sigma, args.theta,
            args.grid_side, args.target_gamma, args.hidden_dim,
            args.crossing_h, args.samples, device)
        x_values = np.clip(np.linspace(args.s_min, args.s_max, args.s_steps), 1e-3, None)
        separation, signal, task_err = sr_scan(net, obs, targets, x_values,
                                               args.sigma, args.theta, device)
        xlabel = "test-time noise strength  s  =  peak recruited-unit std"
        trained_level = args.base_std

    k = int(np.argmax(signal))
    interior = 0 < k < len(x_values) - 1
    print(f"\n{args.sweep}-sweep summary ({args.model}):")
    print(f"  stimulus-locked signal peaks at {x_values[k]:.3f}  "
          f"({'INTERIOR optimum -> SR confirmed' if interior else 'at an endpoint'})")
    print(f"  signal:   ends={signal[0]:.4f} / {signal[-1]:.4f}, peak={signal[k]:.4f}")
    print(f"  separation at optimum = {separation[k]:.4f}")
    print(f"  task err: ends={task_err[0]:.4f} / {task_err[-1]:.4f}, "
          f"min={task_err.min():.4f}")

    if args.save:
        x_name = "sigma_train" if args.sweep == "train" else "test_s"
        save_curves(args.save, x_values, separation, signal, task_err, args, x_name)

    if args.no_show:
        print("  (--no-show: skipping the plot window)")
    else:
        print("\nOpening SR-curve window (close it to exit)...")
        plot_curves(x_values, separation, signal, task_err, args.model, xlabel,
                    trained_level=trained_level, sweep=args.sweep)


if __name__ == "__main__":
    main()
