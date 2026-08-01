"""(c1) Online, non-stationary (concept drift): does adapting the FEATURE MAP
online (forward-only) beat adapting only a linear read-out?
(docs/idea_reservoir.md §13.3 path (c).)

A streaming task whose target function SWITCHES mid-stream. The target is a
non-polynomial gaussian detector of a delayed input, and the switch changes BOTH
the delay and the detector centre -- so the required nonlinear feature changes,
not just a linear reweighting. Both methods use the SAME sliding-window online
protocol (re-fit on the recent window, predict the next block):

    NG-RC  fixed polynomial features; only the ridge READ-OUT re-fits online.
    LMU(A) / Ours(B)  the learned crossing FEATURE MAP keeps training forward-only
           (warm-started) on the recent window -> the features re-specialise.

If the target needed only a linear reweighting of fixed features, NG-RC's instant
ridge would track it and there is no advantage. Because it needs a DIFFERENT
nonlinear feature per regime, the feature-adapting methods should track below
NG-RC's fixed-basis floor. Metric: rolling test NRMSE over the stream + mean
steady-state NRMSE per regime (excluding the post-switch transient).
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import reservoir as R
from reservoir.baselines import NGRC
from reservoir.readout import ridge_fit, standardize_fit, nrmse

THETA, HF = 40, 32


def lag(u, k):
    z = np.zeros_like(u); z[k:] = u[:len(u) - k]; return z


def drift_target(u, switches):
    """Piecewise gaussian detector of a delayed input; regime params per segment.
    regimes cycle (delay, centre): A=(4, 0.0), B=(20, -0.5)."""
    regimes = [(4, 0.0), (20, -0.5)]
    y = np.zeros_like(u)
    bounds = [0] + list(switches) + [len(u)]
    for si in range(len(bounds) - 1):
        d, c = regimes[si % 2]
        seg = slice(bounds[si], bounds[si + 1])
        y[seg] = np.exp(-((lag(u, d)[seg] - c) / 0.35) ** 2)
    return y


def rolling_ngrc(u, y, blocks, W, Hist, alpha=1e-2, delay=22):
    # delay just covers the deepest regime lag (20); F<<Hist keeps ridge stable.
    X = NGRC(delay=delay, degree=2, stride=1).features(u)
    errs = []
    for i in blocks:
        win = slice(i - Hist, i)
        mu, sd = standardize_fit(X[win])
        Xw = (X - mu) / sd
        Xb = np.concatenate([Xw, np.ones((len(u), 1))], 1)
        Wt = ridge_fit(Xb[win], y[win], alpha)
        pred = Xb @ Wt
        errs.append(nrmse(y[i:i + W], pred[i:i + W]))
    return np.array(errs)


def rolling_map(u, y, blocks, W, Hist, kind, Ho, k_epochs):
    A = R.LDNField(H=HF, theta=float(THETA)).run(u)
    tr0 = np.zeros(len(u), bool); tr0[:Hist] = True
    if kind == "ours":
        m = R.NoiseModulatedMap(A, y, tr0, Ho=Ho, mix=True, seed=0)
    else:
        m = R.LearnedCrossingMap(A, y, tr0, Ho=Ho, seed=0)
    errs = []
    for i in blocks:
        win = np.zeros(len(u), bool); win[i - Hist:i] = True
        m.tr = win
        for _ in range(k_epochs):
            m.step()
        pred = m.predict_all()
        errs.append(nrmse(y[i:i + W], pred[i:i + W]))
    return np.array(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=9000)
    ap.add_argument("--W", type=int, default=250)
    ap.add_argument("--Hist", type=int, default=1000)
    ap.add_argument("--Ho", type=int, default=24)
    ap.add_argument("--kepochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="out/reservoir_drift")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    switches = [args.T // 3, 2 * args.T // 3]
    start = args.Hist
    blocks = list(range(start, args.T - args.W, args.W))
    bt = np.array([b + args.W / 2 for b in blocks])

    curves = {k: [] for k in ("NG-RC", "LMU(A)", "Ours(B)")}
    for sd in range(args.seeds):
        u = np.random.default_rng(sd).uniform(-1, 1, args.T)
        y = drift_target(u, switches)
        curves["NG-RC"].append(rolling_ngrc(u, y, blocks, args.W, args.Hist))
        curves["LMU(A)"].append(rolling_map(u, y, blocks, args.W, args.Hist,
                                            "lmu", args.Ho, args.kepochs))
        curves["Ours(B)"].append(rolling_map(u, y, blocks, args.W, args.Hist,
                                             "ours", args.Ho, args.kepochs))
        print(f"seed {sd} done", flush=True)

    # steady-state = blocks at least `settle` after a switch
    settle = 3 * args.W
    sw_arr = np.array(switches)
    steady = np.array([min(abs(b - sw_arr).min(), b - start) >= settle
                       for b in blocks])
    print("\n=== mean steady-state NRMSE (excl. post-switch transient) ===")
    stats = {}
    for k, cs in curves.items():
        M = np.stack(cs).mean(0)
        stats[k] = float(M[steady].mean())
        print(f"  {k:8s}: {stats[k]:.3f}")

    STYLE = {"NG-RC": "#55A868", "LMU(A)": "#4C72B0", "Ours(B)": "#C44E52"}
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, cs in curves.items():
        M = np.stack(cs).mean(0); S = np.stack(cs).std(0)
        ax.plot(bt, M, "-", color=STYLE[k], label=f"{k} (steady {stats[k]:.2f})")
        ax.fill_between(bt, M - S, M + S, color=STYLE[k], alpha=0.15)
    for s in switches:
        ax.axvline(s, color="k", ls="--", alpha=0.5)
    ax.text(switches[0], ax.get_ylim()[1] * 0.97, "  regime switch",
            fontsize=8, va="top")
    ax.set(xlabel="stream time (steps)", ylabel="rolling test NRMSE (per block)",
           title=f"(c1) online concept drift — feature adaptation vs read-out-only "
                 f"(Ho={args.Ho}, {args.seeds} seeds)")
    ax.grid(alpha=0.25); ax.legend(fontsize=9)
    fig.tight_layout()
    fp = os.path.join(args.out, "drift.png")
    fig.savefig(fp, dpi=130)
    print(f"saved -> {fp}")


if __name__ == "__main__":
    main()
