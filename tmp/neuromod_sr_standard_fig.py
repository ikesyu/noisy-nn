"""Figures for the standard-benchmark SR survey (tmp/neuromod_sr_standard.py).

Fig A (acute):   behaviour and capability vs test-time concentration, sample vs
                 analytic control.  The mechanism-specific signature is the LEFT
                 ARM: sample collapses to exactly zero (participation gate),
                 the mean-field control degrades but never dies.
Fig B (chronic): capability/behaviour vs TRAINING level.  Near-flat: adaptation
                 absorbs the level (gauge compensation; tolerance), leaving only
                 the absolute floor at very low sigma.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

OUT = Path("tmp/out/sr_standard")
COLS = ("x", "separation", "signal", "task_err", "foods_per_1k",
        "night_home_rate", "shelter_frac", "mean_speed", "stall_frac",
        "wall_frac", "contact_frac", "d_threat_min", "diet_evenness",
        "den_evenness", "path_len")
IDX = {c: i for i, c in enumerate(COLS)}
FONT = {"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
        "legend.fontsize": 10}


def load(pattern):
    files = sorted(glob.glob(str(OUT / pattern)))
    if not files:
        return None
    return np.stack([np.loadtxt(f, delimiter=",") for f in files])


def band(ax, x, A, col, color, label):
    m, s = A[:, :, IDX[col]].mean(0), A[:, :, IDX[col]].std(0)
    ax.plot(x, m, color=color, lw=2, label=label)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.18)


def figure(sample, analytic, xlabel, tag, logx=True):
    plt.rcParams.update(FONT)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    x = sample[0, :, 0]
    panels = [("foods_per_1k", "foraging (foods / 1000 frames)"),
              ("night_home_rate", "sheltering (night-home rate)"),
              ("task_err", "task error (pure-state MSE)")]
    for ax, (col, title) in zip(axes, panels):
        band(ax, x, sample, col, "tab:blue", "sample (mechanism)")
        if analytic is not None:
            band(ax, analytic[0, :, 0], analytic, col, "tab:orange",
                 "analytic (mean field)")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    path = OUT / f"fig_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def figure_grid(G):
    """The decisive 2D figure: dose-response rows collapse in RELATIVE dose."""
    plt.rcParams.update(FONT)
    M = G.mean(0)   # cols: st, c, s_test, sep, signal, err, foods(6), home(7), ...
    sts = sorted(set(M[:, 0]))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(sts)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, xcol, xlabel, title in (
            (axes[0], 1, "acute dose  c = sigma_test / sigma_train",
             "relative axis: one universal curve"),
            (axes[1], 2, "absolute test intensity  sigma_test",
             "absolute axis: curves shift with adaptation")):
        for st, col in zip(sts, cmap):
            rows = M[M[:, 0] == st]
            rows = rows[rows[:, xcol].argsort()]
            ax.plot(rows[:, xcol], rows[:, 6], color=col, lw=2, marker="o",
                    ms=3.5, label=f"adapted at {st:g}")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("foods / 1000 frames")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    path = OUT / "fig_grid.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def main():
    beh_s = load("behavior_seed*.csv")
    beh_a = load("behavior_analytic_seed*.csv")
    tr_s = load("train_seed*.csv")
    tr_a = load("train_analytic_seed*.csv")
    if beh_s is not None:
        figure(beh_s, beh_a, "concentration c (acute, trained at c=1)", "acute")
        # gauge ratio at the behavioural optimum
        foods = beh_s[:, :, IDX["foods_per_1k"]].mean(0)
        c = beh_s[0, :, 0]
        print(f"acute optimum: c*={c[foods.argmax()]:.2f}  "
              f"h/sigma* = 0.2/(0.8*{c[foods.argmax()]:.2f}) = "
              f"{0.2/(0.8*c[foods.argmax()]):.3f}")
    if tr_s is not None:
        figure(tr_s, tr_a, "training level sigma_train (chronic)", "chronic")
    G = load("grid_seed*.csv")
    if G is not None:
        figure_grid(G)


if __name__ == "__main__":
    main()
