"""Regenerate every draft figure that has no other generating script.

Covers (all written to tmp/out/sr_standard/, then copied manually into
docs/draft_neuromod_behavior/):

    fig_pure.png / fig_pure_control.png   experiment 8 (from pure_*.csv)
    fig_lesion.png                        experiment 9 (from lesion_standard.csv)
    fig_field_layout.png                  region method: 3 fields + overlap
    fig_dist_uniform_gauss.png            distribution method: variance-matched pairs
    fig_dist_bimodal_gauss.png
    fig_dist_candidates.png               third-distribution candidates
    fig_dist_triplet_final.png            final working triplet

The remaining draft figures come from their own drivers:
    fig_ethogram / fig_diversity          tmp/neuromod_ethogram.py
    fig_acute / fig_chronic / fig_grid    tmp/neuromod_sr_standard_fig.py
    fig_shape_axis                        tmp/neuromod_shape_axis.py

Style rules (requested for all document figures): no titles, large fonts,
sparse ticks, legends placed clear of the data.

Run from the repository root:
    .venv/bin/python tmp/neuromod_draft_figs.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))
from neuromod import fields as F, world

OUT = Path("tmp/out/sr_standard")
SEEDS = [7, 11, 23]
S0 = 0.8


# ---------------------------------------------------------------- experiment 8
def load_stack(pattern):
    return np.stack([np.loadtxt(OUT / pattern.format(s), delimiter=",",
                                comments="#") for s in SEEDS])


def fig_pure(tag, conc_pat, interp_pat, out_name):
    conc = load_stack(conc_pat)          # [seed, c, (c, sep, signal, err)]
    interp = load_stack(interp_pat)      # [seed, row, (pair, w, lam, err)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    x = conc[0, :, 0]
    for ax, col, ylab, yticks in ((axes[0], 2, "signal", [0, 0.4, 0.8]),
                                  (axes[1], 3, "task error", [0, 0.15, 0.3])):
        m, sd = conc[:, :, col].mean(0), conc[:, :, col].std(0)
        ax.plot(x, m, color="tab:blue", lw=2.5, marker="o", ms=5)
        ax.fill_between(x, m - sd, m + sd, color="tab:blue", alpha=0.2)
        ax.set_xscale("log")
        ax.set_xlabel("concentration c", fontsize=16)
        ax.set_ylabel(ylab, fontsize=16)
        ax.set_yticks(yticks)
        ax.tick_params(labelsize=14)
        ax.grid(alpha=0.3)
    ax = axes[2]
    for pair, color in ((0, "tab:blue"), (1, "tab:orange"), (2, "tab:green")):
        rows = interp[:, interp[0, :, 0] == pair, :]
        w, lam = rows[0, :, 1], rows[:, :, 2].mean(0)
        sd = rows[:, :, 2].std(0)
        ax.plot(w, lam, color=color, lw=2, marker="o", ms=4)
        ax.fill_between(w, lam - sd, lam + sd, color=color, alpha=0.2)
    ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=1)
    ax.set_xlabel("mixing weight w", fontsize=16)
    ax.set_ylabel(r"implied $\hat\lambda$", fontsize=16)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / out_name, dpi=150)
    plt.close(fig)
    print(f"saved {out_name}")


# ---------------------------------------------------------------- experiment 9
def fig_lesion():
    r = np.loadtxt(OUT / "lesion_standard.csv", delimiter=",", comments="#")
    r = r[r[:, 0] == 0]                          # one-hot condition
    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_names = ["foraging | avoidance", "foraging | sheltering",
                  "avoidance | sheltering"]
    groups = ["shared", "A only", "B only", "random"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    w = 0.36
    for ax, (ca, cb), pname in zip(axes, pairs, pair_names):
        sel = r[(r[:, 2] == ca) & (r[:, 3] == cb)]
        ma = [sel[sel[:, 4] == g][:, 6].mean() for g in range(4)]
        mb = [sel[sel[:, 4] == g][:, 7].mean() for g in range(4)]
        x = np.arange(4)
        ax.bar(x - w / 2, ma, w, color="tab:blue", label="behavior A")
        ax.bar(x + w / 2, mb, w, color="tab:red", label="behavior B")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=13)
        ax.set_xlabel(pname, fontsize=15)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(labelsize=14)
    axes[0].set_ylabel(r"$\Delta$ task error", fontsize=16)
    axes[0].set_yticks([0, 0.03, 0.06])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=14,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig_lesion.png", dpi=150)
    plt.close(fig)
    print("saved fig_lesion.png")


# ------------------------------------------------------------- region method
def fig_field_layout():
    flds = F.build_fields(world.CATEGORIES, 144, S0, 0.22, 0.15)
    names = {"food": "foraging", "threat": "avoidance", "shelter": "sheltering"}
    mats = {c: flds[c].numpy().reshape(12, 12) for c in world.CATEGORIES}
    support = sum((m > 0).astype(int) for m in mats.values())
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.9))
    for ax, c in zip(axes[:3], world.CATEGORIES):
        im = ax.imshow(mats[c], origin="lower", cmap="magma", vmin=0, vmax=S0,
                       extent=[0, 1, 0, 1])
        ax.set_xlabel(names[c], fontsize=17)
        ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes[:3], fraction=0.03, pad=0.03)
    cb.set_ticks([0, 0.4, 0.8])
    cb.ax.tick_params(labelsize=14)
    cb.ax.set_ylabel(r"$\sigma_k$", fontsize=16, rotation=0, labelpad=16,
                     va="center")
    im2 = axes[3].imshow(support, origin="lower", cmap="viridis", vmin=0,
                         vmax=3, extent=[0, 1, 0, 1])
    axes[3].set_xlabel("overlap (support count)", fontsize=17)
    axes[3].set_xticks([]); axes[3].set_yticks([])
    cb2 = fig.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.05)
    cb2.set_ticks([0, 1, 2, 3])
    cb2.ax.tick_params(labelsize=14)
    fig.savefig(OUT / "fig_field_layout.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_field_layout.png")


# ------------------------------------------------------- distribution method
XS = np.linspace(-2.6, 2.6, 1200)
SM = 0.3 * S0                                # component width (0.24)
MU = math.sqrt(S0 ** 2 - SM ** 2)            # bimodal mode position
Q, SS = 0.25, 0.15                           # burst mixture
SB = math.sqrt((S0 ** 2 - (1 - Q) * SS ** 2) / Q)
M_SKEW = math.sqrt((S0 ** 2 - SM ** 2) / 3.0)


def gauss(x, mu, s):
    return np.exp(-(x - mu) ** 2 / (2 * s * s)) / (s * math.sqrt(2 * math.pi))


DENSITIES = {
    "gauss": lambda x: gauss(x, 0, S0),
    "uniform": lambda x: np.where(np.abs(x) <= math.sqrt(3) * S0,
                                  1.0 / (2 * math.sqrt(3) * S0), 0.0),
    "bimodal": lambda x: 0.5 * gauss(x, -MU, SM) + 0.5 * gauss(x, MU, SM),
    "burst": lambda x: Q * gauss(x, 0, SB) + (1 - Q) * gauss(x, 0, SS),
    "laplace": lambda x: np.exp(-np.abs(x) / (S0 / math.sqrt(2)))
                         / (2 * S0 / math.sqrt(2)),
    "skewmix": lambda x: 0.75 * gauss(x, -M_SKEW, SM)
                         + 0.25 * gauss(x, 3 * M_SKEW, SM),
}


def fig_densities(curves, out_name, ylim, yticks):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for kind, color, label in curves:
        y = DENSITIES[kind](XS)
        ax.plot(XS, y, color=color, lw=2.5, label=label)
        ax.fill_between(XS, y, color=color, alpha=0.10)
    ax.set_xlabel("noise value", fontsize=16)
    ax.set_ylabel("probability density", fontsize=16)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks(yticks)
    ax.tick_params(labelsize=14)
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(0, ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / out_name, dpi=150)
    plt.close(fig)
    print(f"saved {out_name}")


def main():
    fig_pure("pure", "pure_conc_seed{}.csv", "pure_interp_seed{}.csv",
             "fig_pure.png")
    fig_pure("graded", "pure_graded_conc_seed{}.csv",
             "pure_graded_interp_seed{}.csv", "fig_pure_control.png")
    fig_lesion()
    fig_field_layout()
    fig_densities([("uniform", "tab:blue", "uniform"),
                   ("gauss", "tab:red", "Gaussian")],
                  "fig_dist_uniform_gauss.png", 1.0, [0, 0.5, 1.0])
    fig_densities([("bimodal", "tab:blue", "bimodal"),
                   ("gauss", "tab:red", "Gaussian")],
                  "fig_dist_bimodal_gauss.png", 1.0, [0, 0.5, 1.0])
    fig_densities([("gauss", "tab:red", "Gaussian (current)"),
                   ("laplace", "tab:green", "Laplace"),
                   ("burst", "tab:purple", "scale mixture (burst)")],
                  "fig_dist_candidates.png", 1.55, [0, 0.5, 1.0, 1.5])
    fig_densities([("burst", "tab:purple", "burst (food)"),
                   ("skewmix", "tab:orange", "skewed (threat)"),
                   ("bimodal", "tab:blue", "bimodal (shelter)")],
                  "fig_dist_triplet_final.png", 1.55, [0, 0.5, 1.0, 1.5])


if __name__ == "__main__":
    main()
