"""
fncl_appB_fig.py — Appendix B の図を実行済みデータ (out/fncl_appB/results.json)
から再生成する。学習の再実行は不要。

results.json には alpha 掃引の最終 MSE と CRN 退化率がすべて入っているため、
  fig_gate_mse_vs_alpha.png/.pdf, fig_crn_degeneracy.png/.pdf
を追加データなしで描き直せる。描画関数は fncl_appB.py 本体からも import される。

実行例:
  python tmp/fncl_appB_fig.py              # 既定: --data out/fncl_appB
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 実験固有の部分は tmp/fncl_driver.py を使う。
from fncl_driver import savefig  # noqa: E402


def fig_gate_mse_vs_alpha(alphas, base, gate_mse, crn_mse, path) -> None:
    fig = plt.figure(figsize=(7, 5))
    plt.axhline(base, color="k", ls="--", lw=1.2,
                label=f"cov_deriv_kde (baseline, {base:.3f})")
    plt.plot(alphas, gate_mse, "o-", label="cov_deriv_gate")
    plt.plot(alphas, crn_mse, "s-", label="cov_deriv_gate_crn (2x forwards)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("perturbation strength alpha")
    plt.ylabel("final MSE (log)")
    plt.title("External node perturbation fails on the binary crossing")
    plt.grid(alpha=0.3)
    plt.legend()
    fig.tight_layout()
    savefig(fig, path)


def fig_crn_degeneracy(alphas, fracs_1, fracs_b, block_size, path) -> None:
    fig = plt.figure(figsize=(6.5, 4.5))
    plt.plot(alphas, fracs_1, "o-", label="single perturbed unit per layer")
    plt.plot(alphas, fracs_b, "s-",
             label=f"block of {block_size} (as in the method)")
    plt.xscale("log")
    plt.ylim(0.0, 1.05)
    plt.xlabel("perturbation strength alpha")
    plt.ylabel("P[ L(+xi) - L(-xi) == 0 ]  (exact)")
    plt.title("Pathwise degeneracy of the binary crossing under CRN pairs")
    plt.grid(alpha=0.3)
    plt.legend()
    fig.tight_layout()
    savefig(fig, path)


def plot_negative_composite(alphas, base, gate_mse, crn_mse, fracs_1, fracs_b,
                            block_size, path) -> None:
    """論文用合成図: (a) MSE vs alpha + (b) CRN 退化率."""
    fig = plt.figure(figsize=(10.0, 3.9))
    gs = fig.add_gridspec(1, 2, wspace=0.28, left=0.1, right=0.99,
                          top=0.89, bottom=0.15)
    ax = fig.add_subplot(gs[0, 0])
    ax.axhline(base, color="k", ls="--", lw=1.2,
               label=f"cov_deriv (baseline, {base:.3f})")
    ax.plot(alphas, gate_mse, "o-", color="#E69F00", label="perturbation gate")
    ax.plot(alphas, crn_mse, "s-", color="#0072B2",
            label="CRN gate (2x forwards)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"perturbation strength $\alpha$")
    ax.set_ylabel("final MSE")
    ax.set_title("(a) external node perturbation", fontsize=10, loc="left")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(alphas, fracs_1, "o-", color="#0072B2",
            label="single perturbed unit per layer")
    ax.plot(alphas, fracs_b, "s-", color="#E69F00",
            label=f"block of {block_size} (as in the method)")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"perturbation strength $\alpha$")
    ax.set_ylabel(r"$P[\,L(+\xi) - L(-\xi) = 0\,]$")
    ax.set_title("(b) pathwise degeneracy under CRN pairs", fontsize=10,
                 loc="left")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    savefig(fig, path)


def main() -> None:
    p = argparse.ArgumentParser(
        description="redraw Appendix B figures from saved data")
    p.add_argument("--data", type=str, default="out/fncl_appB")
    args = p.parse_args()
    d = Path(args.data)
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    alphas = res["alphas"]
    mse = res["mse"]
    base = mse["cov_deriv_kde"]
    gate = [mse[f"cov_deriv_gate/alpha={a}"] for a in alphas]
    crn = [mse[f"cov_deriv_gate_crn/alpha={a}"] for a in alphas]
    fig_gate_mse_vs_alpha(alphas, base, gate, crn,
                          d / "fig_gate_mse_vs_alpha.png")

    block = res["config"]["gate_block_size"]
    zf = res["crn_zero_fraction"]
    fracs_1 = [zf["block_1"][str(a)] for a in alphas]
    fracs_b = [zf[f"block_{block}"][str(a)] for a in alphas]
    fig_crn_degeneracy(alphas, fracs_1, fracs_b, block,
                       d / "fig_crn_degeneracy.png")
    plot_negative_composite(alphas, base, gate, crn, fracs_1, fracs_b, block,
                            d / "fig_negative.png")


if __name__ == "__main__":
    main()
