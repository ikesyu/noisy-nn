"""
fncl5_5_fig.py — §5.5 の図を実行済みデータ (out/fncl5_5/fig_data.npz) から
再生成する。学習の再実行は不要。

fncl5_5.py 本体が保存する fig_data.npz (学習曲線・バイアス検証の per-input
配列・相関) のみを読み、
  fig_readout_drift.png/.pdf, fig_bias_scatter.png/.pdf
を描き直す。描画関数は fncl5_5.py 本体からも import される (図の単一定義)。

実行例:
  python tmp/fncl5_5_fig.py                # 既定: --data out/fncl5_5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fncl_common import savefig  # noqa: E402


STYLES = {"backprop": ("k", "-"), "cov_jac_adam": ("tab:blue", "-"),
          "full_cov_adam": ("tab:red", "-"),
          "full_cov_m3_adam": ("tab:green", "-"),
          "full_probe_adam": ("tab:purple", "-"),
          "cov_jac_sgd": ("tab:blue", ":"), "full_cov_sgd": ("tab:red", ":"),
          "full_cov_m3_sgd": ("tab:green", ":"),
          "full_probe_sgd": ("tab:purple", ":")}


def fig_drift(curves: dict, path) -> None:
    fig = plt.figure(figsize=(7.5, 5))
    for name, curve in curves.items():
        c, ls = STYLES.get(name, ("gray", "-"))
        plt.semilogy(curve, color=c, linestyle=ls, lw=1.5, label=name)
    plt.xlabel("epoch")
    plt.ylabel("MSE (eval, log scale)")
    plt.title("Readout-error estimators: raw Cov(L,y) drifts under Adam;\n"
              "m3-corrected / probe stay at backprop level")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def fig_bias_scatter(bias: np.ndarray, skew_term: np.ndarray, corr: float,
                     path) -> None:
    fig = plt.figure(figsize=(5.5, 5.0))
    lim = 1.05 * max(np.abs(bias).max(), np.abs(skew_term).max())
    plt.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="y = x")
    plt.scatter(skew_term, bias, s=14, alpha=0.7, color="tab:red")
    plt.xlabel("predicted skew term  m3 / Var  (per input)")
    plt.ylabel("observed bias  E[g_y] - 2(E[y] - t)")
    plt.title(f"Readout covariance bias vs skewness term  (corr = {corr:.3f})")
    plt.legend()
    plt.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def plot_readout_composite(curves: dict, bias: np.ndarray,
                           skew_term: np.ndarray, corr: float, path) -> None:
    """論文用合成図: (a) ドリフト学習曲線 + (b) 歪度バイアス散布図."""
    fig = plt.figure(figsize=(11.0, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.27,
                          left=0.06, right=0.99, top=0.9, bottom=0.14)
    ax = fig.add_subplot(gs[0, 0])
    for name, curve in curves.items():
        c, ls = STYLES.get(name, ("gray", "-"))
        ax.semilogy(curve, color=c, linestyle=ls, lw=1.2, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("eval MSE")
    ax.set_title("(a) readout-error estimators", fontsize=10, loc="left")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    ax = fig.add_subplot(gs[0, 1])
    lim = 1.05 * max(np.abs(bias).max(), np.abs(skew_term).max())
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="y = x")
    ax.scatter(skew_term, bias, s=12, alpha=0.7, color="#0072B2")
    ax.set_xlabel("predicted skew term  $m_3/\\mathrm{Var}$  (per input)")
    ax.set_ylabel("observed bias")
    ax.set_title(f"(b) skewness bias (corr = {corr:.3f})", fontsize=10,
                 loc="left")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    savefig(fig, path)


def main() -> None:
    p = argparse.ArgumentParser(
        description="redraw §5.5 figures from saved data (no re-training)")
    p.add_argument("--data", type=str, default="out/fncl5_5")
    args = p.parse_args()
    d = Path(args.data)
    npz = d / "fig_data.npz"
    if not npz.exists():
        raise SystemExit(f"{npz} が無い。fncl5_5.py を一度実行して生成すること。")
    z = np.load(npz)
    curves = {k[len("curve_"):]: z[k] for k in z.files if k.startswith("curve_")}
    fig_drift(curves, d / "fig_readout_drift.png")
    fig_bias_scatter(z["bias"], z["skew_term"], float(z["corr"]),
                     d / "fig_bias_scatter.png")
    plot_readout_composite(curves, z["bias"], z["skew_term"], float(z["corr"]),
                           d / "fig_readout.png")


if __name__ == "__main__":
    main()
