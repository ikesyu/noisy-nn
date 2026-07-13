"""
plot_crossing_figure.py — 論文用の交差活性図 (examples/plot_crossing.py のミニマル版).

ガウスノイズ (std=s) のもとでの交差活性の
  (左) 期待応答      phi_bar(d) = 4 P(1-P)
  (右) 局所微分      phi_bar'(d) = 4 (1-2P) p
を、代表の 1 本の曲線だけで描く。軸・曲線を太くし、目盛り数字・凡例・
グリッド・枠線といった余計な要素をすべて省いた、論文掲載用の図を出力する。

生成物 (tmp/out/crossing/):
  fig_crossing.png / .pdf

実行例:
  python tmp/plot_crossing_figure.py            # tmp/ から実行
  python tmp/plot_crossing_figure.py --sigma 1.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import norm  # noqa: E402

LINEWIDTH = 4.0        # 曲線
AXISWIDTH = 3.0        # 軸 (原点を通る十字)


def crossing(x, s):
    """交差活性の期待応答 phi_bar とその局所微分 phi_bar' (ガウスノイズ std=s)."""
    P = norm.cdf(x, loc=0.0, scale=s)
    p = norm.pdf(x, loc=0.0, scale=s)
    phi = 4.0 * P * (1.0 - P)
    dphi = 4.0 * (1.0 - 2.0 * P) * p
    return phi, dphi


def bare_axis(ax, x, y, color):
    """曲線 1 本と原点を通る太い十字軸だけを描き、他の装飾をすべて消す."""
    xr = 1.08 * np.abs(x).max()
    yr = 1.15 * np.abs(y).max()
    ax.axhline(0.0, color="black", lw=AXISWIDTH, zorder=1)
    ax.axvline(0.0, color="black", lw=AXISWIDTH, zorder=1)
    ax.plot(x, y, color=color, lw=LINEWIDTH, solid_capstyle="round", zorder=2)
    ax.set_xlim(-xr, xr)
    ax.set_ylim(-yr, yr)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("auto")


def main() -> None:
    p = argparse.ArgumentParser(description="minimal crossing-activation figure")
    p.add_argument("--sigma", type=float, default=1.0, help="ガウスノイズ std")
    p.add_argument("--out", type=str, default="out/crossing")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-3.0, 3.0, 400)
    phi, dphi = crossing(x, args.sigma)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    bare_axis(axes[0], x, phi, "#0072B2")     # 期待応答
    bare_axis(axes[1], x, dphi, "#CC6788")    # 局所微分
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.12)

    for ext in ("png", "pdf"):
        path = out / f"fig_crossing.{ext}"
        fig.savefig(path, dpi=200, transparent=True)
        print(f"  saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
