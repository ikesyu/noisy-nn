"""
fncl5_2_fig.py — 論文 §5.2 の合成図 (subfigure) を curves_preds.npz から生成する.

fncl5_2.py の本番実行後に走らせる (再学習は不要):
  python tmp/fncl5_2_fig.py                # tmp/ から実行
  python tmp/fncl5_2_fig.py --data out/fncl5_2 --formats png,pdf
  python tmp/fncl5_2_fig.py --data out/fncl6_2   # §6.2 (uniform) にも使える

生成物 (--data と同じディレクトリ):
  fig_main_result.png / .pdf
    (a) 学習曲線 (本文 5 手法, log MSE)
    (b) fit check: target vs backprop / cov_jac (Adam) / cov_jac_full (Adam)
    (c) (b) の残差
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 本文 §5.1 の 5 手法 (npz キー -> 論文表記). 色は Okabe-Ito (CVD-safe).
MAIN_METHODS = [
    ("backprop",          "backprop",          "#000000"),
    ("cov_only",          "cov_only",          "#E69F00"),
    ("cov_deriv_kde",     "cov_deriv",         "#56B4E9"),
    ("cov_jac_adam",      "cov_jac",           "#0072B2"),
    ("cov_jac_full_adam", "cov_jac_full",      "#CC79A7"),
]
# fit check に重ねる手法 (backprop 級であることを示す)
FIT_METHODS = ["backprop", "cov_jac_adam", "cov_jac_full_adam"]

COLOR = {k: c for k, _, c in MAIN_METHODS}
LABEL = {k: l for k, l, _ in MAIN_METHODS}


def main() -> None:
    p = argparse.ArgumentParser(description="compose §5.2 main-result figure")
    p.add_argument("--data", type=str, default="out/fncl5_2",
                   help="curves_preds.npz のあるディレクトリ (出力先も同じ)")
    p.add_argument("--formats", type=str, default="png,pdf")
    args = p.parse_args()

    data_dir = Path(args.data)
    d = np.load(data_dir / "curves_preds.npz")
    x, target = d["x_raw"], d["target"]

    fig = plt.figure(figsize=(9.0, 3.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[2.2, 1.0],
                          hspace=0.12, wspace=0.28,
                          left=0.075, right=0.99, top=0.92, bottom=0.16)

    # ---- (a) learning curves ----
    ax_a = fig.add_subplot(gs[:, 0])
    for key, label, color in MAIN_METHODS:
        lw = 1.2 if key == "backprop" else 1.4
        ls = "--" if key == "backprop" else "-"
        ax_a.plot(d[f"curve_{key}"], color=color, lw=lw, ls=ls, label=label)
    ax_a.set_yscale("log")
    ax_a.set_xlabel("epoch")
    ax_a.set_ylabel("eval MSE")
    ax_a.legend(frameon=False, fontsize=8, loc="upper right", ncol=2,
                columnspacing=1.0, handlelength=1.6)
    ax_a.grid(alpha=0.25, which="both")
    ax_a.set_title("(a) learning curves", fontsize=10, loc="left")

    # ---- (b) fit check ----
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(x, target, color="0.45", lw=2.6, alpha=0.55, label="target")
    for key in FIT_METHODS:
        ls = "--" if key == "backprop" else "-"
        ax_b.plot(x, d[f"pred_{key}"], color=COLOR[key], lw=1.2, ls=ls,
                  label=LABEL[key])
    ax_b.set_ylabel("output")
    ax_b.tick_params(labelbottom=False)
    ax_b.legend(frameon=False, fontsize=8, ncol=2, loc="upper right",
                columnspacing=1.0, handlelength=1.6)
    ax_b.grid(alpha=0.25)
    ax_b.set_ylim(-1.65, 1.95)
    ax_b.set_title("(b) fit and residuals", fontsize=10, loc="left")

    # ---- (c) residuals ----
    ax_c = fig.add_subplot(gs[1, 1], sharex=ax_b)
    ax_c.axhline(0.0, color="0.45", lw=0.8)
    for key in FIT_METHODS:
        ls = "--" if key == "backprop" else "-"
        ax_c.plot(x, d[f"pred_{key}"] - target, color=COLOR[key], lw=1.0, ls=ls)
    ax_c.set_xlabel("x")
    ax_c.set_ylabel("residual")
    ax_c.grid(alpha=0.25)

    for fmt in args.formats.split(","):
        path = data_dir / f"fig_main_result.{fmt.strip()}"
        fig.savefig(path, dpi=200)
        print(f"  saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
