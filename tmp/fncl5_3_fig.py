"""
fncl5_3_fig.py — §5.3 の図を実行済みデータ (out/fncl5_3/) から再生成する.

fncl5_3.py 本体が保存する
  results.json  (mirror の Pearson r / gradient 忠実度の cos・ratio)
  fig_data.npz  (mirror 散布図用の W_hat / W 配列; 未学習状態)
のみを読み、学習の再実行なしに
  fig_mirror_scatter.png/.pdf, fig_grad_cosine.png/.pdf
を描き直す。レイアウト調整はこのファイルの編集と再実行だけで済む。
描画関数は fncl5_3.py 本体からも import される (図の単一定義)。

実行例:
  python tmp/fncl5_3_fig.py                # 既定: --data out/fncl5_3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fncl_common import savefig  # noqa: E402


def plot_mirror_scatter(pairs, path):
    """pairs: list of (title, w_hat, w_true, r). 配列は numpy を想定."""
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 4.0))
    for ax, (title, w_hat, w_true, r) in zip(np.atleast_1d(axes), pairs):
        a = np.asarray(w_true).ravel()
        b = np.asarray(w_hat).ravel()
        lim = max(np.abs(a).max(), np.abs(b).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5)
        ax.scatter(a, b, s=8, alpha=0.6)
        ax.set_xlabel("true W")
        ax.set_ylabel("mirror W_hat")
        ax.set_title(f"{title}\nPearson r = {r:.4f}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def plot_grad_cosine(fid, path):
    """fid[state][estimator][layer] = {"cos":, "ratio":} の層別棒グラフ."""
    states = list(fid.keys())
    layers = ["w0", "w1", "wout"]
    fig, axes = plt.subplots(1, len(states), figsize=(5.2 * len(states), 4.0),
                             sharey=True)
    width = 0.35
    xpos = np.arange(len(layers))
    for ax, state in zip(np.atleast_1d(axes), states):
        for k, est in enumerate(("cov_jac", "cov_deriv")):
            vals = [fid[state][est][l]["cos"] for l in layers]
            ax.bar(xpos + (k - 0.5) * width, vals, width, label=est)
            for xp, v in zip(xpos + (k - 0.5) * width, vals):
                ax.text(xp, min(v + 0.02, 1.05), f"{v:.3f}", ha="center",
                        fontsize=7)
        ax.axhline(1.0, color="k", lw=0.6, alpha=0.5)
        ax.set_xticks(xpos)
        ax.set_xticklabels(layers)
        ax.set_ylim(0.0, 1.15)
        ax.set_title(f"cosine vs autograd gradient ({state})")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("cosine similarity")
    fig.tight_layout()
    savefig(fig, path)


def plot_fidelity_composite(pairs, fid, path):
    """論文用合成図: (a)(b) mirror 散布図 (隠れ層 / 読み出し層) + (c) 勾配 cosine.

    pairs: [(title, w_hat, w_true, r)] の 2 要素。fid は results.json["gradient"]。
    """
    fig = plt.figure(figsize=(11.5, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.45], wspace=0.33,
                          left=0.06, right=0.99, top=0.88, bottom=0.15)
    for i, (title, w_hat, w_true, r) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, i])
        a = np.asarray(w_true).ravel()
        b = np.asarray(w_hat).ravel()
        lim = max(np.abs(a).max(), np.abs(b).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5)
        ax.scatter(a, b, s=8, alpha=0.6, color="#0072B2")
        ax.set_xlabel("true $W$")
        ax.set_ylabel(r"mirror $\hat{W}$")
        ax.set_title(f"({chr(97 + i)}) {title} ($r$ = {r:.4f})",
                     fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[0, 2])
    layers = ["w0", "w1", "wout"]
    xpos = np.arange(len(layers))
    width = 0.19
    series = [("cov_jac", "untrained"), ("cov_jac", "pretrained"),
              ("cov_deriv", "untrained"), ("cov_deriv", "pretrained")]
    colors = {"cov_jac": "#0072B2", "cov_deriv": "#E69F00"}
    for k, (est, state) in enumerate(series):
        vals = [fid[state][est][l]["cos"] for l in layers]
        ax.bar(xpos + (k - 1.5) * width, vals, width, color=colors[est],
               alpha=1.0 if state == "untrained" else 0.5,
               label=f"{est} ({state})")
    ax.axhline(1.0, color="k", lw=0.6, alpha=0.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(layers)
    vmin = min(0.0, min(fid[state][est][l]["cos"]
                        for est, state in series for l in layers))
    ax.set_ylim(vmin - 0.02, 1.14)
    ax.set_ylabel("cosine vs autograd grad.")
    ax.set_title("(c) update-direction fidelity", fontsize=10, loc="left")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, ncol=2)
    savefig(fig, path)


def main() -> None:
    p = argparse.ArgumentParser(
        description="redraw §5.3 figures from saved data (no re-training)")
    p.add_argument("--data", type=str, default="out/fncl5_3")
    args = p.parse_args()
    d = Path(args.data)
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))

    plot_grad_cosine(res["gradient"], d / "fig_grad_cosine.png")

    npz = d / "fig_data.npz"
    if npz.exists():
        z = np.load(npz)
        m = res["mirror"]["untrained"]
        plot_mirror_scatter(
            [("hidden W (corr. with d)", z["w1_hat"], z["w1_true"],
              m["r_hidden_d"]),
             ("readout W", z["wout_hat"], z["wout_true"], m["r_readout"]),
             ("hidden W (corr. with binary z)", z["w1_bin"], z["w1_true"],
              m["r_hidden_binary_z"])],
            d / "fig_mirror_scatter.png")
        plot_fidelity_composite(
            [("hidden $W$", z["w1_hat"], z["w1_true"], m["r_hidden_d"]),
             ("readout $W$", z["wout_hat"], z["wout_true"], m["r_readout"])],
            res["gradient"], d / "fig_fidelity.png")
    else:
        print(f"  {npz} が無いため散布図・合成図はスキップ "
              "(fncl5_3.py の再実行で生成)")


if __name__ == "__main__":
    main()
