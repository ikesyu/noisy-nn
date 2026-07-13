"""
fncl5_4_fig.py — §5.4 の図 (アブレーション棒グラフ) を実行済みデータ
(out/fncl5_4/results.json) から再生成する。学習の再実行は不要。

実行例:
  python tmp/fncl5_4_fig.py                # 既定: --data out/fncl5_4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fncl_common import bar_mse

# 論文用ラベル (本文 §5 の表記に合わせる)。ここに無い run は図から除外する
# (cov_jac_adam/no-track は Appendix E の素材で, §5 では説明しない)。
PAPER_LABELS = {
    "backprop":                 "backprop",
    "cov_deriv_kde/per_input":  "cov_deriv (per-input)",
    "cov_deriv_kde/pooled":     "cov_deriv (pooled)",
    "cov_deriv_analytic":       "cov_deriv (analytic $\\phi'$)",
    "cov_only":                 "cov_only",
    "cov_deriv_kde/adam":       "cov_deriv (Adam)",
    "cov_jac_sgd/track":        "cov_jac (SGD)",
    "cov_jac_adam/track":       "cov_jac (Adam)",
}


def plot_ablation(mse: dict, seeds, path) -> None:
    """§5 で説明済みの run のみを本文表記のラベルで棒グラフにする."""
    shown = {label: mse[name] for name, label in PAPER_LABELS.items()
             if name in mse}
    bar_mse(shown, seeds, "Ablations (final MSE)", path)


def main() -> None:
    p = argparse.ArgumentParser(
        description="redraw §5.4 ablation bar chart from saved data")
    p.add_argument("--data", type=str, default="out/fncl5_4")
    args = p.parse_args()
    d = Path(args.data)
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    seeds = res["config"]["seed_list"]
    mse = {name: {int(s): v for s, v in per.items()}
           for name, per in res["final_mse"].items()}
    plot_ablation(mse, seeds, d / "fig_ablation_bar.png")


if __name__ == "__main__":
    main()
