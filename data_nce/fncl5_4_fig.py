"""
fncl5_4_fig.py — §5.4 の図 (アブレーション棒グラフ) を実行済みデータ
(out/fncl5_4/results.json) から再生成する。学習の再実行は不要。

実行例:
  python data_nce/fncl5_4_fig.py                # 既定: --data out/fncl5_4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fncl_common import bar_mse


def plot_ablation(mse: dict, seeds, path) -> None:
    """論文 Fig.4: 実験行列の全 run (cov_jac_adam/no-track 含む) を run 名の
    ままの順で棒グラフにする (本文 §5.4.1–5.4.4 がすべての棒に言及する)."""
    bar_mse(mse, seeds, "Ablations (final MSE)", path)


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
