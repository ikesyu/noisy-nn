"""
fncl5_6.py — 論文 §5.6「sin(x) 以外のベンチマークでの一般化」

sin(x) 回帰 (§5.2) と同一プロトコル (H=64, T=64, 1500 epoch, 同一初期重み,
オプティマイザは §5.1 の適合選択) のまま、タスクだけを差し替えて本文 5 手法を
比較する。現行の学習コード (train_cov) は読み出しがスカラー 1 出力である
ことを前提とするため、ベンチマークはスカラー出力のものを選ぶ:

  friedman1 : Friedman #1 回帰 (5 次元入力の古典的ベンチマーク,
              y = 10 sin(pi x1 x2) + 20 (x3-.5)^2 + 10 x4 + 5 x5 を [-1,1] に正規化)
  moons     : two moons 二値分類 (2 次元, 目標 ±1 の MSE 学習, accuracy も報告)
  circles   : 同心円 二値分類 (2 次元, 同上)

データセットは固定 rng (seed 12345) で 1 回だけ生成し、seed 0,1,2 は
初期重み・学習中のノイズのみを変える (§5.2 と同じ流儀)。

生成物 (out/fncl5_6/):
  table_benchmarks.md  -> タスク x 手法の最終 MSE (mean±std) + 分類 accuracy
  results.json         -> 数値一式
  curves_preds_<task>.npz -> 学習曲線・予測 (先頭 seed; 図が必要になった場合用)

実行例:
  python tmp/fncl5_6.py                # 本番 (3 seeds x 3 tasks x 5 手法)
  python tmp/fncl5_6.py --quick        # 動作確認
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from fncl_common import (add_common_args, finalize_args, config_dict, run_method,
                         write_text, save_json)
import forward_noise_covariance_learning as fncl
from nnn import model as nnn_model

DATA_SEED = 12345
N_POINTS = 128

# 本文 §5.1 の 5 手法 (オプティマイザは誤差の性質に適合させた既定の組)
BENCH_METHODS = [
    ("backprop",          {"kind": "backprop"}),
    ("cov_only",          {"method": "cov_only"}),
    ("cov_deriv_kde",     {"method": "cov_deriv", "slope": "kde"}),
    ("cov_jac_adam",      {"method": "cov_jac", "opt": "adam", "jac_track": True}),
    ("cov_jac_full_adam", {"method": "cov_jac_full", "opt": "adam",
                           "jac_track": True, "jac_out": "cov_m3"}),
]


# ============================================================
# タスク生成 (すべてスカラー出力; 入力は ~[-2, 2] に正規化)
# ============================================================
def make_friedman1(rng: np.random.Generator):
    X = rng.uniform(0.0, 1.0, (N_POINTS, 5))
    y = (10.0 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20.0 * (X[:, 2] - 0.5) ** 2
         + 10.0 * X[:, 3] + 5.0 * X[:, 4])
    x = 4.0 * (X - 0.5)                                   # [-2, 2]
    y = 2.0 * (y - y.min()) / (y.max() - y.min()) - 1.0   # [-1, 1]
    return x.astype(np.float32), y.astype(np.float32), "regression"


def make_moons(rng: np.random.Generator, noise: float = 0.10):
    m = N_POINTS // 2
    th0 = rng.uniform(0.0, np.pi, m)
    th1 = rng.uniform(0.0, np.pi, N_POINTS - m)
    x0 = np.stack([np.cos(th0), np.sin(th0)], axis=1)
    x1 = np.stack([1.0 - np.cos(th1), 0.5 - np.sin(th1)], axis=1)
    X = np.concatenate([x0, x1]) + rng.normal(0.0, noise, (N_POINTS, 2))
    y = np.concatenate([-np.ones(m), np.ones(N_POINTS - m)])
    X = (X - X.mean(axis=0)) / X.std(axis=0) * 1.2        # ~[-2, 2]
    return X.astype(np.float32), y.astype(np.float32), "classification"


def make_circles(rng: np.random.Generator, noise: float = 0.08):
    m = N_POINTS // 2
    th0 = rng.uniform(0.0, 2.0 * np.pi, m)
    th1 = rng.uniform(0.0, 2.0 * np.pi, N_POINTS - m)
    x0 = np.stack([np.cos(th0), np.sin(th0)], axis=1)
    x1 = 0.5 * np.stack([np.cos(th1), np.sin(th1)], axis=1)
    X = np.concatenate([x0, x1]) + rng.normal(0.0, noise, (N_POINTS, 2))
    y = np.concatenate([-np.ones(m), np.ones(N_POINTS - m)])
    X = X * 1.6                                           # ~[-2, 2]
    return X.astype(np.float32), y.astype(np.float32), "classification"


TASKS = {"friedman1": make_friedman1, "moons": make_moons, "circles": make_circles}


# ============================================================
# 多次元入力のモデル (fncl.build_model の 1 次元 bump タイル初期化を使わない版)
# ============================================================
def model_factory_nd(noise: str, in_dim: int, args, device: torch.device):
    structure = [in_dim, args.hidden_dim, args.hidden_dim, 1]

    def build():
        if noise == "gaussian":
            net = nnn_model.SimpleNNNSample(structure=structure, std=args.sigma,
                                            h=args.crossing_h, t=args.num_samples,
                                            output_bias=True)
        else:
            net = nnn_model.SimpleNNNUniformSample(
                structure=structure, radius=args.radius, center=fncl.UNIFORM_CENTER,
                h=args.crossing_h, t=args.num_samples, output_bias=True)
        net.noise_field = [torch.ones(args.hidden_dim, device=device)
                           for _ in range(len(structure) - 2)]
        return net.to(device)

    net0 = build()
    init_state = copy.deepcopy(net0.state_dict())

    def fresh():
        n = build()
        n.load_state_dict(init_state)
        return n

    return fresh


# ============================================================
# 実行・集計
# ============================================================
def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.sign(pred) == np.sign(target)))


def main() -> None:
    p = argparse.ArgumentParser(
        description="§5.6 generalization: 5 methods on simple benchmarks "
                    "beyond sin(x).")
    add_common_args(p)
    p.add_argument("--noise", choices=("gaussian", "uniform"), default="gaussian")
    p.add_argument("--tasks", type=str, default=",".join(TASKS),
                   help="カンマ区切りタスク名")
    args = finalize_args(p.parse_args(), default_out="out/fncl5_6")
    device = torch.device(args.device)
    log_every = max(1, args.epochs // 5)

    results = {}         # results[task][method][seed] = {"mse": ..., "acc": ...}
    for task_name in args.tasks.split(","):
        x_np, t_np, kind = TASKS[task_name](np.random.default_rng(DATA_SEED))
        x = torch.tensor(x_np, device=device)
        t = torch.tensor(t_np, device=device).unsqueeze(1)
        results[task_name] = {"kind": kind, "per_method": {}}
        curves, preds = {}, {}
        for seed in args.seed_list:
            for name, spec in BENCH_METHODS:
                torch.manual_seed(seed)
                np.random.seed(seed)
                fresh = model_factory_nd(args.noise, x_np.shape[1], args, device)
                losses, pred, _ = run_method(spec, fresh, x, t, args.noise, args,
                                             log_every)
                entry = {"mse": float(np.mean((pred - t_np) ** 2))}
                if kind == "classification":
                    entry["acc"] = accuracy(pred, t_np)
                results[task_name]["per_method"].setdefault(name, {})[seed] = entry
                extra = (f"  acc = {entry['acc']:.3f}"
                         if kind == "classification" else "")
                print(f"[{task_name} | seed {seed}] {name:20s} "
                      f"final MSE = {entry['mse']:.5f}{extra}", flush=True)
                if seed == args.seed_list[0]:
                    curves[name] = losses
                    preds[name] = pred
        np.savez(args.out_dir / f"curves_preds_{task_name}.npz",
                 x=x_np, target=t_np,
                 **{f"curve_{k}": np.asarray(v) for k, v in curves.items()},
                 **{f"pred_{k}": np.asarray(v) for k, v in preds.items()})
        print(f"  saved {args.out_dir / f'curves_preds_{task_name}.npz'}")

    # ---- summary table: rows = methods, cols = tasks (MSE mean±std [, acc]) ----
    lines = [f"**Final MSE (mean ± std over seeds {args.seeds}), noise={args.noise}, "
             f"H={args.hidden_dim}, T={args.num_samples}, epochs={args.epochs}, "
             f"lr={args.lr}** (分類タスクは括弧内に accuracy)", ""]
    task_names = list(results.keys())
    lines.append("| 手法 | " + " | ".join(task_names) + " |")
    lines.append("|---" * (len(task_names) + 1) + "|")
    for name, _ in BENCH_METHODS:
        cells = []
        for tn in task_names:
            per_seed = results[tn]["per_method"][name]
            mses = [per_seed[s]["mse"] for s in args.seed_list]
            cell = f"{np.mean(mses):.5f} ± {np.std(mses):.5f}"
            if results[tn]["kind"] == "classification":
                accs = [per_seed[s]["acc"] for s in args.seed_list]
                cell += f" ({np.mean(accs):.3f})"
            cells.append(cell)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_benchmarks.md", table)
    save_json(args.out_dir / "results.json",
              {"config": config_dict(args), "noise": args.noise,
               "data_seed": DATA_SEED, "results": results})


if __name__ == "__main__":
    main()
