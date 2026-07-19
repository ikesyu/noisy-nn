"""
fncl_common.py — NCE 論文ドラフト (docs/draft_nce.md) §5 以降の実験スクリプト
(fncl5_*.py, fncl6_*.py) の共通基盤.

fncl パッケージ (data_nce/fncl/) のモデル構築・学習ルーチンをそのまま再利用し、
ここでは
  - 実験行列 (seed × 手法) の実行,
  - 最終 MSE 表の標準出力への表示と results.json の保存,
  - 図 (PNG) の保存 (ヘッドレス, Agg バックエンド)
を共通化する。各 fncl5_*.py は「論文のどの図表を作るか」だけを定義する薄い
スクリプトになる。

すべてのスクリプトは --quick で縮小設定の動作確認ができ、生成物は --out
(既定 out/fncl<sec>_<n>/) に書き出される。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 保存専用 (fncl パッケージが pyplot を import する前に設定)
import torch

DATA_DIR = Path(__file__).resolve().parent
if str(DATA_DIR) not in sys.path:
    sys.path.append(str(DATA_DIR))
import fncl  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


# ============================================================
# argparse
# ============================================================
def add_common_args(p: argparse.ArgumentParser, *, epochs: int = 1500,
                    hidden_dim: int = 64, num_samples: int = 64,
                    seeds: str = "0,1,2") -> argparse.ArgumentParser:
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--hidden-dim", type=int, default=hidden_dim)
    p.add_argument("--num-samples", type=int, default=num_samples,
                   help="T: モデルが内部で引く確率サンプル数")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--sigma", type=float, default=0.5, help="ガウスノイズ std")
    p.add_argument("--radius", type=float, default=1.0, help="一様ノイズ半幅")
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--credit", choices=("pooled", "per_input"), default="per_input")
    p.add_argument("--credit-passes", type=int, default=1)
    p.add_argument("--jac-ema", type=float, default=0.9)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seeds", type=str, default=seeds, help="カンマ区切り seed リスト")
    p.add_argument("--out", type=str, default=None, help="出力ディレクトリ")
    p.add_argument("--quick", action="store_true", help="縮小設定での動作確認")
    return p


def finalize_args(args: argparse.Namespace, default_out: str) -> argparse.Namespace:
    if args.quick:
        args.epochs = min(args.epochs, 60)
        args.num_samples = min(args.num_samples, 16)
        args.hidden_dim = min(args.hidden_dim, 16)
        args.seeds = str(args.seeds).split(",")[0]
    args.seed_list = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    args.out_dir = Path(args.out or default_out)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return args


def config_dict(args: argparse.Namespace) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


# ============================================================
# タスク・モデル
# ============================================================
def make_task(device: torch.device):
    """回帰タスク y = sin(x), x in [-2pi, 2pi] (入力は ~[-2, 2] に正規化)."""
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, fncl.NUM_POINTS, dtype=np.float32)
    target = np.sin(x_raw).astype(np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    t = torch.tensor(target, device=device).unsqueeze(1)
    return x_raw, target, x, t


def model_factory(noise: str, args: argparse.Namespace, device: torch.device,
                  field: torch.Tensor = None):
    """同一初期重みのネットワークを繰り返し生成する fresh() を返す."""
    net0 = fncl.build_model(noise, args.hidden_dim, args.sigma, args.radius,
                            args.crossing_h, args.num_samples, device, field=field)
    init_state = copy.deepcopy(net0.state_dict())

    def fresh():
        n = fncl.build_model(noise, args.hidden_dim, args.sigma, args.radius,
                             args.crossing_h, args.num_samples, device, field=field)
        n.load_state_dict(init_state)
        return n

    return fresh


# ============================================================
# 実験行列
# ============================================================
# 検証セット (論文 §5.1): 8 手法。cov_jac 系は --jac-track 相当 ON。
# cov_jac_full は読み出し誤差も共分散から推定する変種 (§4.4/§5.5);
# 既定の jac_out="cov_m3" は歪度バイアスの 3 次モーメント補正 (raw の "cov" は
# Adam でドリフトする — その掃引は fncl5_5.py が担当)。
VERIFICATION_METHODS = [
    ("backprop",           {"kind": "backprop"}),
    ("cov_only",           {"method": "cov_only"}),
    ("cov_deriv_analytic", {"method": "cov_deriv", "slope": "analytic"}),
    ("cov_deriv_kde",      {"method": "cov_deriv", "slope": "kde"}),
    ("cov_jac_sgd",        {"method": "cov_jac", "opt": "sgd",  "jac_track": True}),
    ("cov_jac_adam",       {"method": "cov_jac", "opt": "adam", "jac_track": True}),
    ("cov_jac_full_sgd",   {"method": "cov_jac_full", "opt": "sgd",
                            "jac_track": True, "jac_out": "cov_m3"}),
    ("cov_jac_full_adam",  {"method": "cov_jac_full", "opt": "adam",
                            "jac_track": True, "jac_out": "cov_m3"}),
]


def run_method(spec: dict, fresh, x, t, noise: str, args, log_every: int = 0):
    """spec に従い 1 手法を学習して (losses, pred[np.ndarray], net) を返す.

    spec 例: {"kind": "backprop"} または
             {"method": "cov_jac", "opt": "adam", "jac_track": True, ...}
    method/opt/credit/jac_ema 以外のキー (slope, jac_track, gate_* など) は
    そのまま train_cov に渡される。
    """
    net = fresh()
    spec = dict(spec)
    if spec.pop("kind", None) == "backprop":
        losses = fncl.train_backprop(net, x, t, args.lr, args.epochs, log_every)
    else:
        method = spec.pop("method")
        opt = spec.pop("opt", "sgd")
        credit = spec.pop("credit", args.credit)
        jac_ema = spec.pop("jac_ema", args.jac_ema)
        losses, _ = fncl.train_cov(
            net, x, t, noise, args.sigma, args.radius, method, args.lr, args.epochs,
            credit=credit, credit_passes=args.credit_passes, opt=opt,
            lr_decay="none", log_every=log_every, jac_ema=jac_ema, **spec)
    pred = fncl.predict(net, x)
    return losses, pred, net


def run_matrix(methods, noise: str, args, log_every: int = 0):
    """seed × 手法 の実験行列を実行する.

    返り値: (mse[name][seed], curves[name] (先頭 seed), preds[name] (先頭 seed),
             target, x_raw)
    各 (seed, 手法) の直前に torch/np を seed し直すので、同一 seed のすべての
    手法は同一の初期重み・同一の乱数系列から開始する。
    """
    device = torch.device(args.device)
    x_raw, target, x, t = make_task(device)
    mse = {name: {} for name, _ in methods}
    curves, preds = {}, {}
    for seed in args.seed_list:
        for name, spec in methods:
            torch.manual_seed(seed)
            np.random.seed(seed)
            fresh = model_factory(noise, args, device)
            losses, pred, _ = run_method(spec, fresh, x, t, noise, args, log_every)
            mse[name][seed] = float(np.mean((pred - target) ** 2))
            print(f"[seed {seed}] {name:24s} final MSE = {mse[name][seed]:.5f}",
                  flush=True)
            if seed == args.seed_list[0]:
                curves[name] = losses
                preds[name] = pred
    return mse, curves, preds, target, x_raw


# ============================================================
# 保存ヘルパ
# ============================================================
def mse_table_md(mse: dict, seed_list, caption: str = "") -> str:
    lines = []
    if caption:
        lines.append(f"**{caption}**")
        lines.append("")
    lines.append("| method | " + " | ".join(f"seed {s}" for s in seed_list)
                 + " | mean ± std |")
    lines.append("|---" * (len(seed_list) + 2) + "|")
    for name, per_seed in mse.items():
        vals = [per_seed[s] for s in seed_list]
        cells = " | ".join(f"{v:.5f}" for v in vals)
        lines.append(f"| {name} | {cells} | "
                     f"{np.mean(vals):.5f} ± {np.std(vals):.5f} |")
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")
    print(f"  saved {path}")


def save_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    print(f"  saved {path}")


def savefig(fig, path: Path) -> None:
    """PNG (確認用) と PDF (論文用) を常に併せて保存する."""
    path = Path(path)
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {path} (+ .pdf)")


def bar_mse(mse: dict, seed_list, title: str, path: Path) -> None:
    """最終 MSE の棒グラフ (log scale, seed 間の誤差棒つき)."""
    names = list(mse.keys())
    means = [float(np.mean([mse[n][s] for s in seed_list])) for n in names]
    stds = [float(np.std([mse[n][s] for s in seed_list])) for n in names]
    fig = plt.figure(figsize=(max(6.0, 1.1 * len(names)), 4.5))
    plt.bar(range(len(names)), means, yerr=stds, capsize=3, color="tab:blue")
    plt.yscale("log")
    plt.xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
    plt.ylabel("final MSE (log)")
    plt.title(title)
    plt.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    savefig(fig, path)


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().cpu().numpy().ravel()
    b = b.detach().cpu().numpy().ravel()
    return float(np.corrcoef(a, b)[0, 1])


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().ravel()
    b = b.detach().ravel()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def norm_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(a.detach().norm() / (b.detach().norm() + 1e-12))


# ============================================================
# 検証セット一式 (fncl5_2 / fncl5_5 が使用)
# ============================================================
def run_verification(noise: str, args) -> dict:
    """検証 8 手法を複数 seed で走らせ、Tab.1 / Fig.3 / Fig.4 相当を保存する."""
    log_every = max(1, args.epochs // 5)
    mse, curves, preds, target, x_raw = run_matrix(
        VERIFICATION_METHODS, noise, args, log_every)
    out = args.out_dir
    caption = (f"Final MSE (8-pass predict), noise={noise}, "
               f"H={args.hidden_dim}, T={args.num_samples}, "
               f"epochs={args.epochs}, lr={args.lr}")
    table = mse_table_md(mse, args.seed_list, caption)
    print("\n" + table)                     # 表は標準出力のみ (ファイル保存しない)
    save_json(out / "results.json",
              {"config": config_dict(args), "noise": noise, "final_mse": mse})
    np.savez(out / "curves_preds.npz", x_raw=x_raw, target=target,
             **{f"curve_{k}": np.asarray(v) for k, v in curves.items()},
             **{f"pred_{k}": np.asarray(v) for k, v in preds.items()})
    print(f"  saved {out / 'curves_preds.npz'}")
    savefig(fncl.plot_losses(curves), out / "fig_learning_curves.png")
    savefig(fncl.plot_predictions(x_raw, target, preds),
            out / "fig_predictions.png")
    savefig(fncl.plot_fit_check(x_raw, target, preds), out / "fig_fit_check.png")
    return mse
