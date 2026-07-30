"""
fncl_baselines.py — 査読対応 A2「競合 forward-only 手法の実測ベースライン」(docs/draft_nce.md 追記 §A2)
（旧名: fncl_a2.py．出力ディレクトリ tmp/out/ は旧名のまま）

§5.1 の sin(x) プロトコル (1-64-64-1, T=64, 1500 epoch, gaussian sigma=0.5,
seed 0-2, 同一初期重み) を一切変えずに、NNN 上で競合 forward-only 手法を走らせる:

  fa_{sgd,adam}    : feedback alignment (Lillicrap+ 2016)。cov_jac の再帰の W_hat を
                     固定ランダム行列 B に置き換えただけの構成 (局所傾きは同じ KDE slope)。
  dfa_{sgd,adam}   : direct feedback alignment (Nøkland 2016)。出力誤差を固定ランダム
                     行列 B_l で各隠れ層へ直接投影。
  np_a<alpha>_*    : node perturbation。既存の cov_deriv_gate_crn (antithetic /
                     common-random-number, Appendix B) を全ユニットゲート
                     (gate_block_size=H) で使用 = 不偏・低分散の最良好意実装。
  pepita_f<scale>  : PEPITA (Dellaferrera & Kreiman 2022)。誤差変調した 2 回目の
                     forward とアンサンブル平均活性の差で隠れ層を更新。

--benchmarks を付けると §5.6 の 3 ベンチマーク (friedman1 / moons / circles) で
既定手法 (fa_adam, dfa_adam, np_a0.1_sgd) を走らせる。

生成物 (tmp/out/fncl_a2/):
  results.json / results_bench.json, (標準出力) 最終 MSE 表

実行例:
  .venv/bin/python tmp/fncl_baselines.py --quick
  .venv/bin/python tmp/fncl_baselines.py                 # sin(x) 本番
  .venv/bin/python tmp/fncl_baselines.py --benchmarks    # 3 ベンチマーク本番
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data_nce"
sys.path.insert(0, str(DATA_DIR))
import fncl  # noqa: E402
from fncl_common import (add_common_args, finalize_args, config_dict,  # noqa: E402
                         make_task, model_factory, mse_table_md, save_json)
from fncl5_6 import TASKS, model_factory_nd, accuracy, DATA_SEED  # noqa: E402


# ============================================================
# FA / DFA (cov_jac の再帰から W_hat を固定ランダム行列に置き換えたもの)
# ============================================================
def train_fa(net, x, t_target, method: str, lr: float, epochs: int,
             opt: str = "adam", log_every: int = 0):
    """feedback alignment ("fa") / direct feedback alignment ("dfa").

    隠れ層 credit:
      fa : a^{L-1} = e B_out,  a^l = (a^{l+1} * slope^{l+1}) B^{l+1}
           (cov_jac の再帰と同形; W_hat -> 固定ランダム B)
      dfa: a^l = e B_l  (出力誤差を各層へ直接投影)
    局所傾きは cov_jac と同じ分布フリー KDE slope。読み出し層は全手法共通の
    厳密な局所勾配 (アンサンブル平均特徴上)。
    """
    cap = fncl.Capture(net)
    n_hidden = cap.n_hidden
    N = x.shape[0]
    out_dim = net.fcs[-1].weight.shape[0]
    hidden_sizes = list(net.structure[1:-1])

    # 固定ランダムフィードバック行列 (前向き重みの既定初期化と同じ一様スケール)
    B = {}
    if method == "fa":
        for l in range(1, n_hidden):
            fan_in = hidden_sizes[l - 1]
            B[l] = (torch.rand_like(net.fcs[l].weight) * 2 - 1) / math.sqrt(fan_in)
        B["out"] = ((torch.rand_like(net.fcs[-1].weight) * 2 - 1)
                    / math.sqrt(hidden_sizes[-1]))
    else:  # dfa
        for l in range(n_hidden):
            B[l] = ((torch.rand(out_dim, hidden_sizes[l], device=x.device) * 2 - 1)
                    / math.sqrt(hidden_sizes[l]))

    optim = fncl.ManualOpt(opt)
    losses = []
    for epoch in range(epochs):
        with torch.no_grad():
            y = net(x)                                       # [N, C]; fires hooks
            d = [cap.d[l] for l in range(n_hidden)]
            z = [cap.z[l] for l in range(n_hidden)]
            T = z[0].shape[1]
            e = 2.0 * (y - t_target)                         # [N, C]
            slope_full = [fncl.kde_slope(cap.crossings[l], d[l])
                          for l in range(n_hidden)]          # [N, T, H]
            slope_mean = [s.mean(dim=1) for s in slope_full]  # [N, H]

            a = [None] * n_hidden
            if method == "fa":
                a[-1] = e @ B["out"]                          # [N, H]
                for l in range(n_hidden - 2, -1, -1):
                    a[l] = (a[l + 1] * slope_mean[l + 1]) @ B[l + 1]
            else:
                for l in range(n_hidden):
                    a[l] = e @ B[l]

            z_prev = ([x.unsqueeze(1).expand(N, T, x.shape[1])]
                      + [z[l] for l in range(n_hidden - 1)])
            for l in range(n_hidden):
                delta = a[l].unsqueeze(1) * slope_full[l]     # [N, T, H]
                gW = torch.einsum("nto,nti->oi", delta, z_prev[l]) / (N * T)
                gb = delta.mean(dim=(0, 1))
                optim.update(f"w{l}", net.fcs[l].weight, gW, lr)
                if net.fcs[l].bias is not None:
                    optim.update(f"b{l}", net.fcs[l].bias, gb, lr)

            z_bar = z[-1].mean(dim=1)                         # [N, H]
            gWout = torch.einsum("no,ni->oi", e, z_bar) / N
            optim.update("wout", net.fcs[-1].weight, gWout, lr)
            if net.fcs[-1].bias is not None:
                optim.update("bout", net.fcs[-1].bias, e.mean(dim=0), lr)

            losses.append(float(((y - t_target) ** 2).mean()))
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  [{method}/{opt}] epoch {epoch:5d}  mse={losses[-1]:.5f}",
                  flush=True)
    cap.remove()
    return losses


# ============================================================
# PEPITA (誤差変調した 2 回目の forward; アンサンブル平均活性で構成)
# ============================================================
def train_pepita(net, x, t_target, lr: float, epochs: int, fscale: float,
                 opt: str = "sgd", log_every: int = 0):
    cap = fncl.Capture(net)
    n_hidden = cap.n_hidden
    N, in_dim = x.shape
    out_dim = net.fcs[-1].weight.shape[0]
    F = fscale * torch.randn(in_dim, out_dim, device=x.device) / math.sqrt(out_dim)
    optim = fncl.ManualOpt(opt)
    losses = []
    for epoch in range(epochs):
        with torch.no_grad():
            y1 = net(x)                                       # standard pass
            zbar1 = [cap.z[l].mean(dim=1) for l in range(n_hidden)]
            e = y1 - t_target                                 # [N, C]
            x_mod = x + e @ F.T                               # modulated input
            y2 = net(x_mod)                                   # modulated pass
            zbar2 = [cap.z[l].mean(dim=1) for l in range(n_hidden)]
            prev2 = [x_mod] + zbar2[:-1]
            for l in range(n_hidden):
                g = zbar1[l] - zbar2[l]                       # [N, H]
                gW = torch.einsum("no,ni->oi", g, prev2[l]) / N
                optim.update(f"w{l}", net.fcs[l].weight, gW, lr)
                if net.fcs[l].bias is not None:
                    optim.update(f"b{l}", net.fcs[l].bias, g.mean(dim=0), lr)
            gWout = torch.einsum("no,ni->oi", e, zbar2[-1]) / N
            optim.update("wout", net.fcs[-1].weight, gWout, lr)
            if net.fcs[-1].bias is not None:
                optim.update("bout", net.fcs[-1].bias, e.mean(dim=0), lr)
            losses.append(float(((y1 - t_target) ** 2).mean()))
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  [pepita f={fscale}] epoch {epoch:5d}  mse={losses[-1]:.5f}",
                  flush=True)
    cap.remove()
    return losses


# ============================================================
# 実験行列
# ============================================================
def run_spec(spec: dict, fresh, x, t, args, log_every: int):
    net = fresh()
    kind = spec["kind"]
    if kind in ("fa", "dfa"):
        losses = train_fa(net, x, t, kind, args.lr, args.epochs,
                          opt=spec.get("opt", "adam"), log_every=log_every)
    elif kind == "pepita":
        losses = train_pepita(net, x, t, args.lr, args.epochs,
                              fscale=spec["fscale"], opt=spec.get("opt", "sgd"),
                              log_every=log_every)
    elif kind == "np":
        # node perturbation = 既存 CRN ゲート (最良好意実装) を全ユニットゲートで
        losses, _ = fncl.train_cov(
            net, x, t, args.noise, args.sigma, args.radius, "cov_deriv_gate_crn",
            args.lr, args.epochs, credit="per_input", opt=spec.get("opt", "sgd"),
            gate_block_size=args.hidden_dim, gate_alpha=spec["alpha"],
            log_every=log_every)
    else:
        raise ValueError(kind)
    pred = fncl.predict(net, x)
    return losses, pred


SIN_METHODS = [
    ("fa_sgd",       {"kind": "fa", "opt": "sgd"}),
    ("fa_adam",      {"kind": "fa", "opt": "adam"}),
    ("dfa_sgd",      {"kind": "dfa", "opt": "sgd"}),
    ("dfa_adam",     {"kind": "dfa", "opt": "adam"}),
    ("np_a0.1_sgd",  {"kind": "np", "alpha": 0.1, "opt": "sgd"}),
    ("np_a0.3_sgd",  {"kind": "np", "alpha": 0.3, "opt": "sgd"}),
    ("np_a0.1_adam", {"kind": "np", "alpha": 0.1, "opt": "adam"}),
    ("pepita_f0.1",  {"kind": "pepita", "fscale": 0.1}),
    ("pepita_f0.5",  {"kind": "pepita", "fscale": 0.5}),
    ("pepita_f1.0",  {"kind": "pepita", "fscale": 1.0}),
]

BENCH_METHODS = [
    ("fa_adam",      {"kind": "fa", "opt": "adam"}),
    ("dfa_adam",     {"kind": "dfa", "opt": "adam"}),
    ("np_a0.1_sgd",  {"kind": "np", "alpha": 0.1, "opt": "sgd"}),
    ("np_a0.3_sgd",  {"kind": "np", "alpha": 0.3, "opt": "sgd"}),
]


def main() -> None:
    p = argparse.ArgumentParser(description="A2: competing forward-only baselines "
                                            "(FA/DFA/NP/PEPITA) on the NNN.")
    add_common_args(p)
    p.add_argument("--noise", choices=("gaussian", "uniform"), default="gaussian")
    p.add_argument("--benchmarks", action="store_true",
                   help="sin(x) の代わりに §5.6 の 3 ベンチマークを走らせる")
    args = finalize_args(p.parse_args(), default_out="tmp/out/fncl_a2")
    device = torch.device(args.device)
    log_every = max(1, args.epochs // 5)

    if not args.benchmarks:
        x_raw, target, x, t = make_task(device)
        mse = {name: {} for name, _ in SIN_METHODS}
        for seed in args.seed_list:
            for name, spec in SIN_METHODS:
                torch.manual_seed(seed)
                np.random.seed(seed)
                fresh = model_factory(args.noise, args, device)
                losses, pred = run_spec(spec, fresh, x, t, args, log_every)
                mse[name][seed] = float(np.mean((pred - target) ** 2))
                print(f"[seed {seed}] {name:16s} final MSE = {mse[name][seed]:.5f}",
                      flush=True)
        table = mse_table_md(mse, args.seed_list,
                             f"A2 baselines on sin(x), noise={args.noise}, "
                             f"H={args.hidden_dim}, T={args.num_samples}, "
                             f"epochs={args.epochs}, lr={args.lr}")
        print("\n" + table)
        save_json(args.out_dir / "results.json",
                  {"config": config_dict(args), "final_mse": mse})
        return

    # ---- §5.6 の 3 ベンチマーク ----
    results = {}
    for task_name, make in TASKS.items():
        x_np, t_np, kind = make(np.random.default_rng(DATA_SEED))
        x = torch.tensor(x_np, device=device)
        t = torch.tensor(t_np, device=device).unsqueeze(1)
        results[task_name] = {"kind": kind, "per_method": {}}
        for seed in args.seed_list:
            for name, spec in BENCH_METHODS:
                torch.manual_seed(seed)
                np.random.seed(seed)
                fresh = model_factory_nd(args.noise, x_np.shape[1], args, device)
                losses, pred = run_spec(spec, fresh, x, t, args, log_every)
                entry = {"mse": float(np.mean((pred - t_np) ** 2))}
                if kind == "classification":
                    entry["acc"] = accuracy(pred, t_np)
                results[task_name]["per_method"].setdefault(name, {})[seed] = entry
                extra = (f"  acc = {entry['acc']:.3f}"
                         if kind == "classification" else "")
                print(f"[{task_name} | seed {seed}] {name:16s} "
                      f"final MSE = {entry['mse']:.5f}{extra}", flush=True)

    task_names = list(results.keys())
    lines = ["| method | " + " | ".join(task_names) + " |",
             "|---" * (len(task_names) + 1) + "|"]
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
    print("\n" + "\n".join(lines) + "\n")
    save_json(args.out_dir / "results_bench.json",
              {"config": config_dict(args), "results": results})


if __name__ == "__main__":
    main()
