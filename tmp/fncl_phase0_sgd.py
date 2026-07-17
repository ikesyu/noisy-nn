"""
fncl_phase0_sgd.py — FPGA 実装 Phase 0 実験①「Adam なしで論文水準に届くか」

FPGA 実装計画 (Phase 0: アルゴリズム仕様の凍結) の最初の検証。Adam は
パラメータごとに sqrt + 除算 + 状態 2 語を要し、論文 §6.1 の資源分析
(累算・乗減算・ユニットあたり 1 回の除算・MAC のみ) を学習系全体では
成立させなくなる。そこで cov_jac (既定; --method cov_jac_full も可) を

    optimizer ∈ { sgd, sgdm (古典的 momentum; 状態 1 語, sqrt/除算なし) }
  x lr 減衰   ∈ { none, cosine } (exp も指定可; lr_at と同じ実装)
  x lr        ∈ { 0.01, 0.03, 0.1, 0.3 } (既定; SGD は勾配を正規化しない)

で掃引し、基準 2 本

    backprop     : Adam 参照 (論文 Tab.2/5 と同じ)
    cov_jac_adam : 論文の主結果設定 (Adam, decay なし, lr=--lr)

と比較する。判定は「HW 実装可能な構成 (sgd/sgdm) の最良 MSE が backprop の
2 倍以内に収まるか」。ノイズは既定 uniform (HW ターゲット; 論文 §6.2 の
プロトコル)。乱数系列は (seed, 構成) ごとに再シードするので、同一 seed の
全構成は同一の初期重み・同一のノイズ系列から出発する。

sgdm の update() は実際に適用したステップを返すため、jac_track=True の
Kolen-Pollack mirror 追跡 (既知の増分だけ mirror をずらす) は厳密なまま。

生成物 (out/fncl_phase0_sgd/):
  table_sweep.md          -> 全構成の最終 MSE 表 (mean ± std over seeds)
  table_best.md           -> backprop / cov_jac_adam / 最良 sgd / 最良 sgdm と判定
  fig_learning_curves.png -> 学習曲線 (基準 2 本 + 各 opt の最良構成, log scale)
  fig_mse_vs_lr.png       -> 最終 MSE vs lr (opt x decay ごとの線; 頑健性の確認)
  results.json

実行例:
  python tmp/fncl_phase0_sgd.py --quick             # 動作確認 (~1 分)
  python tmp/fncl_phase0_sgd.py                     # 本実験 (数十分)
  python tmp/fncl_phase0_sgd.py --noise gaussian    # ガウスノイズでの対照
  python tmp/fncl_phase0_sgd.py --method cov_jac_full
  python tmp/fncl_phase0_sgd.py --lrs 0.003,0.01,0.03,0.1 --decays none,cosine,exp
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 保存専用 (fncl_driver が pyplot を import する前に設定)
import matplotlib.pyplot as plt  # noqa: E402

# 実装は tmp/fncl_driver.py (旧 fncl_lib_tmp.py; ライブラリ部は nnn パッケージ) を使う。
import fncl_driver as fncl  # noqa: E402
from fncl_driver import (add_common_args, finalize_args, make_task,  # noqa: E402
                         model_factory, config_dict, write_text, save_json,
                         savefig)


# ============================================================
# ManualOpt + momentum ("sgdm") — HW 候補のオプティマイザ
# ============================================================
class ManualOptPhase0(fncl.ManualOpt):
    """ManualOpt に kind == "sgdm" (古典的 momentum) を追加する.

        v <- mu * v + grad;  param -= lr * v

    追加コストはパラメータあたり状態 1 語と乗算 1 回のみで、Adam と違い
    sqrt も除算も不要 (FPGA の固定小数点データパスにそのまま載る)。
    update() は実際に適用したステップを返すので Kolen-Pollack 追跡と整合。
    momentum 係数はクラス属性 (main が --momentum から設定)。
    """
    momentum = 0.9

    def __init__(self, kind: str, **kw):
        self.sgdm = (kind == "sgdm")
        super().__init__("sgd" if self.sgdm else kind, **kw)
        self.vel = {}

    def update(self, key: str, param, grad, lr: float):
        if not self.sgdm:
            return super().update(key, param, grad, lr)
        if key not in self.vel:
            self.vel[key] = torch.zeros_like(grad)
        v = self.vel[key]
        v.mul_(self.momentum).add_(grad)
        step = lr * v
        param.data -= step
        return step


# train_cov はモジュールグローバルの ManualOpt を参照するので差し替える。
fncl.ManualOpt = ManualOptPhase0


# ============================================================
# lr_at + "step" 減衰 — シフト演算だけで実装できる HW 候補のスケジュール
# ============================================================
_lr_at_orig = fncl.lr_at


def lr_at_phase0(lr0: float, epoch: int, epochs: int, decay: str) -> float:
    """epochs/8 ごとに半減 (lr >>= 1; 終端 lr0/128)。cosine/exp と違い乗算器も
    テーブルも不要で、エポックカウンタの上位ビットでシフト量を決めるだけ。"""
    if decay == "step":
        return lr0 * 0.5 ** (epoch // max(1, epochs // 8))
    return _lr_at_orig(lr0, epoch, epochs, decay)


fncl.lr_at = lr_at_phase0


# ============================================================
# 1 構成の学習 (fncl.run_method は lr / lr_decay を固定するため使わない)
# ============================================================
def train_one(fresh, x, t, noise: str, args, opt: str, lr: float,
              lr_decay: str, log_every: int = 0):
    """cov_jac (または --method cov_jac_full) を 1 構成ぶん学習する."""
    net = fresh()
    extra = {"jac_out": args.jac_out} if args.method == "cov_jac_full" else {}
    losses, _ = fncl.train_cov(
        net, x, t, noise, args.sigma, args.radius, args.method, lr, args.epochs,
        credit=args.credit, credit_passes=args.credit_passes, opt=opt,
        lr_decay=lr_decay, log_every=log_every, jac_ema=args.jac_ema,
        jac_track=True, **extra)
    pred = fncl.predict(net, x)
    return losses, pred


def run_config(name: str, runner, args, x, t, noise, target, mse, curves,
               log_every):
    """seed ごとに再シードして runner を実行し、最終 MSE と先頭 seed の曲線を記録."""
    device = torch.device(args.device)
    mse[name] = {}
    for seed in args.seed_list:
        torch.manual_seed(seed)
        np.random.seed(seed)
        fresh = model_factory(noise, args, device)
        losses, pred = runner(fresh)
        mse[name][seed] = float(np.mean((pred - target) ** 2))
        print(f"[seed {seed}] {name:32s} final MSE = {mse[name][seed]:.5f}",
              flush=True)
        if seed == args.seed_list[0]:
            curves[name] = losses


# ============================================================
# 表・図
# ============================================================
def mean_std(mse: dict, name: str, seed_list):
    vals = [mse[name][s] for s in seed_list]
    return float(np.mean(vals)), float(np.std(vals))


def sweep_table(mse: dict, seed_list, caption: str) -> str:
    lines = [f"**{caption}**", "",
             "| config | " + " | ".join(f"seed {s}" for s in seed_list)
             + " | mean ± std |",
             "|---" * (len(seed_list) + 2) + "|"]
    for name in mse:
        m, s = mean_std(mse, name, seed_list)
        cells = " | ".join(f"{mse[name][sd]:.5f}" for sd in seed_list)
        lines.append(f"| {name} | {cells} | {m:.5f} ± {s:.5f} |")
    return "\n".join(lines) + "\n"


def fig_curves(curves: dict, path) -> None:
    fig = plt.figure(figsize=(7.0, 4.5))
    for name, losses in curves.items():
        plt.plot(losses, label=name, lw=1.2)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("eval MSE (log)")
    plt.title("Phase 0: optimizer comparison (baselines + best HW-friendly configs)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def fig_mse_vs_lr(mse: dict, sweep_names: dict, lrs, seed_list, baselines,
                  path) -> None:
    """opt x decay ごとに最終 MSE (mean) を lr の関数として描く (頑健性の確認)."""
    fig = plt.figure(figsize=(7.0, 4.5))
    for (opt, decay), names_by_lr in sweep_names.items():
        means = [mean_std(mse, names_by_lr[lr], seed_list)[0] for lr in lrs]
        plt.plot(lrs, means, marker="o", label=f"{opt} / decay={decay}")
    for name, style in baselines:
        m, _ = mean_std(mse, name, seed_list)
        plt.axhline(m, ls=style, c="k", lw=1.0, label=name)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("learning rate")
    plt.ylabel("final MSE (mean over seeds, log)")
    plt.title("Phase 0: final MSE vs lr")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3, which="both")
    fig.tight_layout()
    savefig(fig, path)


# ============================================================
# main
# ============================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase 0-1: can cov_jac reach backprop-level MSE without Adam?")
    add_common_args(p)
    p.add_argument("--noise", choices=("uniform", "gaussian"), default="uniform",
                   help="HW ターゲットは uniform (既定)")
    p.add_argument("--method", choices=("cov_jac", "cov_jac_full"),
                   default="cov_jac")
    p.add_argument("--jac-out", choices=("cov", "cov_m3", "probe"),
                   default="cov_m3", help="cov_jac_full の読み出し誤差推定")
    p.add_argument("--opts", type=str, default="sgd,sgdm",
                   help="掃引するオプティマイザ (カンマ区切り; sgd, sgdm)")
    p.add_argument("--decays", type=str, default="none,cosine",
                   help="掃引する lr 減衰 (none, cosine, exp, step; "
                        "step は epochs/8 ごとに半減 = シフトのみで実装可)")
    p.add_argument("--lrs", type=str, default="0.01,0.03,0.1,0.3",
                   help="掃引する学習率 (カンマ区切り; Adam と違い勾配を正規化"
                        "しないため大きめまで掃引する)")
    p.add_argument("--momentum", type=float, default=0.9, help="sgdm の係数")
    args = finalize_args(p.parse_args(), default_out="out/fncl_phase0_sgd")
    args.opt_list = [s for s in args.opts.split(",") if s.strip()]
    args.decay_list = [s for s in args.decays.split(",") if s.strip()]
    args.lr_list = [float(s) for s in args.lrs.split(",") if s.strip()]
    if args.quick:
        args.lr_list = args.lr_list[:2]
        args.decay_list = args.decay_list[:1]
    ManualOptPhase0.momentum = args.momentum

    device = torch.device(args.device)
    x_raw, target, x, t = make_task(device)
    log_every = max(1, args.epochs // 5)
    mse, curves = {}, {}

    # --- 基準 2 本 -------------------------------------------------------
    def bp(fresh):
        net = fresh()
        losses = fncl.train_backprop(net, x, t, args.lr, args.epochs, log_every)
        return losses, fncl.predict(net, x)

    run_config("backprop", bp, args, x, t, args.noise, target, mse, curves,
               log_every)
    adam_name = f"{args.method}_adam_lr{args.lr:g}"
    run_config(adam_name,
               lambda fresh: train_one(fresh, x, t, args.noise, args,
                                       "adam", args.lr, "none", log_every),
               args, x, t, args.noise, target, mse, curves, log_every)

    # --- 掃引: opt x decay x lr ------------------------------------------
    sweep_names = {}                       # (opt, decay) -> {lr: config name}
    for opt in args.opt_list:
        for decay in args.decay_list:
            sweep_names[(opt, decay)] = {}
            for lr in args.lr_list:
                name = f"{args.method}_{opt}_{decay}_lr{lr:g}"
                sweep_names[(opt, decay)][lr] = name
                run_config(name,
                           lambda fresh, o=opt, l=lr, d=decay:
                               train_one(fresh, x, t, args.noise, args,
                                         o, l, d, log_every),
                           args, x, t, args.noise, target, mse, curves,
                           log_every)

    # --- 集計と判定 -------------------------------------------------------
    bp_mean, _ = mean_std(mse, "backprop", args.seed_list)
    adam_mean, _ = mean_std(mse, adam_name, args.seed_list)
    best = {}                              # opt -> (name, mean, std)
    for opt in args.opt_list:
        cands = [n for (o, _), by_lr in sweep_names.items() if o == opt
                 for n in by_lr.values()]
        name = min(cands, key=lambda n: mean_std(mse, n, args.seed_list)[0])
        best[opt] = (name, *mean_std(mse, name, args.seed_list))

    caption = (f"Phase 0-1 optimizer sweep: {args.method}, noise={args.noise}, "
               f"H={args.hidden_dim}, T={args.num_samples}, "
               f"epochs={args.epochs}, seeds={args.seed_list}, "
               f"momentum={args.momentum}")
    table = sweep_table(mse, args.seed_list, caption)
    print("\n" + table)
    write_text(args.out_dir / "table_sweep.md", table)

    lines = [f"**Phase 0-1 verdict ({args.method}, noise={args.noise})**", "",
             "| config | final MSE (mean ± std) | vs backprop | vs adam |",
             "|---|---|---|---|",
             f"| backprop (Adam 参照) | {bp_mean:.5f} | x1.00 | — |",
             f"| {adam_name} | {adam_mean:.5f} | x{adam_mean / bp_mean:.2f} | x1.00 |"]
    for opt, (name, m, s) in best.items():
        lines.append(f"| best {opt}: {name} | {m:.5f} ± {s:.5f} "
                     f"| x{m / bp_mean:.2f} | x{m / adam_mean:.2f} |")
    hw_best = min(best.values(), key=lambda v: v[1])
    ok = hw_best[1] <= 2.0 * bp_mean
    lines += ["",
              f"HW 実装可能な最良構成: `{hw_best[0]}` "
              f"(MSE {hw_best[1]:.5f} = backprop の x{hw_best[1] / bp_mean:.2f})",
              ("=> **PASS**: Adam なしで backprop 水準 (x2 以内)。"
               "§6.1 の資源分析が学習系全体で成立する。" if ok else
               "=> **FAIL**: x2 を超過。lr / momentum / decay の追加掃引か、"
               "層ごと適応スケーリングの検討が必要。")]
    verdict = "\n".join(lines) + "\n"
    print("\n" + verdict)
    write_text(args.out_dir / "table_best.md", verdict)

    save_json(args.out_dir / "results.json",
              {"config": config_dict(args), "noise": args.noise,
               "final_mse": mse,
               "best": {opt: {"name": n, "mean": m, "std": s}
                        for opt, (n, m, s) in best.items()},
               "backprop_mean": bp_mean, "adam_mean": adam_mean,
               "pass": bool(ok)})

    shown = ["backprop", adam_name] + [best[opt][0] for opt in best]
    fig_curves({k: curves[k] for k in shown if k in curves},
               args.out_dir / "fig_learning_curves.png")
    fig_mse_vs_lr(mse, sweep_names, args.lr_list, args.seed_list,
                  [("backprop", ":"), (adam_name, "--")],
                  args.out_dir / "fig_mse_vs_lr.png")


if __name__ == "__main__":
    main()
