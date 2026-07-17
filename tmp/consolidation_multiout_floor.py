"""
consolidation_multiout_floor.py — 多出力タスクでの圧縮フロア則の検証
(docs/idea_consolidation.md §12.6.8(3) の法則の追試)

§12.6.8 の法則:

    floor_l = dim(下流が層 l に要求する関数空間) <= ceil(er_l)

単出力 sin 回帰では最終隠れ層の下流（線形読み出し）が rank-1 だったため
floor_L2 = 1 だった。本スクリプトは出力次元 m を振って「下流の要求次元」を
直接操作する：

    m=1: y = sin(x)
    m=2: y = [sin(x), cos(x)]
    m=3: y = [sin(x), cos(x), sin(2x)]

出力成分は互いに線形独立な関数なので、読み出しが層2に要求する関数空間は
m 次元。予測は floor_L2 = m（かつ floor_L2 <= ceil(er2)）。層1 の事前学習
有効ランク er1 も記録し、要求関数族の複雑化（sin(2x) はより細かい基底を要る）
に伴う増加を確認する。

方法: 各 (seed, m) で cov_jac 事前学習 → 層2 のみ 32 -> 1 の限界超過
アニール（層1 は無傷）→ フロア = スナップ MSE <= tol の最小生存数。
m=1 は §12.6.8 の再現を兼ねる対照。

生成物 (out/consolidation_multiout_floor/):
  fig_multiout.png    m 別の圧縮曲線（MSE vs 層2生存数、er2 線・フロア線つき）
  table_multiout.md   m x seed のフロア vs 予測
  results.json

実行例:
  python tmp/consolidation_multiout_floor.py --quick
  python tmp/consolidation_multiout_floor.py --seeds 0,1
"""
import argparse

import numpy as np
import torch
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fncl_driver as fncl  # noqa: E402
from fncl_driver import save_json, savefig, write_text  # noqa: E402
import consolidation_poc as poc  # noqa: E402
from consolidation_floor_test import (StreamingRankTrainer,  # noqa: E402
                                      run_condition, floor_of)
from consolidation_stop_signal import effective_rank  # noqa: E402


def make_task_multi(device, m: int):
    """y = [sin, cos, sin(2x)][:m] on x in [-2pi, 2pi]（入力は ~[-2,2] に正規化）."""
    N = 128
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, N, dtype=np.float32)
    comps = [np.sin(x_raw), np.cos(x_raw), np.sin(2.0 * x_raw)][:m]
    target = np.stack(comps, axis=1).astype(np.float32)    # [N, m]
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    t = torch.tensor(target, device=device)
    return x_raw, target, x, t


def main():
    p = argparse.ArgumentParser(
        description="compression-floor law under multi-output tasks")
    poc.add_poc_args(p)
    p.add_argument("--ms", type=str, default="1,2,3",
                   help="出力次元 m のリスト（カンマ区切り）")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/consolidation_multiout_floor")
    args.m_list = [int(v) for v in args.ms.split(",") if v.strip()]
    if args.quick:
        args.epochs_per_step = 5
        args.m_list = args.m_list[:2]

    device = torch.device(args.device)
    H = args.hidden_dim
    log_every = max(1, args.epochs // 5)

    # run_route_B 内部のトレーナをストリーミング版へ（run_condition が erank を呼ぶ）
    poc.CovJacTrainer = StreamingRankTrainer

    all_res = {}
    for seed in args.seed_list:
        all_res[seed] = {}
        for m in args.m_list:
            print(f"\n===== seed {seed}, m={m} =====")
            torch.manual_seed(seed)
            np.random.seed(seed)
            x_raw, target, x, t = make_task_multi(device, m)

            net = poc.build_net(H, args.sigma, args.crossing_h,
                                args.num_samples, device, out_dim=m)
            trainer = StreamingRankTrainer(net, x, t, lr=args.pre_lr,
                                           opt=args.opt, jac_ema=args.jac_ema)
            for e in range(args.epochs):
                loss = trainer.step()
                if e % log_every == 0 or e == args.epochs - 1:
                    print(f"  [pretrain] epoch {e:5d} mse={loss:.5f}")
            trainer.close()
            base = float(np.mean(trainer.losses[-min(100, len(trainer.losses)):]))
            tol = base * args.drift_mult + args.drift_abs
            # m=1 では fncl.predict が [N] を返すため target も 1 次元に揃える
            # （[N] - [N,1] のブロードキャスト事故で MSE が壊れるのを防ぐ）
            target_e = target[:, 0] if m == 1 else target
            pred0 = fncl.predict(net, x, passes=16)
            pre_mse = float(np.mean((pred0 - target_e) ** 2))

            _, zbar0 = poc.collect_stats(net, x, passes=args.stat_passes)
            er0 = {l: effective_rank(zbar0[l], list(range(H))) for l in (0, 1)}
            print(f"  pretrained: MSE={pre_mse:.5f} tol={tol:.5f} "
                  f"er=(L1 {er0[0]:.2f}, L2 {er0[1]:.2f})")

            rec, epochs, holds = run_condition(
                net, x, target_e, pred0, layer=1, keep=1, tol=tol, args=args)
            fl = floor_of(rec, tol)
            print(f"  floor_L2={fl}  (m={m}, er2_pre={er0[1]:.2f}, holds={holds})")
            all_res[seed][m] = {
                "tol": tol, "pre_mse": pre_mse, "floor": fl, "holds": holds,
                "er1_pre": er0[0], "er2_pre": er0[1],
                "rec": {k: list(map(float, v)) for k, v in rec.items()}}

    # --- 図（先頭 seed） -------------------------------------------------
    seed0 = args.seed_list[0]
    fig, axes = plt.subplots(1, len(args.m_list),
                             figsize=(4.2 * len(args.m_list), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for col, m in enumerate(args.m_list):
        r = all_res[seed0][m]
        na = np.asarray(r["rec"]["na"])
        ax = axes[col]
        ax.plot(na, r["rec"]["mse"], "o-", ms=3, label="task MSE at snap")
        ax.axhline(r["tol"], ls="--", c="k", lw=0.8, label="tolerance")
        ax.axvline(m, color="tab:green", lw=1.4, ls="-.",
                   label=f"predicted floor = m = {m}")
        ax.axvline(r["er2_pre"], color="purple", lw=1.2, ls="--",
                   label=f"pretrain er2 = {r['er2_pre']:.2f}")
        ax.axvline(r["floor"], color="r", lw=1.0, ls=":",
                   label=f"observed floor = {r['floor']}")
        ax.set_yscale("log")
        ax.set_xlabel("survivors in layer 2")
        if col == 0:
            ax.set_ylabel("task MSE")
        ax.set_title(f"m = {m}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
        ax.invert_xaxis()
    fig.suptitle(f"Compression floor of the last hidden layer vs output "
                 f"dimension (H={H}, seed {seed0})", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    savefig(fig, args.out_dir / "fig_multiout.png")

    # --- 表 ---------------------------------------------------------------
    lines = [f"**Multi-output compression floor** (H={H}, "
             f"seeds={args.seed_list}, ms={args.m_list})", "",
             "| seed | m | pre MSE | er1 | er2 | ceil(er2) | predicted floor "
             "(=m) | observed floor | holds |",
             "|---" * 9 + "|"]
    for seed in args.seed_list:
        for m in args.m_list:
            r = all_res[seed][m]
            lines.append(
                f"| {seed} | {m} | {r['pre_mse']:.4f} | {r['er1_pre']:.2f} "
                f"| {r['er2_pre']:.2f} | {int(np.ceil(r['er2_pre']))} "
                f"| {m} | **{r['floor']}** | {r['holds']} |")
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_multiout.md", table)
    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args),
               "results": {str(s): {str(m): all_res[s][m]
                                    for m in all_res[s]} for s in all_res}})


if __name__ == "__main__":
    main()
