"""
consolidation_caseB.py — 案B: 獲得自体を共有場 + リハーサルで行う
(docs/idea_consolidation.md §12.9.11)

案A（二層共有, §12.9.9）は L1 基底を凍結して再利用した。案B はさらに進め、
**獲得を最初から案4 の regime で行う**: 単一の共有場（全ユニット動員）の上で、
新タスクの学習を過去タスクの交互提示リハーサルと同時に走らせる。凍結は
一切ない — 全タスクが全パラメータを共同で使い、干渉は多目的閉ループ
（過去タスクの損失 EMA が許容超過ならリハーサルを追加）が管理する。
忘却保証は「厳密ゼロ」ではなく「許容内に有界」。

プロトコル（容量研究 §12.9.8 と同一のタスク族・閾値）:
  stage t: ctx_t を追加し、ctxs[1..t] の交互提示で epochs-stage ラウンド学習
           （過去タスクの EMA 超過で追加ラウンド = 閉ループ）。stage 末に
           各タスクの許容値を平衡から再較正し、全タスクの MSE を記録。
  最終:    union anneal（多目的 + 絶対格納基準の受理条件 §12.9.10）で圧縮し、
           容量と union を確定する。

比較対象: soft 4/12（§12.9.8）、softA 6/12（§12.9.9）。案B は関数空間の
内在次元（6）まで union が縮む可能性がある一方、リハーサルコストは
stage t あたり t 倍（合計 Σt ∝ K²/2）かかる。

生成物 (out/consolidation_caseB/):
  fig_caseB.png / table_caseB.md / results.json

実行例:
  python tmp/consolidation_caseB.py --quick
  python tmp/consolidation_caseB.py --seeds 0
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
import consolidation_joint as cjoint  # noqa: E402
import consolidation_capacity as ccap  # noqa: E402
from consolidation_lib import (  # noqa: E402
    TaskCtx, alive, joint_round, predict)


def eval_ctx(net, x, ctx, passes: int = 16) -> float:
    ctx.activate(net)
    pred = predict(net, x, passes=passes)
    return float(np.mean((pred - ctx.target) ** 2))


def main():
    p = argparse.ArgumentParser(description="case B: acquisition in the "
                                            "shared-field regime")
    poc.add_poc_args(p)
    p.add_argument("--epochs-stage", type=int, default=400,
                   help="タスク追加 1 stage あたりの交互提示ラウンド数")
    p.add_argument("--stop-recovery", type=int, default=8)
    p.add_argument("--layer-fails", type=int, default=3)
    p.add_argument("--mse-star", type=float, default=0.05)
    p.add_argument("--n-phases", type=int, default=4)
    p.add_argument("--freqs", type=str, default=None,
                   help="カンマ区切り周波数リスト（既定: 1,2,3 / quick: 1,2）")
    p.add_argument("--multiscale", action="store_true",
                   help="マルチスケール L1 初期化 (§12.9.12) を使う")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/consolidation_caseB")
    if args.quick:
        args.epochs_stage = 80
        args.epochs_per_step = 5
        args.stop_recovery = 3
        args.n_phases = 2

    device = torch.device(args.device)
    N = 128
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, N, dtype=np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    if args.freqs:
        freqs = tuple(int(v) for v in args.freqs.split(","))
    else:
        freqs = (1, 2) if args.quick else (1, 2, 3)
    tasks = ccap.make_tasks(x_raw, freqs, args.n_phases)
    K = len(tasks)
    H = args.hidden_dim
    thr = args.mse_star

    all_results = {}
    for seed in args.seed_list:
        print(f"\n===== seed {seed}  ({K} tasks, shared-field acquisition) "
              f"=====")
        torch.manual_seed(seed)
        np.random.seed(seed)
        from consolidation_multiscale import SCALES
        net = poc.build_net(H, args.sigma, args.crossing_h, args.num_samples,
                            device,
                            scales=SCALES if args.multiscale else None)
        # 単一の共有場: 全ユニットを常時動員（圧縮は最後にまとめて行う）
        net.sigma_vecs = [torch.full((H,), float(args.sigma), device=device)
                          for _ in range(2)]
        net.h_vecs = [torch.full((H,), float(args.crossing_h), device=device)
                      for _ in range(2)]

        ctxs = []
        stage_mses = []                      # stage_mses[t] = 各タスクの MSE
        holds_acq = 0
        for ti, (name, target) in enumerate(tasks):
            r = {"name": name, "target": target, "tol": float("inf"),
                 "wout": torch.zeros_like(net.fcs[2].weight.data),
                 "b_out": torch.zeros_like(net.fcs[2].bias.data),
                 "sigma0": float(args.sigma), "h0": float(args.crossing_h)}
            ctxs.append(TaskCtx.from_registry(net, x, r, args))
            n = 0
            while n < args.epochs_stage:
                joint_round(net, ctxs)
                n += 1
                h = 0
                while (h < args.max_holds
                       and any(c.trainer.ema() > c.tol for c in ctxs[:-1])):
                    joint_round(net, ctxs)      # 閉ループ: 過去タスク保護
                    h += 1
                holds_acq += h
            # 許容値を今 stage の平衡から再較正（§12.9.4 の規約）
            for c in ctxs:
                eq = float(np.mean(c.trainer.losses[-50:]))
                c.tol = eq * args.drift_mult + args.drift_abs
            mses = [eval_ctx(net, x, c) for c in ctxs]
            stage_mses.append(mses)
            print(f"  [stage {ti + 1}] {name}: new MSE={mses[-1]:.5f}; "
                  f"worst old MSE="
                  f"{(max(mses[:-1]) if len(mses) > 1 else 0.0):.5f} "
                  f"(holds={holds_acq})")

        # ---------------- 最終圧縮（絶対格納基準つき union anneal） ----------
        mses_pre = [eval_ctx(net, x, c) for c in ctxs]
        protected = [c for c, m in zip(ctxs, mses_pre) if m <= thr]
        print(f"  [pre-compress] stored {len(protected)}/{K}; "
              f"union = {len(alive(net, 0))}+{len(alive(net, 1))}")

        def accept():
            for c in protected:
                if eval_ctx(net, x, c) > thr:
                    return False
            return True

        for c in ctxs:
            c.tol_a = c.tol
        _, holds_cmp, removed = cjoint.union_anneal(net, x, ctxs, args,
                                                    accept_fn=accept)
        U = {l: alive(net, l) for l in (0, 1)}
        n_union = len(U[0]) + len(U[1])
        mses_post = [eval_ctx(net, x, c) for c in ctxs]
        stored_post = ccap.stored_flags(mses_post, thr)
        cap = sum(stored_post)
        total_steps = sum(len(c.trainer.losses) for c in ctxs)
        print(f"  [compress] union 64 -> {len(U[0])}+{len(U[1])} = {n_union} "
              f"(holds={holds_cmp})")
        for c, m0, m1 in zip(ctxs, mses_pre, mses_post):
            print(f"  [final] {c.name}: MSE pre-compress {m0:.5f} -> "
                  f"post {m1:.5f}")
        print(f"\n  ===== case B capacity: {cap}/{K} in {n_union} units "
              f"(units/fn {n_union / max(1, cap):.1f}; total task-steps "
              f"{total_steps}) =====")

        all_results[seed] = {
            "stage_mses": stage_mses, "mses_pre": mses_pre,
            "mses_post": mses_post, "stored": stored_post,
            "capacity": cap, "union": n_union,
            "union_l": [len(U[0]), len(U[1])],
            "holds_acq": holds_acq, "holds_compress": holds_cmp,
            "total_steps": total_steps}

        # ---------------- 図（先頭 seed） ----------------
        if seed == args.seed_list[0]:
            fig = plt.figure(figsize=(13, 4.6))
            gs = fig.add_gridspec(1, 3)
            ax = fig.add_subplot(gs[0, 0])
            M = np.full((K, K), np.nan)
            for t, mses in enumerate(stage_mses):
                M[:len(mses), t] = mses
            for i in range(K):
                ax.plot(range(i + 1, K + 1), M[i, i:], lw=1.0,
                        label=tasks[i][0] if i < 4 else None)
            ax.axhline(thr, ls="--", c="k", lw=0.9)
            ax.set_yscale("log")
            ax.set_xlabel("stage (tasks so far)")
            ax.set_ylabel("per-task MSE")
            ax.set_title("Bounded interference during shared acquisition",
                         fontsize=10)
            ax.legend(fontsize=6)
            ax.grid(alpha=0.3, which="both")

            ax = fig.add_subplot(gs[0, 1])
            idx = np.arange(K)
            colors = ["tab:green" if ok else "tab:red" for ok in stored_post]
            ax.bar(idx, mses_post, color=colors)
            ax.axhline(thr, ls="--", c="k", lw=0.9)
            ax.set_yscale("log")
            ax.set_xticks(idx)
            ax.set_xticklabels([t[0] for t in tasks], rotation=60, fontsize=6)
            ax.set_ylabel("final MSE (after compress)")
            ax.set_title(f"Stored {cap}/{K} in {n_union} units", fontsize=10)
            ax.grid(alpha=0.3, axis="y", which="both")

            ax = fig.add_subplot(gs[0, 2])
            bars = {"hard\n(§12.9.8)": 3, "soft\n(§12.9.8)": 4,
                    "softA\n(§12.9.9)": 6, "case B": cap}
            ax.bar(range(len(bars)), list(bars.values()),
                   color=["tab:gray", "tab:blue", "tab:red", "tab:purple"])
            ax.set_xticks(range(len(bars)))
            ax.set_xticklabels(list(bars.keys()), fontsize=8)
            ax.set_ylabel(f"capacity (/{K})")
            ax.set_title("Acquisition-regime comparison (seed 0)",
                         fontsize=10)
            ax.grid(alpha=0.3, axis="y")
            fig.suptitle(f"Case B: shared-field acquisition (H={H}, "
                         f"seed {seed})", fontsize=12)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
            savefig(fig, args.out_dir / "fig_caseB.png")

    # ---------------- 表 ----------------
    seed0 = args.seed_list[0]
    r0 = all_results[seed0]
    lines = [f"**Case B: shared-field acquisition** (H={H}, "
             f"T={args.num_samples}, epochs-stage={args.epochs_stage}, "
             f"thr={thr}, seeds={args.seeds})", "",
             "| task | MSE pre-compress | MSE post-compress |",
             "|---" * 3 + "|"]
    for i, (nm, _) in enumerate(tasks):
        st = "**" if r0["stored"][i] else ""
        lines.append(f"| {nm} | {r0['mses_pre'][i]:.4f} "
                     f"| {st}{r0['mses_post'][i]:.4f}{st} |")
    lines += ["",
              f"- capacity: **{r0['capacity']}/{K}** in **{r0['union']}** "
              f"units ({r0['union_l'][0]}+{r0['union_l'][1]}; units/fn "
              f"{r0['union'] / max(1, r0['capacity']):.1f})",
              f"- holds: acquisition {r0['holds_acq']}, compress "
              f"{r0['holds_compress']}; total task-steps "
              f"{r0['total_steps']}",
              f"- 比較 (seed 0): hard 3/12, soft 4/12 (§12.9.8), "
              f"softA 6/12 (§12.9.9)", ""]
    for seed in args.seed_list:
        r = all_results[seed]
        lines.append(f"- seed {seed}: capacity {r['capacity']}/{K} in "
                     f"{r['union']} units")
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_caseB.md", table)
    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args),
               "results": {str(sd): all_results[sd] for sd in all_results}})


if __name__ == "__main__":
    main()
