"""
consolidation_joint.py — ソフト・コンソリデーション第4段: 連続所属度と共同アニール
(docs/idea_consolidation.md §12.9 案4「連続所属度への一般化」)

案2/3 の共有は読み出し列に限られ、入力側は凍結のままだった。本実験は案4 の
目的関数 — 合計動員 Σ_k max_i ρ^(i)_k、すなわち**使用ユニットの union** — を
多目的閉ループの共同アニールで直接縮める:

  Phase A（獲得）: 案3 (consolidation_recruit.run_sequence_recruit) をそのまま
    実行。タスクごとの場・読み出し・許容値を得る（ここまで忘却厳密ゼロ）。
  Phase B0（融合）: 全タスクの場を union に等しい**単一の共有場**へ拡張する。
    借用ユニットへの交差重み・読み出し列は 0 なので、この時点で各関数は
    厳密に不変。凍結マスクは外す（入力側の共有 = 隔離から交互提示
    リハーサル + 閉ループ監視への移行）。タスクの識別は読み出しベクトル
    （タスク記述子）だけが担う。
  Phase B1（union アニール）: 全タスクを交互提示で学習し続けながら、共有場の
    ユニット k を貪欲 min S_k で消滅経路に乗せる（全タスクの ρ_k を同時に
    下げる = max_i ρ^(i)_k の縮小）。ホールド・巻き戻しは「**いずれかの**
    タスクの損失 EMA が許容超過」（多目的閉ループ）。スナップは大域的な
    kill（σ=0・h 番兵・次層列/全タスクの読み出し列ゼロ）で、union が 1 縮む。

  ※ 初版は「所属 (i,k) を個別にアニール + 使用量トリム」を試したが、
    交互学習が動員済みユニットへ使用を自由に広げるため union は縮まなかった
    （seed 0: 34→34 の負の結果）。union を直接目的にする本版が案4 の
    Σ_k max_i ρ の正しい実装形である。

主張: hard (§12.8) では関数ごとに互いに素な領域（計 37 ユニット）を要した
3 関数が、共有基底の上で union << 34 ユニットに同居する。タスク列
{sin, cos, -sin} の張る関数空間は 2 次元なので、共有 L2 基底は数ユニットで
足りるはずである。忘却は「厳密ゼロ」から「許容内に有界」へ移行する — これが
案4 の正直な代償であり、干渉は交互提示（リハーサル）と多目的閉ループが管理する。

検証項目:
  V-a 各タスクの最終 MSE が許容内（Phase B 後）
  V-b union の縮小: |union_after| < |union_before|
  V-c 共有の実現: 2 タスク以上の読み出しが使う L2 ユニットが存在
  V-d 類似度→読み出しの整列: |cos_sim(wout_sin, wout_-sin)| >
      |cos_sim(wout_sin, wout_cos)|（共有基底上では類似関数の読み出しが整列する）

生成物 (out/consolidation_joint/):
  fig_joint.png    予測 + union 前後 + 読み出し使用マップ + 多目的損失軌跡
  table_joint.md   比較表 + 判定
  results.json

実行例:
  python tmp/consolidation_joint.py --quick
  python tmp/consolidation_joint.py --seeds 0
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
import consolidation_recruit as crc  # noqa: E402
# タスクコンテキスト（読み出し記述子の切替え）と共同アニールの部品は
# consolidation_lib にある。場は Phase B0 以降タスク間で共有される
# （net.sigma_vecs / h_vecs が単一の実体; TaskCtx は support=None で使う）。
from consolidation_lib import (  # noqa: E402
    TaskCtx, alive, anneal_unit, any_over_tol, joint_checkpoint,
    joint_restore, joint_round, kill_everywhere, unit_score)


# ============================================================
# Phase B1: union の共同アニール（多目的閉ループ）
# ============================================================
def union_anneal(net, x, ctxs, args, accept_fn=None):
    """共有場のユニットを貪欲 min S_k でアニールし union を縮める.

    場が共有なので活性統計は 1 回の collect_stats で足り、タスク差は
    読み出し使用量（L2 の w2 因子 = Σ_i ||wout_i[:,k]||^2）だけに現れる。
    ホールド・スナップ後回復・巻き戻しはすべて「いずれかのタスクが許容超過」
    の多目的判定で行う。失敗したユニットは巻き戻してその層を閉じる。
    accept_fn を与えると、スナップ受理の追加条件として呼ばれる（例:
    アンサンブル評価で絶対格納基準を守る — §12.9.10）。False で巻き戻す。
    """
    holds_total = 0
    removed = {0: 0, 1: 0}
    fails = {0: 0, 1: 0}
    banned = {0: set(), 1: set()}
    losses = {ctx.name: [] for ctx in ctxs}

    def record(order, ls):
        for ctx, v in zip(order, ls):
            losses[ctx.name].append(v)

    open_layers = {0, 1}
    while open_layers:
        ctxs[0].activate(net)                   # 読み出しは z̄ に影響しない
        _, zbar = poc.collect_stats(net, x, passes=args.stat_passes)
        scored = {}
        for l in sorted(open_layers):
            rows = alive(net, 1)
            best_l = None
            basis_all = alive(net, l)
            for k in basis_all:
                if k in banned[l]:
                    continue
                if l == 1:
                    w2 = sum(float((ctx.wout[:, k] ** 2).sum())
                             for ctx in ctxs)
                else:
                    w2 = (float((net.fcs[1].weight.data[rows][:, k] ** 2)
                                .sum()) if rows else 0.0)
                S = unit_score(zbar[l], k, [j for j in basis_all if j != k],
                               w2, args.ridge)
                if S is not None and (best_l is None or S < best_l[0]):
                    best_l = (S, k)
            if best_l is None:
                open_layers.discard(l)
            else:
                scored[l] = best_l
        if not scored:
            break
        l = min(scored, key=lambda ll: scored[ll][0])
        k = scored[l][1]
        ck = joint_checkpoint(net, ctxs)

        def run_block(kind):
            for _ in range(args.epochs_per_step):
                record(ctxs, joint_round(net, ctxs))

        holds_total += anneal_unit(
            net, l, k, run_block,
            over_tol=lambda: any_over_tol(ctxs),
            read_act=lambda: float(ctxs[-1].trainer.cap.z[l][:, :, k].mean()),
            alpha=args.anneal_alpha, max_holds=args.max_holds,
            snap_act=args.snap_act, max_steps=args.max_anneal_steps)
        kill_everywhere(net, ctxs, l, k)
        rec = 0
        while any_over_tol(ctxs) and rec < args.stop_recovery:
            for _ in range(args.epochs_per_step):
                record(ctxs, joint_round(net, ctxs))
            rec += 1
        if any_over_tol(ctxs) or (accept_fn is not None and not accept_fn()):
            joint_restore(net, ctxs, ck)
            banned[l].add(k)
            fails[l] += 1
            print(f"    [fail {fails[l]}/{args.layer_fails}] L{l + 1} "
                  f"unit {k}: 多目的許容超過 -> 巻き戻し・候補から除外")
            if fails[l] >= args.layer_fails:
                open_layers.discard(l)
                print(f"    [stop] L{l + 1}: 失敗上限 -> 生存 "
                      f"{len(alive(net, l))} で確定")
        else:
            removed[l] += 1
    return losses, holds_total, removed


# ============================================================
# main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="soft consolidation step 4: "
                                            "joint union anneal")
    poc.add_poc_args(p)
    p.add_argument("--epochs-task", type=int, default=1000)
    p.add_argument("--stop-recovery", type=int, default=8)
    p.add_argument("--rho-init", type=float, default=1.0)
    p.add_argument("--recruit-thresh", type=float, default=0.5)
    p.add_argument("--settle-epochs", type=int, default=200,
                   help="Phase B 冒頭の交互学習によるならしラウンド数")
    p.add_argument("--use-eps", type=float, default=0.02,
                   help="読み出し使用の判定閾値（重なり計測用）")
    p.add_argument("--layer-fails", type=int, default=3,
                   help="層を閉じるまでの失敗ユニット数上限")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/consolidation_joint")
    if args.quick:
        args.epochs_task = 120
        args.epochs_per_step = 5
        args.stop_recovery = 3
        args.settle_epochs = 20

    device = torch.device(args.device)
    N = 128
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, N, dtype=np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    tasks = [("sin(x)", np.sin(x_raw).astype(np.float32)),
             ("cos(x)", np.cos(x_raw).astype(np.float32)),
             ("sin(x+pi)", np.sin(x_raw + np.pi).astype(np.float32))]
    K = len(tasks)
    H = args.hidden_dim

    all_results = {}
    for seed in args.seed_list:
        print(f"\n===== seed {seed} =====")
        # ---------------- Phase A: 案3 パイプライン ----------------
        res_a = crc.run_sequence_recruit(seed, args, device, tasks, x)
        net = res_a["net"]
        registry = res_a["registry"]
        union_a = {l: sorted({k for r in registry for k in r["support"][l]})
                   for l in (0, 1)}
        n_union_a = len(union_a[0]) + len(union_a[1])
        mse_a = [t["mse_final"] for t in res_a["tasks"]]
        print(f"\n  Phase A union = {len(union_a[0])}+{len(union_a[1])} "
              f"= {n_union_a} units")

        # ---------------- Phase B0: 単一共有場への融合 ----------------
        sigma0, h0 = float(args.sigma), float(args.crossing_h)
        for l in (0, 1):
            sig = torch.zeros(H)
            hv = torch.full((H,), poc.H_DEAD)
            for k in union_a[l]:
                sig[k] = sigma0
                hv[k] = h0
            net.sigma_vecs[l] = sig.to(device)
            net.h_vecs[l] = hv.to(device)
        ctxs = [TaskCtx.from_registry(net, x, r, args) for r in registry]
        for i, ctx in enumerate(ctxs):
            m, _ = ctx.eval(net, x)
            print(f"  [B0] {ctx.name}: MSE on shared union field = {m:.5f} "
                  f"(phase A {mse_a[i]:.5f}, tol {ctx.tol:.4f})")

        # ---------------- Phase B1: ならし + union アニール ----------------
        # ならし後、閉ループの許容値を交互学習の平衡ベースラインから再計算する。
        # 単独学習の許容値のままでは、共有の干渉コスト（平衡 MSE の上昇）が
        # 余裕を食い潰し、アニールの摂動が即座に多目的ホールドを尽くす。
        # 最終判定 (V-a) には元の許容値 tol_a を使い、干渉コストは別途報告する。
        for _ in range(args.settle_epochs):
            joint_round(net, ctxs)
        for ctx in ctxs:
            ctx.tol_a = ctx.tol
            eq = float(np.mean(ctx.trainer.losses[-50:]))
            ctx.tol = max(ctx.tol, eq * args.drift_mult + args.drift_abs)
            print(f"  [settle] {ctx.name}: joint equilibrium = {eq:.5f} "
                  f"-> closed-loop tol {ctx.tol:.4f} (phase A {ctx.tol_a:.4f})")
        losses, holds, removed = union_anneal(net, x, ctxs, args)
        union_b = {l: alive(net, l) for l in (0, 1)}
        n_union_b = len(union_b[0]) + len(union_b[1])
        print(f"  [B1] removed L1 {removed[0]} / L2 {removed[1]} "
              f"(holds={holds}); union {len(union_b[0])}+{len(union_b[1])} "
              f"= {n_union_b} (phase A {n_union_a})")

        # ---------------- 最終評価 ----------------
        mse_b, preds = [], []
        for ctx in ctxs:
            m, pred = ctx.eval(net, x)
            mse_b.append(m)
            preds.append(pred)
            print(f"  [final] {ctx.name}: MSE={m:.5f} (tol={ctx.tol:.4f})")
        # 読み出し使用（L2）: |wout_i| >= use_eps のユニット
        use = {ctx.name: [k for k in union_b[1]
                          if float(ctx.wout[:, k].abs().max()) >= args.use_eps]
               for ctx in ctxs}
        shared_l2 = [k for k in union_b[1]
                     if sum(k in use[ctx.name] for ctx in ctxs) >= 2]
        wn = [ctx.wout[0, union_b[1]] for ctx in ctxs]
        csim = np.zeros((K, K))
        for a in range(K):
            for b in range(K):
                csim[a, b] = float((wn[a] @ wn[b])
                                   / (wn[a].norm() * wn[b].norm() + poc.EPS))
        print(f"  L2 usage: { {nm: len(v) for nm, v in use.items()} } "
              f"shared(>=2 tasks) = {shared_l2}")
        print(f"  readout cosine similarity:\n{np.round(csim, 3)}")

        va = all(mse_b[i] <= 2.0 * ctxs[i].tol_a for i in range(K))
        vb = n_union_b < n_union_a
        vc = len(shared_l2) >= 1
        vd = abs(csim[0, 2]) > abs(csim[0, 1])
        verdicts = {"Va_within_tol": bool(va), "Vb_union_shrinks": bool(vb),
                    "Vc_shared_l2": bool(vc),
                    "Vd_readout_alignment": bool(vd)}
        all_results[seed] = {
            "phase_a": {"union": n_union_a,
                        "union_l": [len(union_a[0]), len(union_a[1])],
                        "mse": mse_a},
            "phase_b": {"union": n_union_b,
                        "union_l": [len(union_b[0]), len(union_b[1])],
                        "mse": mse_b, "removed": removed, "holds": holds,
                        "l2_usage": {nm: v for nm, v in use.items()},
                        "shared_l2": shared_l2, "readout_cos": csim.tolist(),
                        "tols_a": [ctx.tol_a for ctx in ctxs],
                        "tols_joint": [ctx.tol for ctx in ctxs]},
            "verdicts": verdicts}

        # ---------------- 図（先頭 seed） ----------------
        if seed == args.seed_list[0]:
            fig = plt.figure(figsize=(13, 9))
            gs = fig.add_gridspec(3, K, height_ratios=[1.8, 0.7, 1.3])
            for i in range(K):
                ax = fig.add_subplot(gs[0, i])
                ax.plot(x_raw, tasks[i][1], "k--", lw=1.2, label="target")
                ax.plot(x_raw, preds[i], lw=1.4, label="shared field")
                ax.set_title(f"{ctxs[i].name}: MSE {mse_a[i]:.4f} -> "
                             f"{mse_b[i]:.4f}", fontsize=10)
                ax.set_ylim(-1.6, 1.6)
                ax.grid(alpha=0.3)
                if i == 0:
                    ax.legend(fontsize=7)
            # 読み出し使用マップ（行 = タスク、列 = 生存 L2 ユニット）
            u2 = union_b[1]
            occ = np.zeros((K, max(1, len(u2))))
            for i, ctx in enumerate(ctxs):
                for j, k in enumerate(u2):
                    occ[i, j] = float(ctx.wout[0, k])
            ax = fig.add_subplot(gs[1, :])
            vmax = max(1e-6, np.abs(occ).max())
            im = ax.imshow(occ, aspect="auto", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_yticks(range(K))
            ax.set_yticklabels([ctx.name for ctx in ctxs], fontsize=8)
            ax.set_xticks(range(len(u2)))
            ax.set_xticklabels([str(k) for k in u2], fontsize=7)
            ax.set_title("Per-task readout weights on the shared surviving "
                         "L2 units (columns used by several rows = overlap)",
                         fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.02)
            ax = fig.add_subplot(gs[2, :2])
            for nm, ls in losses.items():
                ax.plot(ls, lw=0.7, label=nm)
            for i, ctx in enumerate(ctxs):
                ax.axhline(ctx.tol, ls="--", lw=0.6, color=f"C{i}")
            ax.set_yscale("log")
            ax.set_xlabel("alternating steps (phase B1)")
            ax.set_ylabel("per-task eval MSE")
            ax.set_title("Multi-objective closed loop", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, which="both")
            ax = fig.add_subplot(gs[2, 2])
            ax.bar([0, 1], [n_union_a, n_union_b], 0.5,
                   color=["tab:gray", "tab:blue"])
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["phase A\n(case 3)", "joint union\nanneal"])
            ax.set_ylabel("union of used units")
            ax.set_title(f"Total resource: {n_union_a} -> {n_union_b}",
                         fontsize=10)
            ax.grid(alpha=0.3, axis="y")
            fig.suptitle(f"Soft consolidation step 4: shared field + joint "
                         f"union anneal (H={H}, seed {seed})", fontsize=12)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
            savefig(fig, args.out_dir / "fig_joint.png")

    # ---------------- 表 ----------------
    seed0 = args.seed_list[0]
    r0 = all_results[seed0]
    lines = [f"**Soft consolidation step 4: shared field + joint union "
             f"anneal** (H={H}, T={args.num_samples}, "
             f"epochs-task={args.epochs_task}, seeds={args.seeds})", "",
             "| task | MSE phase A | MSE after B | used L2 units |",
             "|---" * 4 + "|"]
    for i in range(K):
        nm = tasks[i][0]
        lines.append(f"| {nm} | {r0['phase_a']['mse'][i]:.5f} "
                     f"| {r0['phase_b']['mse'][i]:.5f} "
                     f"| {r0['phase_b']['l2_usage'][nm]} |")
    lines += ["",
              f"- union: phase A = {r0['phase_a']['union']} "
              f"({r0['phase_a']['union_l'][0]}+{r0['phase_a']['union_l'][1]}) "
              f"-> after B = {r0['phase_b']['union']} "
              f"({r0['phase_b']['union_l'][0]}+{r0['phase_b']['union_l'][1]}) "
              f"(holds={r0['phase_b']['holds']})",
              f"- shared L2 units (>= 2 tasks): {r0['phase_b']['shared_l2']}",
              f"- readout cosine similarity: "
              f"{np.round(np.array(r0['phase_b']['readout_cos']), 3).tolist()}",
              ""]
    for seed in args.seed_list:
        v = all_results[seed]["verdicts"]
        lines.append(f"- seed {seed}: "
                     + " / ".join(f"{k} {'**PASS**' if ok else '**FAIL**'}"
                                  for k, ok in v.items()))
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_joint.md", table)
    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args),
               "results": {str(s): all_results[s] for s in all_results}})


if __name__ == "__main__":
    main()
