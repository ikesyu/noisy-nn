"""
consolidation_capacity.py — 容量研究: 同一サイズのネットワークに何関数入るか
(docs/idea_consolidation.md §12.9.8)

「ソフト・コンソリデーションは訓練済み表現を再利用するので、同じサイズの
ネットワークにより多くの関数を覚えられるはず」という仮説を、固定 H の
ネットワークへの逐次格納で検証する。

タスク族: y = sin(k x + φ)、周波数 k ∈ {1, 2, 3} × 位相 4 種 = 12 タスク。
関数空間の次元は 6（周波数ごとに 2）なので、各周波数ブロックの最初の
~2 タスクは新しい次元（新ユニットが必要）、残りは既存の張る空間内
（再利用可能）という規則的な構造を持つ。読み出し共有の語彙は過去の L2
チューニング曲線なので、再利用は同一周波数内でしか効かない — ソフトの
限界（周波数をまたぐ L1 の再利用には入力側共有 = 案4 が要る）も同時に測る。

アーム:
  hard : §12.8。互いに素な領域への逐次格納 (csoft.run_sequence("hard"))。
  soft : 案3 (crc.run_sequence_recruit)。プローブ動員 + 読み出し共有。
続けて soft の最終状態に対し:
  compress : 案4。格納済みタスク（閾値内）の場を union に融合し、交互提示
             リハーサル + 多目的閉ループで union を圧縮 (cjoint.union_anneal)。
  retry    : 圧縮後の union を単一の凍結語彙として扱い（読み出し共有 =
             案2 の規約に戻る; 以後の忘却は再び厳密ゼロ）、解放された
             ユニットに**容量切れで失敗したタスクを再提示**して継続格納する。

学習可能性の対照（本スクリプトの結果解釈に必須）: 全ネットワーク (H=32,
seed 0) からの単独学習で sin(2x) は MSE 0.010 (600 epochs)、sin(3x) は
0.041（限界的）、sin(4x) は 0.47（現行ハイパーパラメータでは学習不能 —
バンプ幅がノイズ幅 σ/|w| で決まり、周期がそれと同程度になるため）。
したがって容量の検証は k <= 3 の範囲で行う。

格納判定: 最終評価 MSE <= mse-star（固定の絶対閾値。学習失敗時には適応的
許容値が無意味になる — §12.9.1 知見2 — ため使わない）。

生成物 (out/consolidation_capacity/):
  fig_capacity.png  容量曲線 + ユニット消費 + 最終 MSE + 資源軌跡
  table_capacity.md 比較表
  results.json

実行例:
  python tmp/consolidation_capacity.py --quick
  python tmp/consolidation_capacity.py --seeds 0
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
import consolidation_soft as csoft  # noqa: E402
import consolidation_recruit as crc  # noqa: E402
import consolidation_joint as cjoint  # noqa: E402
from consolidation_lib import (  # noqa: E402
    TaskCtx, alive, eval_with_descriptor, freeze_masks, joint_round, predict,
    region_drift, region_snapshot, set_field, zero_cross_columns)


def make_tasks(x_raw, freqs, n_phases):
    tasks = []
    for k in freqs:
        for i in range(n_phases):
            phi = i * 0.4 * np.pi
            name = f"sin({k}x+{phi / np.pi:.1f}pi)" if phi else f"sin({k}x)"
            tasks.append((name, np.sin(k * x_raw + phi).astype(np.float32)))
    return tasks


def stored_flags(mses, thr):
    return [m <= thr for m in mses]


# ============================================================
# 案4 圧縮: 格納済みタスクの場を union へ融合して共同アニール
# ============================================================
def compress(net, x, registry, stored_idx, args, device,
             accept_thr: float = None):
    """accept_thr を与えると、各ユニットのスナップ受理条件に「全圧縮対象
    タスクのアンサンブル評価 MSE <= accept_thr」を追加する（§12.9.10:
    閉ループの相対許容だけでは限界品質のタスクが絶対基準を割りうる）。"""
    if not stored_idx:
        print("  [compress] no stored task within threshold -> skip")
        return [], {0: [], 1: []}, 0, 0
    H = args.hidden_dim
    union = {l: sorted({k for i in stored_idx
                        for k in registry[i]["support"][l]}) for l in (0, 1)}
    sigma0, h0 = float(args.sigma), float(args.crossing_h)
    for l in (0, 1):
        sig = torch.zeros(H)
        hv = torch.full((H,), poc.H_DEAD)
        for k in union[l]:
            sig[k] = sigma0
            hv[k] = h0
        net.sigma_vecs[l] = sig.to(device)
        net.h_vecs[l] = hv.to(device)
    ctxs = [TaskCtx.from_registry(net, x, registry[i], args)
            for i in stored_idx]
    for _ in range(args.settle_epochs):
        joint_round(net, ctxs)
    for ctx in ctxs:
        ctx.tol_a = ctx.tol
        eq = float(np.mean(ctx.trainer.losses[-50:]))
        ctx.tol = max(ctx.tol, eq * args.drift_mult + args.drift_abs)
    accept = None
    if accept_thr is not None:
        def accept():
            for ctx in ctxs:
                ctx.activate(net)
                pred = predict(net, x, passes=16)
                if float(np.mean((pred - ctx.target) ** 2)) > accept_thr:
                    return False
            return True
    _, holds, removed = cjoint.union_anneal(net, x, ctxs, args,
                                            accept_fn=accept)
    U = {l: alive(net, l) for l in (0, 1)}
    n_before = len(union[0]) + len(union[1])
    n_after = len(U[0]) + len(U[1])
    print(f"  [compress] union {len(union[0])}+{len(union[1])} = {n_before} "
          f"-> {len(U[0])}+{len(U[1])} = {n_after} "
          f"(removed L1 {removed[0]} / L2 {removed[1]}, holds={holds})")
    for ctx in ctxs:
        ctx.trainer.close()
    return ctxs, U, n_before, n_after


def eval_shared(net, x, ctxs, U, sigma0, h0):
    """圧縮後の共有場 U 上で各タスク（読み出し記述子）を評価する."""
    set_field(net, U, sigma0, h0)
    mses = []
    for ctx in ctxs:
        net.fcs[2].weight.data.copy_(ctx.wout)
        net.fcs[2].bias.data.copy_(ctx.b_out)
        pred = predict(net, x, passes=16)
        mses.append(float(np.mean((pred - ctx.target) ** 2)))
    return mses


# ============================================================
# 継続格納: 圧縮済み union を単一の凍結語彙として追加タスクを格納（案2 の規約）
# ============================================================
def extend(net, x, U, extra_tasks, args, device, share_l1: bool = False):
    """案2 の逐次ループを初期 past = U（圧縮済み共有語彙）で続ける.

    再提示中に新しく格納した領域も past（凍結語彙）へ蓄積されるので、
    同一周波数の後続タスクは新領域の語彙を再利用できる。share_l1=True は
    二層共有（§12.9.9 案A）: 新タスクの L2 行が past の L1 基底を読める。
    """
    H = args.hidden_dim
    past = {l: list(U[l]) for l in (0, 1)}
    free = {l: [k for k in range(H) if k not in U[l]] for l in (0, 1)}
    results = []
    for name, target in extra_tasks:
        print(f"\n  ===== [extend] {name}  (free L1 {len(free[0])} / "
              f"L2 {len(free[1])}, vocab {len(past[0])}+{len(past[1])}) =====")
        net.fcs[2].weight.data.zero_()
        net.fcs[2].bias.data.zero_()
        zero_cross_columns(net, past)
        masks = freeze_masks(H, past, device, share_l1=share_l1)
        mobil = {l: sorted(set(free[l]) | set(past[l])) for l in (0, 1)}
        set_field(net, mobil, args.sigma, args.crossing_h)
        t = torch.tensor(target, device=device).unsqueeze(1)
        trainer = poc.CovJacTrainer(net, x, t, lr=args.pre_lr, opt=args.opt,
                                    jac_ema=args.jac_ema)
        trainer.grad_masks = masks
        for e in range(args.epochs_task):
            loss = trainer.step()
            if e % max(1, args.epochs_task // 2) == 0:
                print(f"    [learn] epoch {e:5d} mse={loss:.5f}")
        trainer.close()
        base = float(np.mean(trainer.losses[-min(100, len(trainer.losses)):]))
        tol = base * args.drift_mult + args.drift_abs
        _, holds = csoft.anneal_until_stop(net, x, target, tol, args,
                                           eligible=free, grad_masks=masks,
                                           vocab=past, share_l1=share_l1)
        region = {l: [k for k in free[l] if float(net.sigma_vecs[l][k]) > 0]
                  for l in (0, 1)}
        r = {"name": name, "target": target, "region": region,
             "support": {l: sorted(region[l] + past[l]) for l in (0, 1)},
             "sigma0": float(args.sigma), "h0": float(args.crossing_h),
             "wout": net.fcs[2].weight.data.clone(),
             "b_out": net.fcs[2].bias.data.clone(), "tol": tol}
        pred = eval_with_descriptor(net, x, r)
        mse = float(np.mean((pred - target) ** 2))
        print(f"    own L1 {len(region[0])} + L2 {len(region[1])}; "
              f"MSE={mse:.5f}")
        results.append({"name": name, "mse": mse, "descriptor": r,
                        "n_own": {"0": len(region[0]), "1": len(region[1])},
                        "holds": holds})
        # 格納に成功した領域だけを凍結語彙へ蓄積する
        if mse <= args.mse_star:
            past = {l: sorted(past[l] + region[l]) for l in (0, 1)}
            free = {l: [k for k in free[l] if k not in region[l]]
                    for l in (0, 1)}
        elif args.cleanup:
            # 掃除: 失敗タスクの残骸を回収（kill して free に残す）
            for l in (0, 1):
                for k in region[l]:
                    poc.kill_unit(net, l, k)
            print(f"    [cleanup] 格納失敗 (MSE > {args.mse_star}) -> "
                  f"L1 {len(region[0])} + L2 {len(region[1])} を回収")
        else:
            free = {l: [k for k in free[l] if k not in region[l]]
                    for l in (0, 1)}
    return results


# ============================================================
# main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="capacity study: how many "
                                            "functions fit in one network")
    poc.add_poc_args(p)
    p.add_argument("--epochs-task", type=int, default=600)
    p.add_argument("--stop-recovery", type=int, default=8)
    p.add_argument("--rho-init", type=float, default=1.0)
    p.add_argument("--recruit-thresh", type=float, default=0.5)
    p.add_argument("--settle-epochs", type=int, default=100)
    p.add_argument("--layer-fails", type=int, default=3)
    p.add_argument("--use-eps", type=float, default=0.02)
    p.add_argument("--mse-star", type=float, default=0.05,
                   help="格納成功の絶対 MSE 閾値")
    p.add_argument("--cleanup", action="store_true",
                   help="格納失敗タスクの残骸領域を回収する（§12.9.8 の掃除）")
    p.add_argument("--n-phases", type=int, default=4)
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/consolidation_capacity")
    if args.quick:
        args.epochs_task = 120
        args.epochs_per_step = 5
        args.stop_recovery = 3
        args.settle_epochs = 20
        args.n_phases = 2

    device = torch.device(args.device)
    N = 128
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, N, dtype=np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    freqs = (1, 2) if args.quick else (1, 2, 3)
    tasks = make_tasks(x_raw, freqs, args.n_phases)
    K = len(tasks)
    H = args.hidden_dim
    thr = args.mse_star

    all_results = {}
    for seed in args.seed_list:
        print(f"\n===== seed {seed}  ({K} tasks: "
              f"{[t[0] for t in tasks]}) =====")
        res = {}

        # ---------------- arm: hard ----------------
        cthr = thr if args.cleanup else None
        out_h = csoft.run_sequence("hard", seed, args, device, tasks, x,
                                   x_raw, cleanup_thr=cthr)
        mses_h = [t["mse_final"] for t in out_h["tasks"]]
        own_h = [t["n_own"]["0"] + t["n_own"]["1"] for t in out_h["tasks"]]
        res["hard"] = {"mses": mses_h, "own": own_h,
                       "stored": stored_flags(mses_h, thr)}

        # ---------------- arm: soft (案3) ----------------
        out_s = crc.run_sequence_recruit(seed, args, device, tasks, x,
                                         cleanup_thr=cthr)
        mses_s = [t["mse_final"] for t in out_s["tasks"]]
        own_s = [t["n_own"]["0"] + t["n_own"]["1"] for t in out_s["tasks"]]
        res["soft"] = {"mses": mses_s, "own": own_s,
                       "stored": stored_flags(mses_s, thr)}
        net = out_s["net"]
        registry = out_s["registry"]

        # ---------------- 案4 圧縮 + 失敗タスクの再提示 (soft の状態から) -----
        stored_idx = [i for i, ok in enumerate(res["soft"]["stored"]) if ok]
        extra_tasks = [tasks[i] for i, ok in enumerate(res["soft"]["stored"])
                       if not ok]
        ctxs, U, n_before, n_after = compress(net, x, registry, stored_idx,
                                              args, device)
        mses_c = eval_shared(net, x, ctxs, U, args.sigma, args.crossing_h)
        snap_U = region_snapshot(net, U)
        ext = extend(net, x, U, extra_tasks, args, device)
        mses_c2 = eval_shared(net, x, ctxs, U, args.sigma, args.crossing_h)
        drift_U = region_drift(net, U, snap_U)
        for i, ctx in enumerate(ctxs):
            print(f"  [compress] {ctx.name}: MSE after compress "
                  f"{mses_c[i]:.5f} / after extend {mses_c2[i]:.5f}")
        print(f"  [extend] union drift after extension = {drift_U:.2e}")
        res["compress"] = {
            "n_stored_in": len(stored_idx), "union_before": n_before,
            "union_after": n_after,
            "mses_after_compress": mses_c, "mses_after_extend": mses_c2,
            "stored_after": stored_flags(mses_c2, thr), "drift_U": drift_U}
        res["extend"] = {"tasks": [{k: v for k, v in e.items()
                                    if k != "descriptor"} for e in ext],
                         "stored": stored_flags([e["mse"] for e in ext], thr)}

        # ---------------- 集計 ----------------
        cap_h = sum(res["hard"]["stored"])
        cap_s = sum(res["soft"]["stored"])
        cap_c = sum(res["compress"]["stored_after"])
        cap_e = sum(res["extend"]["stored"])
        summary = {
            "capacity_hard": cap_h, "capacity_soft": cap_s,
            "capacity_soft_02": sum(stored_flags(mses_s, 0.02)),
            "capacity_hard_02": sum(stored_flags(mses_h, 0.02)),
            "capacity_after_compress": cap_c,
            "capacity_retry": cap_e,
            "capacity_extended_total": cap_c + cap_e,
            "units_per_fn_hard": (sum(own_h) / max(1, cap_h)),
            "units_per_fn_soft": (sum(own_s) / max(1, cap_s)),
            "units_per_fn_compressed": n_after / max(1, cap_c)}
        res["summary"] = summary
        print(f"\n  ===== capacity (H={H}, thr={thr}): hard {cap_h}/{K}, "
              f"soft {cap_s}/{K}, after compress {cap_c} in {n_after} units, "
              f"after retry {cap_c + cap_e}/{K} =====")
        all_results[seed] = res

        # ---------------- 図（先頭 seed） ----------------
        if seed == args.seed_list[0]:
            fig = plt.figure(figsize=(13, 9))
            gs = fig.add_gridspec(2, 2)
            ax = fig.add_subplot(gs[0, 0])
            for arm, color in (("hard", "tab:gray"), ("soft", "tab:blue")):
                ax.step(range(1, K + 1), np.cumsum(res[arm]["stored"]),
                        where="post", color=color, lw=1.6, label=arm)
            tot = list(np.cumsum(res["soft"]["stored"]))
            ext_cum = [cap_c + s for s in
                       np.cumsum(res["extend"]["stored"])]
            ax.step(range(K + 1, K + len(extra_tasks) + 1), ext_cum,
                    where="post", color="tab:green", lw=1.6,
                    label="soft + compress + retry")
            ax.plot([K, K + 1], [tot[-1], ext_cum[0]], ":", c="tab:green",
                    lw=1.0)
            ax.set_xlabel("tasks presented")
            ax.set_ylabel(f"functions stored (MSE <= {thr})")
            ax.set_title("Capacity: cumulative stored functions")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            ax = fig.add_subplot(gs[0, 1])
            idx = np.arange(K)
            wd = 0.4
            ax.bar(idx - wd / 2, own_h, wd, color="tab:gray", label="hard")
            ax.bar(idx + wd / 2, own_s, wd, color="tab:blue", label="soft")
            ax.set_xticks(idx)
            ax.set_xticklabels([t[0] for t in tasks], rotation=60, fontsize=6)
            ax.set_ylabel("own units consumed")
            ax.set_title("Per-task unit consumption")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis="y")

            ax = fig.add_subplot(gs[1, 0])
            ax.plot(range(1, K + 1), mses_h, "o-", c="tab:gray", lw=1.0,
                    ms=4, label="hard")
            ax.plot(range(1, K + 1), mses_s, "o-", c="tab:blue", lw=1.0,
                    ms=4, label="soft")
            ax.plot(range(K + 1, K + len(extra_tasks) + 1),
                    [e["mse"] for e in ext], "s-", c="tab:green", lw=1.0,
                    ms=4, label="retry (after compress)")
            ax.axhline(thr, ls="--", c="k", lw=0.9, label=f"thr {thr}")
            ax.set_yscale("log")
            ax.set_xlabel("task index")
            ax.set_ylabel("final MSE")
            ax.set_title("Final MSE per task")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, which="both")

            ax = fig.add_subplot(gs[1, 1])
            occ_h = np.cumsum(own_h)
            occ_s = np.cumsum(own_s)
            ax.plot(range(1, K + 1), occ_h, "-", c="tab:gray", lw=1.6,
                    label="hard occupied")
            ax.plot(range(1, K + 1), occ_s, "-", c="tab:blue", lw=1.6,
                    label="soft occupied")
            ax.plot([K, K + 0.5], [occ_s[-1], n_after], "v-", c="tab:green",
                    lw=1.4, label=f"compress -> {n_after}")
            ext_occ = n_after + np.cumsum(
                [e["n_own"]["0"] + e["n_own"]["1"] for e in ext])
            ax.plot(range(K + 1, K + len(extra_tasks) + 1), ext_occ, "-",
                    c="tab:green", lw=1.6)
            ax.axhline(2 * H, ls=":", c="k", lw=0.9, label=f"2H = {2 * H}")
            ax.set_xlabel("tasks presented")
            ax.set_ylabel("units occupied")
            ax.set_title("Resource trajectory")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            fig.suptitle(f"Capacity study (H={H}, {K}+{len(extra_tasks)} "
                         f"tasks, seed {seed})", fontsize=12)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
            savefig(fig, args.out_dir / "fig_capacity.png")

    # ---------------- 表 ----------------
    seed0 = args.seed_list[0]
    r0 = all_results[seed0]
    lines = [f"**Capacity study** (H={H}, T={args.num_samples}, "
             f"epochs-task={args.epochs_task}, thr={thr}, "
             f"seeds={args.seeds})", "",
             f"tasks: {[t[0] for t in tasks]}; retry = soft の失敗タスク", "",
             "| task | hard MSE | hard own | soft MSE | soft own |",
             "|---" * 5 + "|"]
    for i, (nm, _) in enumerate(tasks):
        sh = "**" if r0["hard"]["stored"][i] else ""
        ss = "**" if r0["soft"]["stored"][i] else ""
        lines.append(f"| {nm} | {sh}{r0['hard']['mses'][i]:.4f}{sh} "
                     f"| {r0['hard']['own'][i]} "
                     f"| {ss}{r0['soft']['mses'][i]:.4f}{ss} "
                     f"| {r0['soft']['own'][i]} |")
    for e in r0["extend"]["tasks"]:
        lines.append(f"| {e['name']} (retry) | — | — | {e['mse']:.4f} "
                     f"| {e['n_own']['0'] + e['n_own']['1']} |")
    s = r0["summary"]
    c = r0["compress"]
    lines += ["",
              f"- capacity (MSE <= {thr}): hard = **{s['capacity_hard']}/{K}**"
              f", soft = **{s['capacity_soft']}/{K}** "
              f"(at 0.02: {s['capacity_hard_02']} / {s['capacity_soft_02']})",
              f"- units per stored function: hard {s['units_per_fn_hard']:.1f}"
              f", soft {s['units_per_fn_soft']:.1f}, after compress "
              f"{s['units_per_fn_compressed']:.1f}",
              f"- compress: union {c['union_before']} -> {c['union_after']}; "
              f"stored kept {sum(c['stored_after'])}/{c['n_stored_in']}; "
              f"drift(U) after extend = {c['drift_U']:.2e}",
              f"- total after compress + retry: "
              f"**{s['capacity_extended_total']}/{K}**",
              ""]
    for seed in args.seed_list:
        s = all_results[seed]["summary"]
        lines.append(f"- seed {seed}: hard {s['capacity_hard']}, soft "
                     f"{s['capacity_soft']}, +compress+extend "
                     f"{s['capacity_extended_total']}")
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_capacity.md", table)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k != "descriptor"}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args),
               "results": {str(sd): clean(all_results[sd])
                           for sd in all_results}})


if __name__ == "__main__":
    main()
