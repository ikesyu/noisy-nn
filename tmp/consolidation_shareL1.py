"""
consolidation_shareL1.py — 案A: 獲得時からの L1 入力側共有（二層共有）
(docs/idea_consolidation.md §12.9.9)

容量研究 (§12.9.8) の知見3 — 読み出し共有は部分空間（周波数）をまたがない —
への対処。読み出し共有の凍結規約のうち「過去 L1 列 -> 現役 L2 行の交差重みを
0 に凍結」だけを緩め、**新タスクの L2 ユニットが過去 L1 のバンプ基底を読める**
ようにする（share_l1 フラグ）:

  - 過去 L1 の入力側 (W1 行) と過去 L2 行は凍結のまま。新タスクが育てる
    交差重み W2[新 L2 行, 過去 L1 列] は、過去タスクの推論では相手行が
    休眠 (z=0) のため機能に影響しない -> **忘却は厳密ゼロのまま**
    （検証は W2 行全体のスナップショット full_rows で行う）。
  - L1 語彙は（L2 語彙のプローブ動員とは独立に）常時全動員。
  - アニールの L1 候補の補償基底に語彙 L1 を含める（交差列が学習可能に
    なったので補償が流せる）。

検証は容量研究と同一のプロトコル（12 タスク sin(kx+φ), k=1..3、掃除あり）:
  soft  : 案3（読み出し共有のみ）— §12.9.8 の基準アームの再現
  softA : 案3 + share_l1（二層共有）
続けて softA の最終状態に compress（案4）+ retry（失敗タスク再提示、share_l1）。

検証項目:
  V-a 忘却厳密ゼロ: softA 全タスクの drift（W2 行全体）= 0
  V-b 容量: capacity(softA) > capacity(soft)
  V-c 資源効率: ユニット/関数 (softA) < (soft)
  V-d パイプライン: softA+compress+retry >= 8（読み出し共有の到達点）

生成物 (out/consolidation_shareL1/):
  fig_shareL1.png / table_shareL1.md / results.json

実行例:
  python tmp/consolidation_shareL1.py --quick
  python tmp/consolidation_shareL1.py --seeds 0
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
import consolidation_capacity as ccap  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="two-tier sharing (case A): "
                                            "let new L2 read the past L1 basis")
    poc.add_poc_args(p)
    p.add_argument("--epochs-task", type=int, default=600)
    p.add_argument("--stop-recovery", type=int, default=8)
    p.add_argument("--rho-init", type=float, default=1.0)
    p.add_argument("--recruit-thresh", type=float, default=0.5)
    p.add_argument("--settle-epochs", type=int, default=100)
    p.add_argument("--layer-fails", type=int, default=3)
    p.add_argument("--use-eps", type=float, default=0.02)
    p.add_argument("--mse-star", type=float, default=0.05)
    p.add_argument("--n-phases", type=int, default=4)
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/consolidation_shareL1")
    args.cleanup = True                       # 掃除は常時有効 (§12.9.8 追記)
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
    tasks = ccap.make_tasks(x_raw, freqs, args.n_phases)
    K = len(tasks)
    H = args.hidden_dim
    thr = args.mse_star

    all_results = {}
    for seed in args.seed_list:
        print(f"\n===== seed {seed}  ({K} tasks) =====")
        res = {}

        # ---------------- arm: soft (読み出し共有のみ; 基準) ----------------
        out_s = crc.run_sequence_recruit(seed, args, device, tasks, x,
                                         cleanup_thr=thr)
        res["soft"] = {
            "mses": [t["mse_final"] for t in out_s["tasks"]],
            "own": [t["n_own"]["0"] + t["n_own"]["1"] for t in out_s["tasks"]],
            "drift": [t["drift"] for t in out_s["tasks"]]}
        res["soft"]["stored"] = ccap.stored_flags(res["soft"]["mses"], thr)

        # ---------------- arm: softA (二層共有) ----------------
        out_a = crc.run_sequence_recruit(seed, args, device, tasks, x,
                                         cleanup_thr=thr, share_l1=True)
        res["softA"] = {
            "mses": [t["mse_final"] for t in out_a["tasks"]],
            "own": [t["n_own"]["0"] + t["n_own"]["1"] for t in out_a["tasks"]],
            "drift": [t["drift"] for t in out_a["tasks"]]}
        res["softA"]["stored"] = ccap.stored_flags(res["softA"]["mses"], thr)
        net = out_a["net"]
        registry = out_a["registry"]

        # ---------------- compress + retry (softA の状態から) ----------------
        # 圧縮対象は整理直後 MSE 基準（掃除と同じ基準）で選ぶ: 最終評価境界の
        # 揺らぎで stored から外れたタスクの領域を孤児にしないため。
        stored_idx = [i for i, r in enumerate(registry)
                      if r["mse_at_consolidation"] <= thr]
        extra_tasks = [tasks[i] for i in range(K) if i not in stored_idx]
        ctxs, U, n_before, n_after = ccap.compress(net, x, registry,
                                                   stored_idx, args, device,
                                                   accept_thr=thr)
        mses_c = ccap.eval_shared(net, x, ctxs, U, args.sigma, args.crossing_h)
        ext = (ccap.extend(net, x, U, extra_tasks, args, device,
                           share_l1=True) if extra_tasks else [])
        mses_c2 = ccap.eval_shared(net, x, ctxs, U, args.sigma,
                                   args.crossing_h)
        for i, ctx in enumerate(ctxs):
            print(f"  [compress] {ctx.name}: MSE after compress "
                  f"{mses_c[i]:.5f} / after retry {mses_c2[i]:.5f}")
        res["compress"] = {
            "union_before": n_before, "union_after": n_after,
            "mses_after_retry": mses_c2,
            "stored_after": ccap.stored_flags(mses_c2, thr)}
        res["retry"] = {
            "tasks": [{k: v for k, v in e.items() if k != "descriptor"}
                      for e in ext],
            "stored": ccap.stored_flags([e["mse"] for e in ext], thr)}

        # ---------------- 集計・判定 ----------------
        cap_s = sum(res["soft"]["stored"])
        cap_a = sum(res["softA"]["stored"])
        cap_total = (sum(res["compress"]["stored_after"])
                     + sum(res["retry"]["stored"]))
        upf_s = (sum(o for o, ok in zip(res["soft"]["own"],
                                        res["soft"]["stored"]) if ok)
                 / max(1, cap_s))
        upf_a = (sum(o for o, ok in zip(res["softA"]["own"],
                                        res["softA"]["stored"]) if ok)
                 / max(1, cap_a))
        max_drift_a = max(res["softA"]["drift"])
        verdicts = {
            "Va_zero_forgetting": bool(max_drift_a == 0.0),
            "Vb_capacity": bool(cap_a > cap_s),
            "Vc_units_per_fn": bool(upf_a < upf_s),
            "Vd_pipeline": bool(cap_total >= 8)}
        res["summary"] = {
            "capacity_soft": cap_s, "capacity_softA": cap_a,
            "capacity_total": cap_total,
            "units_per_fn_soft": upf_s, "units_per_fn_softA": upf_a,
            "max_drift_softA": max_drift_a}
        all_results[seed] = {k: v for k, v in res.items()}
        all_results[seed]["verdicts"] = verdicts
        print(f"\n  ===== capacity: soft {cap_s}/{K} -> softA {cap_a}/{K} "
              f"-> +compress+retry {cap_total}/{K}  "
              f"(units/fn {upf_s:.1f} -> {upf_a:.1f}; "
              f"max drift softA = {max_drift_a:.2e}) =====")

        # ---------------- 図（先頭 seed） ----------------
        if seed == args.seed_list[0]:
            fig = plt.figure(figsize=(13, 5))
            gs = fig.add_gridspec(1, 3)
            ax = fig.add_subplot(gs[0, 0])
            for arm, color in (("soft", "tab:blue"), ("softA", "tab:red")):
                ax.step(range(1, K + 1), np.cumsum(res[arm]["stored"]),
                        where="post", color=color, lw=1.6, label=arm)
            n_ret = len(res["retry"]["stored"])
            if n_ret:
                base_c = sum(res["compress"]["stored_after"])
                ax.step(range(K + 1, K + n_ret + 1),
                        [base_c + s for s in np.cumsum(res["retry"]["stored"])],
                        where="post", color="tab:green", lw=1.6,
                        label="softA + compress + retry")
            ax.set_xlabel("tasks presented")
            ax.set_ylabel(f"functions stored (MSE <= {thr})")
            ax.set_title("Capacity")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            ax = fig.add_subplot(gs[0, 1])
            idx = np.arange(K)
            wd = 0.4
            ax.bar(idx - wd / 2, res["soft"]["own"], wd, color="tab:blue",
                   label="soft")
            ax.bar(idx + wd / 2, res["softA"]["own"], wd, color="tab:red",
                   label="softA (two-tier)")
            ax.set_xticks(idx)
            ax.set_xticklabels([t[0] for t in tasks], rotation=60, fontsize=6)
            ax.set_ylabel("own units consumed")
            ax.set_title("Per-task unit consumption")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis="y")

            ax = fig.add_subplot(gs[0, 2])
            ax.plot(range(1, K + 1), res["soft"]["mses"], "o-", c="tab:blue",
                    lw=1.0, ms=4, label="soft")
            ax.plot(range(1, K + 1), res["softA"]["mses"], "o-", c="tab:red",
                    lw=1.0, ms=4, label="softA")
            if n_ret:
                ax.plot(range(K + 1, K + n_ret + 1),
                        [e["mse"] for e in res["retry"]["tasks"]], "s-",
                        c="tab:green", lw=1.0, ms=4, label="retry")
            ax.axhline(thr, ls="--", c="k", lw=0.9)
            ax.set_yscale("log")
            ax.set_xlabel("task index")
            ax.set_ylabel("final MSE")
            ax.set_title("Final MSE per task")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, which="both")
            fig.suptitle(f"Two-tier sharing (case A): capacity study "
                         f"(H={H}, seed {seed})", fontsize=12)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
            savefig(fig, args.out_dir / "fig_shareL1.png")

    # ---------------- 表 ----------------
    seed0 = args.seed_list[0]
    r0 = all_results[seed0]
    lines = [f"**Two-tier sharing (case A)** (H={H}, T={args.num_samples}, "
             f"epochs-task={args.epochs_task}, thr={thr}, cleanup on, "
             f"seeds={args.seeds})", "",
             "| task | soft MSE | soft own | softA MSE | softA own "
             "| softA drift |",
             "|---" * 6 + "|"]
    for i, (nm, _) in enumerate(tasks):
        ss = "**" if r0["soft"]["stored"][i] else ""
        sa = "**" if r0["softA"]["stored"][i] else ""
        lines.append(
            f"| {nm} | {ss}{r0['soft']['mses'][i]:.4f}{ss} "
            f"| {r0['soft']['own'][i]} "
            f"| {sa}{r0['softA']['mses'][i]:.4f}{sa} "
            f"| {r0['softA']['own'][i]} "
            f"| {r0['softA']['drift'][i]:.2e} |")
    for e in r0["retry"]["tasks"]:
        lines.append(f"| {e['name']} (retry) | — | — | {e['mse']:.4f} "
                     f"| {e['n_own']['0'] + e['n_own']['1']} | — |")
    s = r0["summary"]
    c = r0["compress"]
    lines += ["",
              f"- capacity: soft **{s['capacity_soft']}/{K}** -> softA "
              f"**{s['capacity_softA']}/{K}** -> +compress+retry "
              f"**{s['capacity_total']}/{K}**",
              f"- units per stored function: soft "
              f"{s['units_per_fn_soft']:.1f} -> softA "
              f"{s['units_per_fn_softA']:.1f}",
              f"- compress: union {c['union_before']} -> {c['union_after']}",
              f"- max drift (softA, full W2 rows): "
              f"{s['max_drift_softA']:.2e}", ""]
    for seed in args.seed_list:
        v = all_results[seed]["verdicts"]
        lines.append(f"- seed {seed}: "
                     + " / ".join(f"{k} {'**PASS**' if ok else '**FAIL**'}"
                                  for k, ok in v.items()))
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_shareL1.md", table)

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
