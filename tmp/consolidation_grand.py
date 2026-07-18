"""
consolidation_grand.py — 統一容量比較: 固定ネットワークに何関数を逐次格納できるか
(docs/idea_consolidation.md §12.9.13)

これまでにテストした全コンソリデーション獲得体制（ハード・ソフト）を、
**同一条件**で比較する。§12.9.8–12.9.12 の結果は条件（掃除・初期化・圧縮の
受理条件）を進化させながら得られたため、確定版として全アームを揃え直す。

統一条件:
  - ネットワーク: H=32 x 隠れ 2 層（計 64 ユニット）
  - 初期化: マルチスケール L1 基底 (§12.9.12) を全アーム標準
    （解像度限界を除去し、タスク族を k=4 まで拡張できる）
  - タスク族: sin(kx+φ), k ∈ {1,2,3,4} × 位相 4 種 = 16 タスク逐次
    （関数空間の次元 8; 学習可能性は §12.9.12 の対照で全周波数を確認済み）
  - 掃除 (§12.9.8 追記) 常時有効、格納判定 MSE <= 0.05、epochs-task 600

アーム（獲得体制）:
  hard  : §12.8。互いに素な領域。忘却保証は h ゲートの物理（厳密ゼロ）。
  soft2 : 案2 readout-only。過去領域を全動員し読み出し列のみ共有（厳密ゼロ）。
  soft3 : 案3 recruit。プローブで語彙を事前選択（厳密ゼロ）。
  softA : 案A 二層共有。L1 基底を凍結共有、L2+読み出しがタスク固有（厳密ゼロ）。
各アームに続けて同一のパイプライン: compress（案4 union anneal + §12.9.10 の
絶対格納基準受理）+ retry（失敗タスクの再提示; softA アームのみ share_l1）。
hard への圧縮適用（= hard 獲得 + 案4 事後圧縮）はここが初の試行になる。

案B（共有場獲得, §12.9.11）は consolidation_caseB.py --freqs 1,2,3,4
--multiscale で同一条件の別実行とし、報告時に統合する。案1 は案3 に部品と
して内包（事後剪定は support 整理であり格納能力を変えない）、案5 は逐次格納
でなく運用時ドリフト適応の手法のため、容量比較のアームには含めない。

生成物 (out/consolidation_grand/):
  table_grand.md / results.json（図は全結果統合後に別途）

実行例:
  python tmp/consolidation_grand.py --quick
  python tmp/consolidation_grand.py --seeds 0 --arms hard,soft2
  python tmp/consolidation_grand.py --seeds 0 --arms soft3,softA
"""
import argparse

import numpy as np
import torch
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")

import fncl_driver as fncl  # noqa: E402
from fncl_driver import save_json, write_text  # noqa: E402
import consolidation_poc as poc  # noqa: E402
import consolidation_soft as csoft  # noqa: E402
import consolidation_recruit as crc  # noqa: E402
import consolidation_capacity as ccap  # noqa: E402
from consolidation_multiscale import SCALES  # noqa: E402

GUARANTEE = {"hard": "厳密ゼロ（h ゲート物理）",
             "soft2": "厳密ゼロ（凍結マスク）",
             "soft3": "厳密ゼロ（凍結マスク）",
             "softA": "厳密ゼロ（凍結マスク; L1 は凍結共有）"}


def run_arm(arm, seed, args, device, tasks, x, x_raw, thr):
    if arm == "hard":
        return csoft.run_sequence("hard", seed, args, device, tasks, x,
                                  x_raw, cleanup_thr=thr)
    if arm == "soft2":
        return csoft.run_sequence("soft", seed, args, device, tasks, x,
                                  x_raw, cleanup_thr=thr)
    if arm == "soft3":
        return crc.run_sequence_recruit(seed, args, device, tasks, x,
                                        cleanup_thr=thr)
    if arm == "softA":
        return crc.run_sequence_recruit(seed, args, device, tasks, x,
                                        cleanup_thr=thr, share_l1=True)
    raise ValueError(arm)


def main():
    p = argparse.ArgumentParser(description="grand unified capacity "
                                            "comparison")
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
    p.add_argument("--freqs", type=str, default="1,2,3,4")
    p.add_argument("--arms", type=str, default="hard,soft2,soft3,softA")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/consolidation_grand")
    args.cleanup = True
    args.l1_scales = SCALES
    if args.quick:
        args.epochs_task = 120
        args.epochs_per_step = 5
        args.stop_recovery = 3
        args.settle_epochs = 20
        args.n_phases = 2
        args.freqs = "1,2"

    device = torch.device(args.device)
    N = 128
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, N, dtype=np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    freqs = tuple(int(v) for v in args.freqs.split(","))
    tasks = ccap.make_tasks(x_raw, freqs, args.n_phases)
    K = len(tasks)
    H = args.hidden_dim
    thr = args.mse_star
    arms = [a.strip() for a in args.arms.split(",")]

    all_results = {}
    for seed in args.seed_list:
        res = {}
        for arm in arms:
            print(f"\n########## seed {seed} / arm {arm} ({K} tasks) "
                  f"##########")
            out = run_arm(arm, seed, args, device, tasks, x, x_raw, thr)
            mses = [t["mse_final"] for t in out["tasks"]]
            own = [t["n_own"]["0"] + t["n_own"]["1"] for t in out["tasks"]]
            drift = [t.get("drift", 0.0) for t in out["tasks"]]
            stored = ccap.stored_flags(mses, thr)
            net, registry = out["net"], out["registry"]
            seq_epochs = (K * args.epochs_task
                          + sum(r.get("anneal_epochs", 0) for r in registry))

            # ---- 統一パイプライン: 絶対基準圧縮 + 失敗タスク再提示 ----
            stored_idx = [i for i, r in enumerate(registry)
                          if r["mse_at_consolidation"] <= thr]
            extras = [tasks[i] for i in range(K) if i not in stored_idx]
            ctxs, U, n_before, n_after = ccap.compress(
                net, x, registry, stored_idx, args, device, accept_thr=thr)
            ext = (ccap.extend(net, x, U, extras, args, device,
                               share_l1=(arm == "softA")) if extras else [])
            mses_c2 = (ccap.eval_shared(net, x, ctxs, U, args.sigma,
                                        args.crossing_h) if ctxs else [])
            stored_pipe = (ccap.stored_flags(mses_c2, thr)
                           + ccap.stored_flags([e["mse"] for e in ext], thr))
            cap_seq, cap_pipe = sum(stored), sum(stored_pipe)
            res[arm] = {
                "mses": mses, "own": own, "drift": drift, "stored": stored,
                "capacity_seq": cap_seq, "capacity_pipe": cap_pipe,
                "seq_epochs": seq_epochs,
                "union_before": n_before, "union_after": n_after,
                "retry": [{"name": e["name"], "mse": e["mse"],
                           "n_own": e["n_own"]} for e in ext],
                "mses_after_pipe": mses_c2,
                "max_drift": max(drift) if drift else 0.0}
            print(f"\n  ===== [{arm}] capacity: seq {cap_seq}/{K} -> "
                  f"pipeline {cap_pipe}/{K} (union {n_before}->{n_after}; "
                  f"max drift {res[arm]['max_drift']:.2e}) =====")
        all_results[seed] = res

    # ---------------- 表 ----------------
    seed0 = args.seed_list[0]
    r0 = all_results[seed0]
    lines = [f"**Grand unified capacity comparison** (H={H}, "
             f"T={args.num_samples}, {K} tasks sin(kx+phi) k={args.freqs}, "
             f"multiscale init, cleanup on, thr={thr}, "
             f"epochs-task={args.epochs_task}, seeds={args.seeds})", "",
             "| arm | capacity seq | capacity +pipeline | union after "
             "| max drift | approx epochs |",
             "|---" * 6 + "|"]
    for arm in arms:
        r = r0[arm]
        lines.append(f"| {arm} | {r['capacity_seq']}/{K} "
                     f"| **{r['capacity_pipe']}/{K}** "
                     f"| {r['union_after']} | {r['max_drift']:.2e} "
                     f"| {r['seq_epochs']} |")
    lines += ["", "per-task final MSE (seq):", "",
              "| task | " + " | ".join(arms) + " |",
              "|---" * (1 + len(arms)) + "|"]
    for i, (nm, _) in enumerate(tasks):
        cells = []
        for arm in arms:
            m = r0[arm]["mses"][i]
            b = "**" if r0[arm]["stored"][i] else ""
            cells.append(f"{b}{m:.4f}{b}")
        lines.append(f"| {nm} | " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    tag = "_".join(arms)
    write_text(args.out_dir / f"table_grand_{tag}.md", table)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    save_json(args.out_dir / f"results_{tag}.json",
              {"config": fncl.config_dict(args),
               "results": {str(sd): clean(all_results[sd])
                           for sd in all_results}})


if __name__ == "__main__":
    main()
