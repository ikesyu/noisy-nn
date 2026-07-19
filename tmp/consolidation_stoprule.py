"""
consolidation_stoprule.py — 課題8: stop 則の保守性 A/B 比較 (§12.9.14)

M4 (idea_multivalued.md §10.4) で観測された「タスク 1 が 31 ユニットを
抱え込む」問題の上流は anneal-until-stop の stop 則にある: 最初の 1 ユニット
が吸収不能になった時点で層全体を閉じるため、周辺化が進む前に大領域が確定
する。

stop 則 v2 (consolidation_lib.anneal_unit / csoft.anneal_until_stop /
cra.anneal_until_stop_prune に実装済み; 既定値では旧動作と一致):
  - stop_layer_fails n: 層を閉じるまでに許す「候補ユニットの失敗」数。
    失敗ユニットは巻き戻して候補から除外 (ban) し、残りの候補で継続する。
    旧則は n=1 (最初の失敗で層を閉じる)。
  - stop_abort_saturated m: 閉ループが m ステップ連続で飽和 (hold を
    max_holds 回使い切っても tol 超過) したら、そのユニットの anneal を
    max_steps まで待たず中断して失敗扱いにする (§12.5 の先行指標)。
    ban との組で「失敗を早く検出し、別の候補を試す」動作になる。
    旧則は None (飽和しても max_steps まで粘る)。

exp1 (--exp floor): 単一タスク床テスト。sin(x), H=32 マルチスケール, 3 seed。
  旧則 vs 新則で自前領域サイズ・コスト (holds / anneal epochs)・MSE を比較。
  参照: §12.5 の実効ランク床 (sin 相当で ~4 ユニット)。
exp2 (--exp capacity): 容量研究の softA アーム (§12.9.13 grand と同一条件,
  sin(kx+phi) k=1..4 x 位相 4 = 16 タスク) を旧則 vs 新則で再実行し、
  タスク 1 の占有・capacity_seq / capacity_pipe の変化を見る。新則は
  獲得と retry の両方に適用する (union anneal の layer_fails は従来どおり
  args.layer_fails で独立)。

実行例:
  python tmp/consolidation_stoprule.py --quick
  python tmp/consolidation_stoprule.py --exp floor --seeds 0,1,2
  python tmp/consolidation_stoprule.py --exp capacity --seeds 0
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


def set_rule(args, rule):
    """rule in ('old', 'new'): stop 則のスイッチを args に載せる。"""
    if rule == "old":
        args.stop_layer_fails = 1
        args.stop_abort_saturated = None
    else:
        args.stop_layer_fails = args.new_layer_fails
        args.stop_abort_saturated = args.new_abort


# ---------------------------------------------------------------- exp1: floor
def exp_floor(args, device, x, x_raw):
    tasks = ccap.make_tasks(x_raw, (1,), 1)  # [sin(x)]
    rows = []
    for rule in ("old", "new"):
        set_rule(args, rule)
        for seed in args.seed_list:
            print(f"\n########## floor / rule {rule} / seed {seed} "
                  f"##########")
            out = csoft.run_sequence("hard", seed, args, device, tasks, x,
                                     x_raw)
            t = out["tasks"][0]
            rows.append({
                "rule": rule, "seed": seed,
                "own1": t["n_own"]["0"], "own2": t["n_own"]["1"],
                "mse": t["mse_final"], "tol": t["tol"],
                "holds": t["holds"], "anneal_epochs": t["anneal_epochs"]})
    lines = [f"**stop 則 A/B — exp1 単一タスク床テスト** (sin(x), H="
             f"{args.hidden_dim} multiscale, T={args.num_samples}, "
             f"epochs-task={args.epochs_task}, 新則 layer_fails="
             f"{args.new_layer_fails} / abort_saturated={args.new_abort}; "
             f"実効ランク床の参照値 ~4)", "",
             "| rule | seed | own L1+L2 | MSE | tol | holds "
             "| anneal epochs |",
             "|---" * 7 + "|"]
    for r in rows:
        lines.append(f"| {r['rule']} | {r['seed']} "
                     f"| {r['own1']}+{r['own2']} | {r['mse']:.4f} "
                     f"| {r['tol']:.4f} | {r['holds']} "
                     f"| {r['anneal_epochs']} |")
    for rule in ("old", "new"):
        sel = [r for r in rows if r["rule"] == rule]
        own = [r["own1"] + r["own2"] for r in sel]
        ep = [r["anneal_epochs"] for r in sel]
        lines.append(f"- {rule}: own 平均 {np.mean(own):.1f} "
                     f"(min {min(own)}, max {max(own)}), "
                     f"anneal epochs 平均 {np.mean(ep):.0f}")
    return rows, "\n".join(lines) + "\n"


# ------------------------------------------------------------- exp2: capacity
def run_softA(args, device, tasks, x, x_raw, thr):
    """grand §12.9.13 の softA アーム + 統一パイプラインをそのまま再現。"""
    K = len(tasks)
    out = crc.run_sequence_recruit(args.seed_list[0], args, device, tasks,
                                   x, cleanup_thr=thr, share_l1=True)
    mses = [t["mse_final"] for t in out["tasks"]]
    own = [t["n_own"]["0"] + t["n_own"]["1"] for t in out["tasks"]]
    stored = ccap.stored_flags(mses, thr)
    net, registry = out["net"], out["registry"]
    stored_idx = [i for i, r in enumerate(registry)
                  if r["mse_at_consolidation"] <= thr]
    extras = [tasks[i] for i in range(K) if i not in stored_idx]
    ctxs, U, n_before, n_after = ccap.compress(
        net, x, registry, stored_idx, args, device, accept_thr=thr)
    ext = (ccap.extend(net, x, U, extras, args, device, share_l1=True)
           if extras else [])
    mses_c2 = (ccap.eval_shared(net, x, ctxs, U, args.sigma,
                                args.crossing_h) if ctxs else [])
    stored_pipe = (ccap.stored_flags(mses_c2, thr)
                   + ccap.stored_flags([e["mse"] for e in ext], thr))
    return {"mses": mses, "own": own, "stored": stored,
            "capacity_seq": int(sum(stored)),
            "capacity_pipe": int(sum(stored_pipe)),
            "union_before": n_before, "union_after": n_after,
            "retry": [{"name": e["name"], "mse": e["mse"]} for e in ext],
            "max_drift": max([t.get("drift", 0.0) for t in out["tasks"]],
                             default=0.0)}


def exp_capacity(args, device, x, x_raw):
    freqs = tuple(int(v) for v in args.freqs.split(","))
    tasks = ccap.make_tasks(x_raw, freqs, args.n_phases)
    K = len(tasks)
    thr = args.mse_star
    res = {}
    for rule in ("old", "new"):
        set_rule(args, rule)
        print(f"\n########## capacity softA / rule {rule} ({K} tasks) "
              f"##########")
        res[rule] = run_softA(args, device, tasks, x, x_raw, thr)
        r = res[rule]
        print(f"\n  ===== [{rule}] capacity: seq {r['capacity_seq']}/{K} "
              f"-> pipeline {r['capacity_pipe']}/{K} "
              f"(union {r['union_before']}->{r['union_after']}) =====")
    lines = [f"**stop 則 A/B — exp2 容量 softA アーム** (H="
             f"{args.hidden_dim}, {K} tasks sin(kx+phi) k={args.freqs}, "
             f"multiscale, cleanup on, thr={thr}, seed "
             f"{args.seed_list[0]}, 新則 layer_fails="
             f"{args.new_layer_fails} / abort_saturated={args.new_abort})",
             "",
             "| rule | capacity seq | capacity +pipeline | task-1 own "
             "| own 上位3 | union after | max drift |",
             "|---" * 7 + "|"]
    for rule in ("old", "new"):
        r = res[rule]
        top3 = sorted(r["own"], reverse=True)[:3]
        lines.append(f"| {rule} | {r['capacity_seq']}/{K} "
                     f"| **{r['capacity_pipe']}/{K}** | {r['own'][0]} "
                     f"| {top3} | {r['union_after']} "
                     f"| {r['max_drift']:.2e} |")
    lines += ["", "per-task (own units / MSE seq):", "",
              "| task | old own | old MSE | new own | new MSE |",
              "|---" * 5 + "|"]
    for i, (nm, _) in enumerate(tasks):
        cells = []
        for rule in ("old", "new"):
            r = res[rule]
            b = "**" if r["stored"][i] else ""
            cells.append(f"{r['own'][i]}")
            cells.append(f"{b}{r['mses'][i]:.4f}{b}")
        lines.append(f"| {nm} | " + " | ".join(cells) + " |")
    return res, "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="stop-rule conservatism A/B")
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
    p.add_argument("--exp", type=str, default="floor,capacity")
    p.add_argument("--new-layer-fails", type=int, default=4,
                   help="新則: 層を閉じるまでに許す失敗数")
    p.add_argument("--new-abort", type=int, default=3,
                   help="新則: 閉ループ飽和が連続 n ステップで中断")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/consolidation_stoprule")
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

    exps = [e.strip() for e in args.exp.split(",")]
    results, tables = {}, []
    if "floor" in exps:
        rows, table = exp_floor(args, device, x, x_raw)
        results["floor"] = rows
        tables.append(table)
        print("\n" + table)
    if "capacity" in exps:
        res, table = exp_capacity(args, device, x, x_raw)
        results["capacity"] = res
        tables.append(table)
        print("\n" + table)

    tag = "_".join(exps)
    write_text(args.out_dir / f"table_stoprule_{tag}.md",
               "\n\n".join(tables))

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
              {"config": fncl.config_dict(args), "results": clean(results)})


if __name__ == "__main__":
    main()
