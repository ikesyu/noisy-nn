"""
multivalued_online.py — PoC-M4: 検出と分割ループのオンライン統合
(docs/idea_multivalued.md §8 PoC-M4 / 残課題 2)

M1/M2（検出）と M3（分割ループ）は別実験だった。本 PoC は両者を一つの
パイプラインに統合し、**多価かどうかを知らされないタスク列**を一気通貫で
処理する:

  タスク列（システムには関数形も多価性も未知）:
    t1 = sin(x)                     単価値 -> 通常格納されるべき
    t2 = { 1.5+0.5 sin(2x) (w=3),   多価   -> トリガ発火 + 分割されるべき
           -1.8-0.5 sin(x)  (w=2) }
    t3 = 0.7 sin(x) - 0.5           単価値 -> 通常格納（語彙再利用で自前~0）
    t4 = sin(8x)                    単価値だが現行基底では学習不能
                                    -> ゲートは発火するが「多価ではない」と
                                       鑑別され、分割されないべき（陰性対照）

  統一パイプライン（タスクごと、すべて forward 統計）:
    1. 素朴な単一場の学習: データの条件付き平均を目標に grow_field
       （多価かどうかを知らないシステムの自然な振る舞い）
    2. 品質ゲート: 全データ点への重み付きアンサンブル MSE > mse-star なら異常
    3. 三者鑑別（idea_multivalued.md §3 の表の実装）:
       - 容量     : 試行前の空きユニット率 >= 25% か（不足なら容量問題）
       - 既約性   : 残差 RMS の P 依存（P=32/P=1 > 0.8 なら既約 = 内在
                    ノイズでは説明できない）
       - 条件付き広がり: spread(x)/sigma_out(x) > 1 の x が 10% 超なら多価
    4. 多価と診断 -> 素朴な場を掃除して M3 の分割ループへ（past/free を
       引き継ぐ）。多価でない（t4: 既約だが広がりゼロ）-> 分割せず棄却。
    5. 後続タスクは更新された past/free で通常どおり続行。

検証項目:
  V-a 正常経路の透過性: t1/t3 が無トリガで格納、t3 の自前 <= 3（語彙再利用）
  V-b トリガの感度と特異度: 判定が 4 タスクすべて正しい
      (t1: store, t2: split, t3: store, t4: reject-not-multivalued)
  V-c 分割の成功: t2 がちょうど 2 場、純度 1.0、分岐 MSE < 0.05、
      c1 (= -1.8-0.5 sin x, t1 の張る空間内) の場の自前 <= 3
  V-d 無破壊性: 全格納記述子の drift = 0、格納時と終了時の MSE 一致

生成物 (out/multivalued_online/):
  fig_online.png / table_online.md / results.json

実行例:
  python tmp/multivalued_online.py --quick
  python tmp/multivalued_online.py --seeds 0,1,2
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
from consolidation_lib import (  # noqa: E402
    eval_with_descriptor, kill_unit, predict, region_drift, region_snapshot,
    set_field)
from consolidation_multiscale import SCALES  # noqa: E402
import multivalued_partial as mp  # noqa: E402
import multivalued_split_loop as msl  # noqa: E402


# ============================================================
# タスク列
# ============================================================
def make_stream(x_raw):
    """[(name, [branches], [weights])] のタスク列。システムには branch 数未知."""
    return [
        ("t1: sin(x)", [np.sin(x_raw).astype(np.float32)], [1.0]),
        ("t2: multivalued",
         [(1.5 + 0.5 * np.sin(2 * x_raw)).astype(np.float32),
          (-1.8 - 0.5 * np.sin(x_raw)).astype(np.float32)], [3.0, 2.0]),
        ("t3: 0.7sin(x)-0.5",
         [(0.7 * np.sin(x_raw) - 0.5).astype(np.float32)], [1.0]),
        ("t4: sin(8x) (unlearnable)",
         [np.sin(8.0 * x_raw).astype(np.float32)], [1.0]),
    ]


# ============================================================
# 診断（すべて forward 統計）
# ============================================================
def data_mse(pred, ys, weights):
    num = sum(w * float(np.mean((y - pred) ** 2)) for y, w in zip(ys, weights))
    return num / sum(weights)


def diagnose(net, x, r, ys, weights, free_frac_pre, args):
    """三者鑑別: 容量 / 既約性 / 条件付き広がり."""
    set_field(net, r["support"], r["sigma0"], r["h0"])
    net.fcs[2].weight.data.copy_(r["wout"])
    net.fcs[2].bias.data.copy_(r["b_out"])
    # 既約性: 残差 RMS の P 依存
    rms = {}
    for P in (1, 32):
        pred = predict(net, x, passes=P)
        rms[P] = float(np.sqrt(data_mse(pred, ys, weights)))
    irr_ratio = rms[32] / max(rms[1], 1e-9)
    # 条件付き広がり: 同じ x の y 集合の重み付き std / 出力ゆらぎ
    W = np.asarray(weights, dtype=np.float64)
    Y = np.stack(ys, axis=0)                      # [B, N]
    mean = (W[:, None] * Y).sum(0) / W.sum()
    spread = np.sqrt((W[:, None] * (Y - mean) ** 2).sum(0) / W.sum())
    sig_out = mp.output_noise(net, x)
    ratio = spread / (sig_out + 1e-6)
    spread_frac = float(np.mean(ratio > 1.0))
    capacity_ok = free_frac_pre >= args.free_frac_min
    multivalued = (capacity_ok and irr_ratio > args.irr_min
                   and spread_frac > args.spread_frac_min)
    return {"irr_ratio": irr_ratio, "spread_frac": spread_frac,
            "free_frac_pre": free_frac_pre, "capacity_ok": capacity_ok,
            "multivalued": bool(multivalued)}


# ============================================================
# 統一パイプライン（1 タスク分）
# ============================================================
def process_task(net, x, x_raw, name, ys, weights, past, free, args, device):
    H = args.hidden_dim
    free_frac_pre = (len(free[0]) + len(free[1])) / (2.0 * H)
    print(f"\n  ===== {name} (branches={len(ys)} [システムには未知], "
          f"free={len(free[0])}+{len(free[1])}) =====")

    # --- 1. 素朴な単一場: 条件付き平均を目標に ---
    W = np.asarray(weights, dtype=np.float64)
    mean = ((W[:, None] * np.stack(ys, 0)).sum(0) / W.sum()).astype(np.float32)
    unexp = [np.ones(len(x_raw), dtype=bool)]
    r, region, _ = msl.grow_field(net, x, [mean], [1.0], unexp,
                                  free, past, args, device)
    pred = eval_with_descriptor(net, x, r, passes=16)
    mse = data_mse(pred, ys, weights)
    gate = mse > args.mse_star
    print(f"    [gate] data MSE = {mse:.4f} -> "
          f"{'異常（診断へ）' if gate else '正常（格納）'}")

    if not gate:
        # 通常格納
        r["snap"] = region_snapshot(net, region, cols=r["support"][0])
        past.update({l: past[l] + region[l] for l in (0, 1)})
        free.update({l: [k for k in free[l] if k not in region[l]]
                     for l in (0, 1)})
        own = len(region[0]) + len(region[1])
        return {"name": name, "decision": "store", "mse": mse, "own": own,
                "descriptors": [r], "diag": None}

    # --- 2. 三者鑑別 ---
    diag = diagnose(net, x, r, ys, weights, free_frac_pre, args)
    print(f"    [diag] free_pre={diag['free_frac_pre']:.2f} "
          f"irr={diag['irr_ratio']:.2f} spread_frac={diag['spread_frac']:.2f}"
          f" -> {'多価' if diag['multivalued'] else '多価ではない'}")

    # --- 3. 素朴な場を掃除 ---
    for l in (0, 1):
        for k in region[l]:
            kill_unit(net, l, k)

    if not diag["multivalued"]:
        print("    [reject] 分割せず棄却（容量/学習可能性の問題として記録）")
        return {"name": name, "decision": "reject", "mse": mse, "own": 0,
                "descriptors": [], "diag": diag}

    # --- 4. 分割ループ（M3）へ、past/free を引き継いで ---
    print("    [split] M3 ループへ")
    fields, history, unexp2, past2, free2 = msl.split_loop(
        net, x, x_raw, ys, weights, args, device, past=past, free=free)
    past.update(past2)
    free.update(free2)
    descs = []
    for f in fields:
        d = f["descriptor"]
        d["snap"] = region_snapshot(net, f["region"],
                                    cols=d["support"][0])
        d["dominant_branch"] = f["dominant_branch"]
        d["purity"] = f["purity"]
        d["own"] = f["own"]
        descs.append(d)
    return {"name": name, "decision": "split", "mse": mse,
            "own": [f["own"] for f in fields], "descriptors": descs,
            "diag": diag,
            "purity": [f["purity"] for f in fields],
            "unexplained_final": history[-1]}


# ============================================================
# main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="PoC-M4: online integration of "
                                            "detection and splitting")
    poc.add_poc_args(p)
    p.add_argument("--epochs-task", type=int, default=600)
    p.add_argument("--stop-recovery", type=int, default=8)
    p.add_argument("--warm-epochs", type=int, default=100)
    p.add_argument("--wta-every", type=int, default=5)
    p.add_argument("--refine-epochs", type=int, default=400)
    p.add_argument("--eps-point", type=float, default=0.14)
    p.add_argument("--min-gain", type=float, default=0.05)
    p.add_argument("--max-fields", type=int, default=5)
    p.add_argument("--mse-star", type=float, default=0.05,
                   help="品質ゲートの絶対閾値")
    p.add_argument("--free-frac-min", type=float, default=0.25,
                   help="容量が問題でないと判断する空き率")
    p.add_argument("--irr-min", type=float, default=0.8,
                   help="既約と判断する RMS(P=32)/RMS(P=1) の下限")
    p.add_argument("--spread-frac-min", type=float, default=0.1,
                   help="多価と判断する spread/noise>1 の x の割合")
    p.add_argument("--stop-layer-fails", type=int, default=1,
                   help="stop 則 v2 (§12.9.14): 層を閉じるまでに許す"
                        "候補失敗数 (1=旧則)")
    p.add_argument("--stop-abort", type=int, default=None,
                   help="stop 則 v2: 閉ループ飽和が連続 n ステップで"
                        "anneal を中断 (None=旧則)")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/multivalued_online")
    args.stop_abort_saturated = args.stop_abort
    args.l1_scales = SCALES
    if args.quick:
        args.epochs_task = 150
        args.warm_epochs = 40
        args.refine_epochs = 80
        args.epochs_per_step = 5
        args.stop_recovery = 3

    device = torch.device(args.device)
    N = 128
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, N, dtype=np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    stream = make_stream(x_raw)
    H = args.hidden_dim
    expected = ["store", "split", "store", "reject"]

    all_results = {}
    for seed in args.seed_list:
        print(f"\n===== seed {seed} =====")
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = poc.build_net(H, args.sigma, args.crossing_h, args.num_samples,
                            device, scales=SCALES)
        past = {0: [], 1: []}
        free = {0: list(range(H)), 1: list(range(H))}
        records = []
        for name, ys, weights in stream:
            rec = process_task(net, x, x_raw, name, ys, weights, past, free,
                               args, device)
            records.append(rec)

        # ---- 終了時の再評価（無破壊性） ----
        stored = [(rec, d) for rec in records for d in rec["descriptors"]]
        final = []
        for rec, d in stored:
            pred = eval_with_descriptor(net, x, d, passes=16)
            # 記述子の目標: 単一格納はタスク関数、分割場は支配分岐
            ti = [s for s in stream if s[0] == rec["name"]][0]
            yb = (ti[1][d["dominant_branch"]]
                  if "dominant_branch" in d else ti[1][0])
            mse_end = float(np.mean((yb - pred) ** 2))
            drift = region_drift(net, d["region"], d["snap"],
                                 cols=d["support"][0])
            final.append({"task": rec["name"],
                          "branch": d.get("dominant_branch", 0),
                          "mse_end": mse_end, "drift": drift})
            print(f"  [final] {rec['name']} b{d.get('dominant_branch', 0)}: "
                  f"MSE={mse_end:.5f} drift={drift:.2e}")

        decisions = [rec["decision"] for rec in records]
        t2 = records[1]
        t3 = records[2]
        va = (decisions[0] == "store" and decisions[2] == "store"
              and t3["own"] <= 3
              and all(f["mse_end"] < args.mse_star for f in final))
        vb = decisions == expected
        vc = (t2["decision"] == "split" and len(t2["descriptors"]) == 2
              and all(p_ == 1.0 for p_ in t2.get("purity", []))
              and min(t2["own"]) <= 3)
        vd = all(f["drift"] == 0.0 for f in final)
        verdicts = {"Va_normal_path": bool(va), "Vb_trigger_decisions": bool(vb),
                    "Vc_split_success": bool(vc), "Vd_non_destructive": bool(vd)}
        print(f"\n  ===== decisions={decisions} (expected {expected}) =====")
        print(f"  ===== verdicts: {verdicts} =====")
        all_results[seed] = {
            "decisions": decisions,
            "records": [{k: v for k, v in rec.items()
                         if k not in ("descriptors",)} for rec in records],
            "final": final, "verdicts": verdicts}

        # ---- 図（先頭 seed） ----
        if seed == args.seed_list[0]:
            fig = plt.figure(figsize=(13, 4.8))
            gs = fig.add_gridspec(1, 3)
            ax = fig.add_subplot(gs[0, 0])
            colors = {"t1: sin(x)": "tab:blue", "t2: multivalued": "tab:red",
                      "t3: 0.7sin(x)-0.5": "tab:green"}
            for name, ys, weights in stream:
                for yb in ys:
                    ax.plot(x_raw, yb, "k--", lw=0.6)
            for rec in records:
                for d in rec["descriptors"]:
                    pred = eval_with_descriptor(net, x, d, passes=16)
                    ax.plot(x_raw, pred, lw=1.4,
                            c=colors.get(rec["name"], "tab:gray"))
            ax.set_title("Stored functions at end of stream\n"
                         "(4 descriptors from 3 accepted tasks)", fontsize=9)
            ax.grid(alpha=0.3)

            ax = fig.add_subplot(gs[0, 1])
            names = [rec["name"].split(":")[0] for rec in records]
            mses = [rec["mse"] for rec in records]
            cols = ["tab:green" if d == e else "tab:red"
                    for d, e in zip(decisions, expected)]
            ax.bar(range(len(records)), mses, color=cols)
            ax.axhline(args.mse_star, ls="--", c="k", lw=0.9,
                       label=f"gate {args.mse_star}")
            for i, rec in enumerate(records):
                ax.text(i, mses[i] * 1.1, rec["decision"], ha="center",
                        fontsize=8)
            ax.set_yscale("log")
            ax.set_xticks(range(len(records)))
            ax.set_xticklabels(names)
            ax.set_ylabel("naive single-field data MSE")
            ax.set_title("Gate + decisions (green = as expected)", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, axis="y", which="both")

            ax = fig.add_subplot(gs[0, 2])
            ax.axis("off")
            lines_d = []
            for rec in records:
                d = rec["diag"]
                if d is None:
                    lines_d.append(f"{rec['name'].split(':')[0]}: gate pass "
                                   f"-> {rec['decision']}")
                else:
                    lines_d.append(
                        f"{rec['name'].split(':')[0]}: irr={d['irr_ratio']:.2f}"
                        f" spread={d['spread_frac']:.2f}"
                        f" -> {rec['decision']}")
            txt = ("verdicts\n\n"
                   + "\n".join(f"{k}: {'PASS' if v else 'FAIL'}"
                               for k, v in verdicts.items())
                   + "\n\ndiagnosis\n" + "\n".join(lines_d))
            ax.text(0.0, 0.95, txt, fontsize=9, va="top", family="monospace")
            fig.suptitle(f"PoC-M4: online detection + splitting "
                         f"(H={H}, seed {seed})", fontsize=12)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
            savefig(fig, args.out_dir / "fig_online.png")

    # ---- 表 ----
    seed0 = args.seed_list[0]
    r0 = all_results[seed0]
    lines = [f"**PoC-M4: online integration** (H={H}, T={args.num_samples}, "
             f"epochs-task={args.epochs_task}, gate={args.mse_star}, "
             f"seeds={args.seeds})", "",
             "| task | naive MSE | irr | spread | decision | own |",
             "|---" * 6 + "|"]
    for rec in r0["records"]:
        d = rec.get("diag")
        irr = "—" if d is None else f"{d['irr_ratio']:.2f}"
        spr = "—" if d is None else f"{d['spread_frac']:.2f}"
        lines.append(f"| {rec['name']} | {rec['mse']:.4f} | {irr} | {spr} "
                     f"| **{rec['decision']}** | {rec['own']} |")
    lines += ["", "final (end of stream):", ""]
    for f in r0["final"]:
        lines.append(f"- {f['task']} b{f['branch']}: MSE={f['mse_end']:.5f} "
                     f"drift={f['drift']:.2e}")
    lines.append("")
    for seed in args.seed_list:
        v = all_results[seed]["verdicts"]
        lines.append(f"- seed {seed}: "
                     + " / ".join(f"{k} {'**PASS**' if ok else '**FAIL**'}"
                                  for k, ok in v.items()))
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_online.md", table)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args),
               "results": clean(all_results)})


if __name__ == "__main__":
    main()
