"""
consolidation_rehearsal.py — ソフト・コンソリデーション第5段: 重なり限定リハーサル
(docs/idea_consolidation.md §12.9 案5「重なり領域限定のリハーサル」)

案2/3 の部分的な重なり（タスク3 の場 = 自前 + sin の領域）は入力側凍結で
忘却ゼロを保つが、その代償として**共有ユニットは新タスクに適応できない**。
本実験は適応が必要になる状況 — タスク3 の目標ドリフト
sin(x+pi) -> sin(x+0.75pi) — を導入し、重なりの解凍と干渉管理を検証する。
ドリフト後の目標は -0.71 sin(x) + 0.71 cos(x) なので、sin 語彙の張る空間の
外に大きな成分を持ち、凍結基底では原理的に適応できない。

4 アーム（すべて同一の Phase A = 案3 終状態から出発、タスク3 を N step 適応）:

  frozen    : 重なりの入力側凍結のまま（案2/3 の規約）。適応は読み出し +
              自前領域のみ -> 適応失敗が予測される。忘却は厳密ゼロ。
  naive     : 重なりを解凍、リハーサルなし -> 適応するが sin を破壊するはず。
  rehearsal : 案5。解凍 + **場が交わるタスクだけ**を交互リハーサル。
              タスク3 の場 ∩ sin の場 ≠ ∅ -> sin を 1:1 でリハーサル
              （sin の EMA が許容超過なら追加リハーサル = 閉ループ）。
              タスク3 の場 ∩ cos の場 = ∅ -> cos のリハーサルは 0 step。
              隔離は h ゲートの物理が自動で与える: タスク3 の step では
              cos のユニットは沈黙 (z=0) なので勾配が厳密に 0。案5 の
              「重なり領域のみ交互提示・非重なり領域は隔離のまま」は、
              per-task 場の物理だけで実装され、マスクを要しない。
  full      : 案4 流の対照。重なりの有無に関わらず全過去タスク（sin + cos）を
              リハーサル -> 同等の保護をより高いコストで。

検証項目:
  V-a 適応: MSE3'(rehearsal) < 0.5 x MSE3'(frozen)
  V-b 保護: sin が許容内に留まる（rehearsal）/ naive では許容を超える
  V-c 局所性: cos はリハーサル 0 step かつパラメータ漂移が厳密に 0
  V-d コスト: rehearsal の総 step < full の総 step、保護は同等

生成物 (out/consolidation_rehearsal/):
  fig_rehearsal.png  適応後の予測 + sin 保護 + コスト比較
  table_rehearsal.md 比較表 + 判定
  results.json

実行例:
  python tmp/consolidation_rehearsal.py --quick
  python tmp/consolidation_rehearsal.py
"""
import argparse
import copy

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
# per-task 場つきのタスクコンテキストと重なり測度は consolidation_lib にある。
from consolidation_lib import (  # noqa: E402
    TaskCtx, freeze_masks, overlaps, region_drift, region_snapshot)


# ============================================================
# 1 アームの実行
# ============================================================
def run_arm(arm: str, net0, x, registry, y3_new, args, device):
    """arm in ('frozen', 'naive', 'rehearsal', 'full')."""
    net = copy.deepcopy(net0)
    H = args.hidden_dim
    r_sin, r_cos, r3 = registry

    ctx3 = TaskCtx.from_registry(net, x, r3, args, support=r3["support"],
                                 target=y3_new)
    ctx_sin = TaskCtx.from_registry(net, x, r_sin, args,
                                    support=r_sin["support"])
    ctx_cos = TaskCtx.from_registry(net, x, r_cos, args,
                                    support=r_cos["support"])

    if arm == "frozen":
        past = {l: sorted(set(r_sin["region"][l]) | set(r_cos["region"][l]))
                for l in (0, 1)}
        ctx3.trainer.grad_masks = freeze_masks(H, past, device)
    # 案5: リハーサル対象 = 場が交わる過去タスクのみ。full: 無条件に全過去タスク
    if arm == "rehearsal":
        rehearse = [c for c in (ctx_sin, ctx_cos)
                    if overlaps(ctx3.support, c.support) > 0]
    elif arm == "full":
        rehearse = [ctx_sin, ctx_cos]
    else:
        rehearse = []

    cos_snap = region_snapshot(net, r_cos["region"])
    steps = {"task3": 0, "sin": 0, "cos": 0}
    holds = 0
    tol_loop = {}
    for n in range(args.adapt_steps):
        ctx3.step(net)
        steps["task3"] += 1
        for c in rehearse:
            c.step(net)
            steps["sin" if c is ctx_sin else "cos"] += 1
        if rehearse and n + 1 == args.rehearsal_settle:
            # 閉ループ許容値を 1:1 交互提示の平衡ベースラインから再計算する
            # （§12.9.4 反復2 と同じ較正: 単独学習由来の許容値は単発 forward の
            # ノイズ床とほぼ同水準で、壊れていなくてもホールドが飽和する）。
            # 最終判定 (V-b) には元の許容値を使う。
            for c in rehearse:
                eq = float(np.mean(c.trainer.losses[-50:]))
                tol_loop[id(c)] = max(c.tol,
                                      eq * args.drift_mult + args.drift_abs)
        if n + 1 < args.rehearsal_settle:
            continue
        h = 0
        while (rehearse and h < args.max_holds
               and any(c.trainer.ema() > tol_loop[id(c)] for c in rehearse)):
            for c in rehearse:                     # 閉ループ: 追加リハーサル
                c.step(net)
                steps["sin" if c is ctx_sin else "cos"] += 1
            h += 1
        holds += h

    mse3, pred3 = ctx3.eval(net, x)
    mse_sin, pred_sin = ctx_sin.eval(net, x)
    mse_cos, _ = ctx_cos.eval(net, x)
    cos_drift = region_drift(net, r_cos["region"], cos_snap)
    total = steps["task3"] + steps["sin"] + steps["cos"]
    tl = tol_loop.get(id(ctx_sin))
    print(f"  [{arm}] MSE: task3'={mse3:.5f} sin={mse_sin:.5f} "
          f"(tol {ctx_sin.tol:.4f}"
          + (f", loop tol {tl:.4f}" if tl else "")
          + f") cos={mse_cos:.5f} cos_drift={cos_drift:.2e}; "
          f"steps={steps} (total {total}, holds={holds})")
    return {"mse3": mse3, "mse_sin": mse_sin, "mse_cos": mse_cos,
            "cos_drift": cos_drift, "steps": steps, "total_steps": total,
            "holds": holds, "tol_sin": ctx_sin.tol,
            "tol_loop_sin": tl,
            "pred3": pred3, "pred_sin": pred_sin}


# ============================================================
# main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="soft consolidation step 5: "
                                            "overlap-limited rehearsal")
    poc.add_poc_args(p)
    p.add_argument("--epochs-task", type=int, default=1000)
    p.add_argument("--stop-recovery", type=int, default=8)
    p.add_argument("--rho-init", type=float, default=1.0)
    p.add_argument("--recruit-thresh", type=float, default=0.5)
    p.add_argument("--adapt-steps", type=int, default=1500,
                   help="タスク3 のドリフト適応 step 数")
    p.add_argument("--drift-phase", type=float, default=0.75,
                   help="タスク3 の新目標 sin(x + drift_phase*pi)")
    p.add_argument("--rehearsal-settle", type=int, default=100,
                   help="ループ許容値の再較正までのならしペア数")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/consolidation_rehearsal")
    if args.quick:
        args.epochs_task = 120
        args.adapt_steps = 200

    device = torch.device(args.device)
    N = 128
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, N, dtype=np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    tasks = [("sin(x)", np.sin(x_raw).astype(np.float32)),
             ("cos(x)", np.cos(x_raw).astype(np.float32)),
             ("sin(x+pi)", np.sin(x_raw + np.pi).astype(np.float32))]
    y3_new = np.sin(x_raw + args.drift_phase * np.pi).astype(np.float32)
    H = args.hidden_dim
    arms = ("frozen", "naive", "rehearsal", "full")

    all_results = {}
    for seed in args.seed_list:
        print(f"\n===== seed {seed} =====")
        res_a = crc.run_sequence_recruit(seed, args, device, tasks, x)
        net0 = res_a["net"]
        registry = res_a["registry"]
        ov_sin = overlaps(registry[2]["support"], registry[0]["support"])
        ov_cos = overlaps(registry[2]["support"], registry[1]["support"])
        print(f"\n  task3 field overlap: with sin = {ov_sin} units, "
              f"with cos = {ov_cos} units")
        print(f"  drifted target: sin(x + {args.drift_phase}*pi)")

        res = {arm: run_arm(arm, net0, x, registry, y3_new, args, device)
               for arm in arms}

        rh, nv, fz, fl = (res["rehearsal"], res["naive"], res["frozen"],
                          res["full"])
        tol_sin = rh["tol_sin"]
        va = rh["mse3"] < 0.5 * fz["mse3"]
        vb = (rh["mse_sin"] <= 2.0 * tol_sin
              and nv["mse_sin"] > 2.0 * tol_sin)
        vc = (rh["cos_drift"] == 0.0 and rh["steps"]["cos"] == 0
              and ov_cos == 0)
        vd = (rh["total_steps"] < fl["total_steps"]
              and rh["mse_sin"] <= 1.5 * fl["mse_sin"] + args.drift_abs)
        verdicts = {"Va_adaptation": bool(va), "Vb_protection": bool(vb),
                    "Vc_locality": bool(vc), "Vd_cost": bool(vd)}
        all_results[seed] = {
            "overlap": {"sin": ov_sin, "cos": ov_cos},
            "arms": {a: {k: v for k, v in res[a].items()
                         if not k.startswith("pred")} for a in arms},
            "verdicts": verdicts}

        # ---------------- 図（先頭 seed） ----------------
        if seed == args.seed_list[0]:
            fig = plt.figure(figsize=(13, 8))
            gs = fig.add_gridspec(2, 3, height_ratios=[1.7, 1.3])
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(x_raw, y3_new, "k--", lw=1.4, label="drifted target")
            for arm, c in (("frozen", "tab:gray"), ("naive", "tab:red"),
                           ("rehearsal", "tab:blue")):
                ax.plot(x_raw, res[arm]["pred3"], lw=1.2, color=c,
                        label=f"{arm} (MSE {res[arm]['mse3']:.3f})")
            ax.set_title(f"task 3 adapts to sin(x+{args.drift_phase}pi)",
                         fontsize=10)
            ax.set_ylim(-1.6, 1.6)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            ax = fig.add_subplot(gs[0, 1])
            ax.plot(x_raw, tasks[0][1], "k--", lw=1.4, label="sin target")
            for arm, c in (("naive", "tab:red"), ("rehearsal", "tab:blue")):
                ax.plot(x_raw, res[arm]["pred_sin"], lw=1.2, color=c,
                        label=f"{arm} (MSE {res[arm]['mse_sin']:.3f})")
            ax.set_title("collateral: what happened to sin(x)", fontsize=10)
            ax.set_ylim(-1.6, 1.6)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            ax = fig.add_subplot(gs[0, 2])
            vals = [res[a]["mse_sin"] for a in arms]
            ax.bar(range(len(arms)), vals,
                   color=["tab:gray", "tab:red", "tab:blue", "tab:green"])
            ax.axhline(tol_sin, ls="--", c="k", lw=0.9,
                       label=f"sin tol {tol_sin:.3f}")
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels(arms, fontsize=8)
            ax.set_yscale("log")
            ax.set_ylabel("sin MSE after adaptation")
            ax.set_title("protection of the overlapped task", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, axis="y", which="both")
            ax = fig.add_subplot(gs[1, :2])
            bot = np.zeros(len(arms))
            for key, c in (("task3", "tab:blue"), ("sin", "tab:orange"),
                           ("cos", "tab:green")):
                v = [res[a]["steps"][key] for a in arms]
                ax.bar(range(len(arms)), v, 0.5, bottom=bot, color=c,
                       label=f"{key} steps")
                bot += np.asarray(v, dtype=float)
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels(arms, fontsize=9)
            ax.set_ylabel("training steps")
            ax.set_title("Rehearsal cost: only tasks whose field overlaps "
                         "are rehearsed (cos: zero)", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis="y")
            ax = fig.add_subplot(gs[1, 2])
            ax.axis("off")
            txt = (f"overlap(task3, sin) = {ov_sin} units\n"
                   f"overlap(task3, cos) = {ov_cos} units\n\n"
                   f"cos param drift:\n"
                   + "\n".join(f"  {a}: {res[a]['cos_drift']:.1e}"
                               for a in arms))
            ax.text(0.0, 0.9, txt, fontsize=10, va="top", family="monospace")
            fig.suptitle(f"Soft consolidation step 5: overlap-limited "
                         f"rehearsal (H={H}, seed {seed})", fontsize=12)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
            savefig(fig, args.out_dir / "fig_rehearsal.png")

    # ---------------- 表 ----------------
    seed0 = args.seed_list[0]
    r0 = all_results[seed0]
    lines = [f"**Soft consolidation step 5: overlap-limited rehearsal** "
             f"(H={H}, T={args.num_samples}, adapt-steps={args.adapt_steps}, "
             f"drift=sin(x+{args.drift_phase}pi), seeds={args.seeds})", "",
             f"overlap(task3, sin) = {r0['overlap']['sin']}, "
             f"overlap(task3, cos) = {r0['overlap']['cos']}", "",
             "| arm | task3' MSE | sin MSE | cos MSE | cos drift | steps "
             "(task3/sin/cos) | holds |",
             "|---" * 7 + "|"]
    for a in ("frozen", "naive", "rehearsal", "full"):
        m = r0["arms"][a]
        s = m["steps"]
        lines.append(f"| {a} | {m['mse3']:.5f} | {m['mse_sin']:.5f} "
                     f"| {m['mse_cos']:.5f} | {m['cos_drift']:.2e} "
                     f"| {s['task3']}/{s['sin']}/{s['cos']} | {m['holds']} |")
    lines += [""]
    for seed in args.seed_list:
        v = all_results[seed]["verdicts"]
        lines.append(f"- seed {seed}: "
                     + " / ".join(f"{k} {'**PASS**' if ok else '**FAIL**'}"
                                  for k, ok in v.items()))
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_rehearsal.md", table)
    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args),
               "results": {str(s): all_results[s] for s in all_results}})


if __name__ == "__main__":
    main()
