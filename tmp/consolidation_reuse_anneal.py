"""
consolidation_reuse_anneal.py — ソフト・コンソリデーション第2段: reuse-then-anneal
(docs/idea_consolidation.md §12.9 案1「部分動員ウォームスタート」)

案2 (readout-only overlap; consolidation_soft.py, §12.9.1) は語彙の共有を実現
したが、タスク i の推論時の動員場 (support) を「自前領域 + 過去領域全部」と
しており、読み出しがほとんど使わない語彙ユニットも動員されたままである。
本実験は案1 の中核 — **重なりの大きさをアニールの選択に決めさせる** — を足す:

  - 初期フィールド = 空き領域 (ρ=1) + 既存領域（ρ=rho-init の部分動員・
    入力側は勾配マスクで凍結。読み出し列のみ共有 = 案2 と同じ隔離）
  - anneal-until-stop の候補に **語彙 L2 ユニット** を加える。語彙候補の
    スナップは「このタスクの support からの離脱」であり、workspace の場
    （σ=0, h=H_DEAD）と現タスクの読み出し列のゼロ化にだけ作用する
    （= poc.kill_unit と同一操作だが、供給元タスクの記述子・入力側
    パラメータには触れないので、供給元の機能は厳密に不変のまま）。
  - 依存関係の簿記: 供給元タスク s の語彙 L2 が全て離脱したら、s の L1
    ブロックも support から外す（凍結行しか feed しないので直接外せる）。

比較は同一 seed の案2 アーム（csoft.run_sequence("soft")）と行い、
support サイズの縮小（動員コストの節約）と重なりの類似度対応を検証する。

検証項目:
  V-a 各タスクが許容内（prune アーム）
  V-b 忘却厳密ゼロ（両アーム、入力側 max|Δ| = 0）
  V-c support 縮小: タスク 2,3 で |support_prune| < |support_readout|、MSE 維持
  V-d 重なりの自動決定が類似度に対応: 残存語彙数 task3 (-sin) > task2 (cos)、
      かつ task3 の残存語彙は sin 由来

生成物 (out/consolidation_reuse_anneal/):
  fig_reuse.png    予測 + support 構成 (readout vs prune) + 残存語彙の由来
  table_reuse.md   比較表 + 判定
  results.json

実行例:
  python tmp/consolidation_reuse_anneal.py --quick
  python tmp/consolidation_reuse_anneal.py
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
from consolidation_lib import (  # noqa: E402,F401
    anneal_unit, eval_with_descriptor, freeze_masks, region_drift,
    region_snapshot, set_field_rho, trainer_rollback, trainer_state,
    unit_score, vocab_energy, zero_cross_columns)


# ============================================================
# anneal-until-stop（語彙 L2 も候補に含む拡張版）
# ============================================================
def anneal_until_stop_prune(net, x, target, tol, args, eligible: dict,
                            grad_masks, vocab: dict, share_l1: bool = False):
    """自前 (L1, L2) + 語彙 L2 の 3 グループを貪欲 min S_k でアニールする.

    グループ = (層, 種別): (0,'own'), (1,'own'), (1,'voc')。停止則は
    csoft.anneal_until_stop と同一（スナップ + 回復猶予後も EMA > tol なら
    巻き戻してそのグループを閉じる）。語彙のスナップも poc.kill_unit だが、
    作用は workspace の場と現タスクの読み出し列に限られる（入力側は凍結）。
    Returns (losses, holds_total, voc_alive)
    """
    layer_fails = getattr(args, "stop_layer_fails", 1)
    abort_sat = getattr(args, "stop_abort_saturated", None)
    t_target = torch.tensor(target, device=x.device).unsqueeze(1)
    trainer = poc.CovJacTrainer(net, x, t_target, lr=args.ft_lr, opt=args.opt,
                                jac_ema=args.jac_ema)
    trainer.grad_masks = grad_masks
    pools = {(0, "own"): list(eligible[0]), (1, "own"): list(eligible[1]),
             (1, "voc"): list(vocab[1])}
    open_groups = {g for g in pools if pools[g]}
    holds_total = 0
    fails = {g: 0 for g in pools}
    banned = {g: set() for g in pools}
    while open_groups:
        _, zbar = poc.collect_stats(net, x, passes=args.stat_passes)
        scored = {}
        for g in sorted(open_groups):
            l, kind = g
            Wn = net.fcs[l + 1].weight.data
            # 補償が実際に流せる基底: L1 は自前 L1 のみ（語彙 L1 への交差列は
            # 凍結 0）。L2 は自前 L2 + 生存語彙 L2（読み出しが両方へ張れる）。
            if l == 0:
                basis_all = pools[(0, "own")] + (list(vocab[0]) if share_l1
                                                 else [])
            else:
                basis_all = pools[(1, "own")] + pools[(1, "voc")]
            best_g = None
            for k in pools[g]:
                if k in banned[g]:
                    continue
                S = unit_score(zbar[l], k, [j for j in basis_all if j != k],
                               float((Wn[:, k] ** 2).sum()), args.ridge)
                if S is not None and (best_g is None or S < best_g[0]):
                    best_g = (S, k)
            if best_g is None:
                open_groups.discard(g)
            else:
                scored[g] = best_g
        if not scored:
            break
        g = min(scored, key=lambda gg: scored[gg][0])
        l, kind = g
        k = scored[g][1]
        ck_net, ck_tr = poc.checkpoint(net), trainer_state(trainer)
        res = anneal_unit(
            net, l, k, lambda _kind: trainer.run(args.epochs_per_step),
            over_tol=lambda: trainer.ema() > tol,
            read_act=lambda: float(trainer.cap.z[l][:, :, k].mean()),
            alpha=args.anneal_alpha, max_holds=args.max_holds,
            snap_act=args.snap_act, max_steps=args.max_anneal_steps,
            abort_saturated=abort_sat)
        holds_inc, completed = (res, True) if abort_sat is None else res
        holds_total += holds_inc
        failed = False
        if completed:
            poc.kill_unit(net, l, k)
            rec = 0
            while trainer.ema() > tol and rec < args.stop_recovery:
                trainer.run(args.epochs_per_step)
                rec += 1
            failed = trainer.ema() > tol
        else:
            failed = True
        if failed:
            poc.restore(net, ck_net)
            trainer_rollback(trainer, ck_tr)
            banned[g].add(k)
            fails[g] += 1
            why = "吸収不能" if completed else "閉ループ飽和で中断"
            if fails[g] >= layer_fails:
                open_groups.discard(g)
                print(f"    [stop] {g} unit {k}: {why} -> 巻き戻し "
                      f"(残 = {len(pools[g])} で確定)")
            else:
                print(f"    [fail {fails[g]}/{layer_fails}] {g} unit {k}: "
                      f"{why} -> 巻き戻し・候補から除外")
        else:
            pools[g].remove(k)
            if not pools[g]:
                open_groups.discard(g)
    trainer.close()
    return trainer.losses, holds_total, pools[(1, "voc")]


# ============================================================
# 逐次実行（prune アーム）
# ============================================================
def run_sequence_prune(seed: int, args, device, tasks, x):
    torch.manual_seed(seed)
    np.random.seed(seed)
    H = args.hidden_dim
    net = poc.build_net(H, args.sigma, args.crossing_h, args.num_samples, device)
    past = {0: [], 1: []}
    free = {0: list(range(H)), 1: list(range(H))}
    registry = []

    for ti, (name, target) in enumerate(tasks):
        vocab = {l: sorted(past[l]) for l in (0, 1)}
        print(f"\n  ===== [prune] task {ti + 1}: {name}  "
              f"(free L1 {len(free[0])} / L2 {len(free[1])}, "
              f"vocab L2 {len(vocab[1])}, rho_init={args.rho_init}) =====")
        net.fcs[2].weight.data.zero_()
        net.fcs[2].bias.data.zero_()
        zero_cross_columns(net, past)
        set_field_rho(net, free, vocab, args.sigma, args.crossing_h,
                      args.rho_init)
        masks = (freeze_masks(H, past, device)
                 if (past[0] or past[1]) else None)

        t = torch.tensor(target, device=device).unsqueeze(1)
        trainer = poc.CovJacTrainer(net, x, t, lr=args.pre_lr, opt=args.opt,
                                    jac_ema=args.jac_ema)
        trainer.grad_masks = masks
        log_every = max(1, args.epochs_task // 4)
        for e in range(args.epochs_task):
            loss = trainer.step()
            if e % log_every == 0 or e == args.epochs_task - 1:
                print(f"    [learn] epoch {e:5d} mse={loss:.5f}")
        trainer.close()
        base = float(np.mean(trainer.losses[-min(100, len(trainer.losses)):]))
        tol = base * args.drift_mult + args.drift_abs

        losses, holds, voc_alive = anneal_until_stop_prune(
            net, x, target, tol, args, eligible=free, grad_masks=masks,
            vocab=vocab)
        region = {l: [k for k in free[l] if float(net.sigma_vecs[l][k]) > 0]
                  for l in (0, 1)}
        # 依存関係の簿記: 語彙 L2 が生き残った供給元の L1 ブロックだけ残す
        src_alive = [s for s in registry
                     if any(k in voc_alive for k in s["region"][1])]
        voc_l1 = sorted({k for s in src_alive for k in s["region"][0]})
        support = {0: sorted(region[0] + voc_l1),
                   1: sorted(region[1] + sorted(voc_alive))}
        r = {"name": name, "target": target, "region": region,
             "support": support, "sigma0": float(args.sigma),
             "h0": float(args.crossing_h),
             "wout": net.fcs[2].weight.data.clone(),
             "b_out": net.fcs[2].bias.data.clone(),
             "snap": region_snapshot(net, region), "tol": tol,
             "holds": holds, "anneal_epochs": len(losses)}
        registry.append(r)
        pred = eval_with_descriptor(net, x, r)
        r["mse_at_consolidation"] = float(np.mean((pred - target) ** 2))
        kept_by_src = {s["name"]: sorted(k for k in voc_alive
                                         if k in s["region"][1])
                       for s in registry[:-1]}
        print(f"    own L1 {len(region[0])} + L2 {len(region[1])}; "
              f"kept vocab L2 = {kept_by_src if kept_by_src else '{}'} "
              f"(offered {len(vocab[1])}); support "
              f"{len(support[0])}+{len(support[1])}; "
              f"MSE={r['mse_at_consolidation']:.5f} (tol={tol:.4f})")
        r["kept_vocab_by_source"] = kept_by_src
        past = {l: past[l] + region[l] for l in (0, 1)}
        free = {l: [k for k in free[l] if k not in region[l]] for l in (0, 1)}

    # ---------------- 最終評価 ----------------
    K = len(tasks)
    out = {"tasks": [], "cross": np.zeros((K, K)), "preds": [],
           "registry": registry}
    for i, r in enumerate(registry):
        pred = eval_with_descriptor(net, x, r)
        out["preds"].append(pred)
        mse = float(np.mean((pred - r["target"]) ** 2))
        drift = region_drift(net, r["region"], r["snap"])
        for j in range(K):
            out["cross"][i, j] = float(np.mean((pred - registry[j]["target"]) ** 2))
        share, by_source = vocab_energy(net, x, r, registry,
                                        passes=args.stat_passes)
        n_voc2 = len(r["support"][1]) - len(r["region"][1])
        out["tasks"].append({
            "name": r["name"],
            "n_own": {"0": len(r["region"][0]), "1": len(r["region"][1])},
            "region": {str(l): r["region"][l] for l in (0, 1)},
            "support_size": [len(r["support"][0]), len(r["support"][1])],
            "kept_vocab_l2": n_voc2,
            "kept_vocab_by_source": r["kept_vocab_by_source"],
            "mse_at_consolidation": r["mse_at_consolidation"],
            "mse_final": mse, "drift": drift,
            "vocab_share": share, "vocab_share_by_source": by_source,
            "holds": r["holds"], "tol": r["tol"],
            "anneal_epochs": r["anneal_epochs"]})
        print(f"  [prune] {r['name']}: final MSE={mse:.5f} drift={drift:.2e} "
              f"support={len(r['support'][0])}+{len(r['support'][1])} "
              f"vocab_share={share:.3f}")
    out["total_own"] = sum(t["n_own"]["0"] + t["n_own"]["1"]
                           for t in out["tasks"])
    return out


# ============================================================
# 図・表
# ============================================================
def make_figure(tasks, x_raw, res_ro, res_pr, H, seed, path):
    K = len(tasks)
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, K, height_ratios=[2.0, 1.6])
    for i in range(K):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(x_raw, tasks[i][1], "k--", lw=1.2, label="target")
        ax.plot(x_raw, res_pr["preds"][i], lw=1.4,
                label="prune (own field)")
        t = res_pr["tasks"][i]
        ax.set_title(f"task {i + 1}: {t['name']}\nMSE={t['mse_final']:.4f}  "
                     f"support={t['support_size'][0]}+{t['support_size'][1]}",
                     fontsize=10)
        ax.set_ylim(-1.6, 1.6)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    # support 構成の比較（readout = 案2, prune = 案1）
    ax = fig.add_subplot(gs[1, :])
    idx = np.arange(K)
    wd = 0.35
    for off, arm, res in ((-wd / 2, "readout (case 2)", res_ro),
                          (wd / 2, "prune (case 1)", res_pr)):
        own = []
        voc = []
        for i, t in enumerate(res["tasks"]):
            n_own = t["n_own"]["0"] + t["n_own"]["1"]
            if "support_size" in t:
                n_sup = t["support_size"][0] + t["support_size"][1]
            else:                      # 案2: support = own + 過去領域全部
                reg = res["registry"][i]
                n_sup = len(reg["support"][0]) + len(reg["support"][1])
            own.append(n_own)
            voc.append(n_sup - n_own)
        color = "tab:gray" if "readout" in arm else "tab:blue"
        ax.bar(idx + off, own, wd, color=color, label=f"{arm}: own")
        ax.bar(idx + off, voc, wd, bottom=own, color=color, alpha=0.45,
               label=f"{arm}: mobilised vocab")
    ax.set_xticks(idx)
    ax.set_xticklabels([t[0] for t in tasks])
    ax.set_ylabel("mobilised units at inference (support)")
    ax.set_title("Support size: annealing prunes the overlap down to what "
                 "is actually needed")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Soft consolidation step 2: reuse-then-anneal "
                 f"(H={H}, rho_init=1.0, seed {seed})", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    savefig(fig, path)


# ============================================================
# main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="soft consolidation step 2: "
                                            "reuse-then-anneal")
    poc.add_poc_args(p)
    p.add_argument("--epochs-task", type=int, default=1000)
    p.add_argument("--stop-recovery", type=int, default=8)
    p.add_argument("--rho-init", type=float, default=1.0,
                   help="語彙の初期動員率 ρ (σ=ρσ0, h=h0/ρ)")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/consolidation_reuse_anneal")
    if args.quick:
        args.epochs_task = 120
        args.epochs_per_step = 5
        args.stop_recovery = 3

    device = torch.device(args.device)
    N = 128
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, N, dtype=np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    tasks = [("sin(x)", np.sin(x_raw).astype(np.float32)),
             ("cos(x)", np.cos(x_raw).astype(np.float32)),
             ("sin(x+pi)", np.sin(x_raw + np.pi).astype(np.float32))]
    H = args.hidden_dim

    all_results = {}
    for seed in args.seed_list:
        print(f"\n===== seed {seed} =====")
        res_ro = csoft.run_sequence("soft", seed, args, device, tasks, x, x_raw)
        res_pr = run_sequence_prune(seed, args, device, tasks, x)

        # --- 判定 ---
        pr, ro = res_pr["tasks"], res_ro["tasks"]
        sup_ro = [len(r["support"][0]) + len(r["support"][1])
                  for r in res_ro["registry"]]
        sup_pr = [t["support_size"][0] + t["support_size"][1] for t in pr]
        va = all(t["mse_final"] <= 2.0 * t["tol"] for t in pr)
        vb = all(t["drift"] == 0.0 for t in pr + ro)
        vc = all(sup_pr[i] < sup_ro[i] for i in (1, 2))
        kept = [t["kept_vocab_l2"] for t in pr]
        kept3_sin = len(pr[2]["kept_vocab_by_source"].get("sin(x)", []))
        vd = kept[2] > kept[1] and kept3_sin >= 1
        verdicts = {"Va_within_tol": bool(va), "Vb_zero_forgetting": bool(vb),
                    "Vc_support_shrinks": bool(vc),
                    "Vd_similarity_overlap": bool(vd)}
        all_results[seed] = {
            "readout": {"tasks": ro, "support": sup_ro,
                        "total_own": res_ro["total_own"]},
            "prune": {"tasks": pr, "support": sup_pr,
                      "total_own": res_pr["total_own"],
                      "cross": res_pr["cross"].tolist()},
            "verdicts": verdicts}
        if seed == args.seed_list[0]:
            make_figure(tasks, x_raw, res_ro, res_pr, H, seed,
                        args.out_dir / "fig_reuse.png")

    # --- 表 ---
    seed0 = args.seed_list[0]
    r0 = all_results[seed0]
    lines = [f"**Soft consolidation step 2: reuse-then-anneal** "
             f"(H={H}, T={args.num_samples}, rho_init={args.rho_init}, "
             f"epochs-task={args.epochs_task}, seeds={args.seeds})", "",
             "| arm | task | own L1+L2 | support | kept vocab L2 | MSE final "
             "| vocab share | drift |",
             "|---" * 8 + "|"]
    for arm in ("readout", "prune"):
        for i, t in enumerate(r0[arm]["tasks"]):
            sup = r0[arm]["support"][i]
            kept = (t.get("kept_vocab_l2", "all") if arm == "prune"
                    else "all offered")
            lines.append(
                f"| {arm} | {t['name']} | {t['n_own']['0']}+{t['n_own']['1']} "
                f"| {sup} | {kept} | {t['mse_final']:.5f} "
                f"| {t['vocab_share']:.3f} | {t['drift']:.2e} |")
    lines += ["", "kept vocab by source (prune):", ""]
    for t in r0["prune"]["tasks"]:
        if t["kept_vocab_by_source"]:
            srcs = ", ".join(f"{k}: {v}"
                             for k, v in t["kept_vocab_by_source"].items())
            lines.append(f"- {t['name']}: {srcs}")
    lines += [""]
    for seed in args.seed_list:
        v = all_results[seed]["verdicts"]
        lines.append(f"- seed {seed}: "
                     + " / ".join(f"{k} {'**PASS**' if ok else '**FAIL**'}"
                                  for k, ok in v.items()))
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_reuse.md", table)
    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args),
               "results": {str(s): all_results[s] for s in all_results}})


if __name__ == "__main__":
    main()
