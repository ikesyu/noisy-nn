"""
consolidation_recruit.py — ソフト・コンソリデーション第3段: similarity-gated
recruitment (docs/idea_consolidation.md §12.9 案3「類似度駆動の動員」)

案1 (reuse-then-anneal; §12.9.2) は語彙を全て動員して学習し、アニールが事後に
重なりを剪定する。本実験は動員の決定を**学習前**へ移す: 新タスクの目標に
対する語彙ユニットのチューニング相関

    score_k = | corr_x( zbar_k(x), y_new(x) ) |

を、語彙を一時動員した少数 forward パス（学習なし・純 forward 統計）で測り、
閾値以上のユニットだけを動員して学習 -> anneal-until-stop に入る。読み出し
共有では語彙は線形にしか使えないので、目標とのチューニング相関は
「読み出しがそのユニットを使えるか」の直接の推定になる。アニールは案1 と
同じく動員済み語彙も候補に含む（プローブの過剰動員に対する安全網）。

検証の核は「安価な事前プローブが、高価なアニールの重なり決定を近似するか」:
同一 seed の案1 アーム（cra.run_sequence_prune）と比較する。

検証項目:
  V-a 各タスクが許容内（recruit アーム）
  V-b 忘却厳密ゼロ（両アーム）
  V-c プローブの類似度順序: 同じ sin 語彙プールに対し
      mean score(task3 = -sin) > mean score(task2 = cos)
  V-d 事前選択が事後剪定と同等の support に到達:
      |support_recruit| <= |support_prune| + 2 （task 2, 3）かつ MSE 許容内

生成物 (out/consolidation_recruit/):
  fig_recruit.png  プローブスコア + support 比較 (prune vs recruit) + 予測
  table_recruit.md 比較表 + 判定
  results.json

実行例:
  python tmp/consolidation_recruit.py --quick
  python tmp/consolidation_recruit.py
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
import consolidation_reuse_anneal as cra  # noqa: E402
from consolidation_lib import (  # noqa: E402
    eval_with_descriptor, freeze_masks, probe_tuning_corr, region_drift,
    region_snapshot, set_field_rho, vocab_energy, zero_cross_columns)


def probe_vocab(net, x, target, vocab: dict, args):
    """語彙だけを一時動員し、L2 語彙の |corr_x(zbar_k, y_new)| を返す
    (§12.9 案3 の類似度プローブ; 本体は consolidation_lib.probe_tuning_corr)."""
    return probe_tuning_corr(net, x, target, vocab, args.sigma,
                             args.crossing_h, passes=args.stat_passes)


# ============================================================
# 逐次実行（recruit アーム）
# ============================================================
def run_sequence_recruit(seed: int, args, device, tasks, x,
                         cleanup_thr: float = None, share_l1: bool = False):
    """案3 の逐次格納。cleanup_thr を与えると、格納に失敗したタスクの
    残骸領域を kill して free へ戻す（§12.9.8 の「掃除」）。share_l1=True は
    二層共有（§12.9.9 案A）: 過去 L1 全体を凍結基底として常時動員し、
    新タスクの L2 行がそれを読める（交差列を学習可能にする）。忘却は
    どちらでも厳密ゼロ。None / False なら従来動作."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    H = args.hidden_dim
    net = poc.build_net(H, args.sigma, args.crossing_h, args.num_samples,
                        device, scales=getattr(args, "l1_scales", None))
    past = {0: [], 1: []}
    free = {0: list(range(H)), 1: list(range(H))}
    registry = []

    for ti, (name, target) in enumerate(tasks):
        vocab_all = {l: sorted(past[l]) for l in (0, 1)}
        print(f"\n  ===== [recruit] task {ti + 1}: {name}  "
              f"(free L1 {len(free[0])} / L2 {len(free[1])}, "
              f"vocab L2 {len(vocab_all[1])}) =====")
        net.fcs[2].weight.data.zero_()
        net.fcs[2].bias.data.zero_()
        zero_cross_columns(net, past)

        # --- 類似度プローブ -> 動員する語彙の事前選択 -------------------
        scores = (probe_vocab(net, x, target, vocab_all, args)
                  if vocab_all[1] else {})
        recruited = sorted(k for k, s in scores.items()
                           if s >= args.recruit_thresh)
        src_rec = [s for s in registry
                   if any(k in recruited for k in s["region"][1])]
        voc = {0: sorted({k for s in src_rec for k in s["region"][0]}),
               1: recruited}
        if share_l1:
            voc = {0: sorted(past[0]), 1: recruited}   # L1 基底は常時全動員
        if scores:
            by_src = {s["name"]: [(k, round(scores[k], 3))
                                  for k in s["region"][1]] for s in registry}
            print(f"    probe scores: {by_src}")
            print(f"    recruited (>= {args.recruit_thresh}): {recruited} "
                  f"-> mobilised vocab {len(voc[0])}+{len(voc[1])}")

        set_field_rho(net, free, voc, args.sigma, args.crossing_h, 1.0)
        masks = (freeze_masks(H, past, device, share_l1=share_l1)
                 if (past[0] or past[1]) else None)

        # --- 初期学習（空き + 動員語彙のみ） ---------------------------
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

        # --- anneal-until-stop（自前 + 動員済み語彙が候補; 安全網） -----
        losses, holds, voc_alive = cra.anneal_until_stop_prune(
            net, x, target, tol, args, eligible=free, grad_masks=masks,
            vocab=voc, share_l1=share_l1)
        region = {l: [k for k in free[l] if float(net.sigma_vecs[l][k]) > 0]
                  for l in (0, 1)}
        src_alive = [s for s in registry
                     if any(k in voc_alive for k in s["region"][1])]
        voc_l1 = sorted({k for s in src_alive for k in s["region"][0]})
        if share_l1:
            voc_l1 = sorted(past[0])       # 交差重みが過去 L1 全体に張れる
        support = {0: sorted(region[0] + voc_l1),
                   1: sorted(region[1] + sorted(voc_alive))}
        r = {"name": name, "target": target, "region": region,
             "support": support, "sigma0": float(args.sigma),
             "h0": float(args.crossing_h),
             "wout": net.fcs[2].weight.data.clone(),
             "b_out": net.fcs[2].bias.data.clone(),
             "snap": region_snapshot(net, region, cols=support[0]),
             "snap_cols": support[0],
             "tol": tol,
             "holds": holds, "anneal_epochs": len(losses)}
        registry.append(r)
        pred = eval_with_descriptor(net, x, r)
        r["mse_at_consolidation"] = float(np.mean((pred - target) ** 2))
        r["probe_scores"] = {str(k): scores[k] for k in scores}
        r["recruited"] = recruited
        r["kept_vocab_by_source"] = {
            s["name"]: sorted(k for k in voc_alive if k in s["region"][1])
            for s in registry[:-1]}
        r["learn_mobilised"] = (len(free[0]) + len(free[1])
                                + len(voc[0]) + len(voc[1]))
        print(f"    own L1 {len(region[0])} + L2 {len(region[1])}; "
              f"recruited {len(recruited)} -> kept {len(voc_alive)}; "
              f"support {len(support[0])}+{len(support[1])}; "
              f"MSE={r['mse_at_consolidation']:.5f} (tol={tol:.4f})")
        if (cleanup_thr is not None
                and r["mse_at_consolidation"] > cleanup_thr):
            # 掃除: 失敗タスクの残骸を回収（kill して free に残す）
            for l in (0, 1):
                for k in region[l]:
                    poc.kill_unit(net, l, k)
            print(f"    [cleanup] 格納失敗 (MSE > {cleanup_thr}) -> "
                  f"L1 {len(region[0])} + L2 {len(region[1])} を回収")
            region = {0: [], 1: []}
            r["region"] = region
            r["support"] = {0: voc_l1, 1: sorted(voc_alive)}
            r["snap"] = region_snapshot(net, region, cols=r["support"][0])
            r["snap_cols"] = r["support"][0]
        past = {l: past[l] + region[l] for l in (0, 1)}
        free = {l: [k for k in free[l] if k not in region[l]] for l in (0, 1)}

    # ---------------- 最終評価 ----------------
    K = len(tasks)
    out = {"tasks": [], "preds": [], "registry": registry, "net": net}
    for i, r in enumerate(registry):
        pred = eval_with_descriptor(net, x, r)
        out["preds"].append(pred)
        mse = float(np.mean((pred - r["target"]) ** 2))
        drift = region_drift(net, r["region"], r["snap"],
                             cols=r["snap_cols"])
        share, by_source = vocab_energy(net, x, r, registry,
                                        passes=args.stat_passes)
        out["tasks"].append({
            "name": r["name"],
            "n_own": {"0": len(r["region"][0]), "1": len(r["region"][1])},
            "support_size": [len(r["support"][0]), len(r["support"][1])],
            "probe_scores": r["probe_scores"], "recruited": r["recruited"],
            "kept_vocab_by_source": r["kept_vocab_by_source"],
            "learn_mobilised": r["learn_mobilised"],
            "mse_final": mse, "drift": drift, "vocab_share": share,
            "vocab_share_by_source": by_source,
            "holds": r["holds"], "tol": r["tol"],
            "anneal_epochs": r["anneal_epochs"]})
        print(f"  [recruit] {r['name']}: final MSE={mse:.5f} "
              f"drift={drift:.2e} "
              f"support={len(r['support'][0])}+{len(r['support'][1])} "
              f"vocab_share={share:.3f}")
    return out


# ============================================================
# 図
# ============================================================
def make_figure(tasks, x_raw, res_pr, res_rc, args, seed, path):
    K = len(tasks)
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, K, height_ratios=[1.6, 1.6])

    # プローブスコア（task 2, 3）と閾値
    for i in range(1, K):
        ax = fig.add_subplot(gs[0, i])
        t = res_rc["tasks"][i]
        ks = sorted(int(k) for k in t["probe_scores"])
        vals = [t["probe_scores"][str(k)] for k in ks]
        colors = ["tab:green" if k in t["recruited"] else "tab:gray"
                  for k in ks]
        ax.bar(range(len(ks)), vals, color=colors)
        ax.axhline(args.recruit_thresh, ls="--", c="k", lw=0.8,
                   label=f"threshold {args.recruit_thresh}")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([str(k) for k in ks], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"probe |corr| for {t['name']}\n"
                     f"(green = recruited)", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        if i == 1:
            ax.set_ylabel("|corr(zbar_k, y_new)|")
            ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x_raw, tasks[2][1], "k--", lw=1.2, label="target")
    ax.plot(x_raw, res_rc["preds"][2], lw=1.4, label="recruit")
    t3 = res_rc["tasks"][2]
    ax.set_title(f"task 3 ({t3['name']}) prediction\n"
                 f"MSE={t3['mse_final']:.4f}", fontsize=9)
    ax.set_ylim(-1.6, 1.6)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # support / 学習時動員数の比較
    idx = np.arange(K)
    wd = 0.35
    ax = fig.add_subplot(gs[1, :])
    for off, arm, res in ((-wd / 2, "prune (case 1)", res_pr),
                          (wd / 2, "recruit (case 3)", res_rc)):
        sup = [t["support_size"][0] + t["support_size"][1]
               for t in res["tasks"]]
        color = "tab:blue" if "prune" in arm else "tab:green"
        ax.bar(idx + off, sup, wd, color=color, label=f"{arm}: support")
        if "recruit" in arm:
            lm = [t["learn_mobilised"] for t in res["tasks"]]
            ax.plot(idx + off, lm, "v", color="tab:red", ms=7,
                    label="recruit: mobilised during learning")
    ax.set_xticks(idx)
    ax.set_xticklabels([t[0] for t in tasks])
    ax.set_ylabel("units")
    ax.set_title("Support after consolidation: post-hoc pruning (case 1) vs "
                 "pre-selection by probe (case 3)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Soft consolidation step 3: similarity-gated recruitment "
                 f"(H={args.hidden_dim}, thresh={args.recruit_thresh}, "
                 f"seed {seed})", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    savefig(fig, path)


# ============================================================
# main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="soft consolidation step 3: "
                                            "similarity-gated recruitment")
    poc.add_poc_args(p)
    p.add_argument("--epochs-task", type=int, default=1000)
    p.add_argument("--stop-recovery", type=int, default=8)
    p.add_argument("--rho-init", type=float, default=1.0)
    p.add_argument("--recruit-thresh", type=float, default=0.5,
                   help="動員するプローブ相関の閾値")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/consolidation_recruit")
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
        res_pr = cra.run_sequence_prune(seed, args, device, tasks, x)
        res_rc = run_sequence_recruit(seed, args, device, tasks, x)

        pr, rc = res_pr["tasks"], res_rc["tasks"]
        sup_pr = [t["support_size"][0] + t["support_size"][1] for t in pr]
        sup_rc = [t["support_size"][0] + t["support_size"][1] for t in rc]
        va = all(t["mse_final"] <= 2.0 * t["tol"] for t in rc)
        vb = all(t["drift"] == 0.0 for t in pr + rc)
        # V-c: 同じ sin 語彙プールへの mean プローブスコアの類似度順序
        sin_units = [str(k) for k in res_rc["registry"][0]["region"][1]]
        m2 = np.mean([rc[1]["probe_scores"][k] for k in sin_units])
        m3 = np.mean([rc[2]["probe_scores"][k] for k in sin_units])
        vc = bool(m3 > m2)
        vd = all(sup_rc[i] <= sup_pr[i] + 2 for i in (1, 2))
        verdicts = {"Va_within_tol": bool(va), "Vb_zero_forgetting": bool(vb),
                    "Vc_probe_similarity": vc, "Vd_support_parity": bool(vd)}
        all_results[seed] = {
            "prune": {"tasks": pr, "support": sup_pr},
            "recruit": {"tasks": rc, "support": sup_rc,
                        "probe_mean_sin_vocab": {"task2": float(m2),
                                                 "task3": float(m3)}},
            "verdicts": verdicts}
        if seed == args.seed_list[0]:
            make_figure(tasks, x_raw, res_pr, res_rc, args, seed,
                        args.out_dir / "fig_recruit.png")

    # --- 表 ---
    seed0 = args.seed_list[0]
    r0 = all_results[seed0]
    lines = [f"**Soft consolidation step 3: similarity-gated recruitment** "
             f"(H={H}, T={args.num_samples}, thresh={args.recruit_thresh}, "
             f"seeds={args.seeds})", "",
             "| arm | task | own L1+L2 | support | recruited -> kept vocab "
             "| MSE final | drift |",
             "|---" * 7 + "|"]
    for arm in ("prune", "recruit"):
        for i, t in enumerate(r0[arm]["tasks"]):
            sup = r0[arm]["support"][i]
            if arm == "recruit":
                rk = (f"{len(t['recruited'])} -> "
                      f"{sum(len(v) for v in t['kept_vocab_by_source'].values())}")
            else:
                rk = f"all -> {t['kept_vocab_l2']}"
            lines.append(
                f"| {arm} | {t['name']} | {t['n_own']['0']}+{t['n_own']['1']} "
                f"| {sup} | {rk} | {t['mse_final']:.5f} | {t['drift']:.2e} |")
    pm = r0["recruit"]["probe_mean_sin_vocab"]
    lines += ["",
              f"- mean probe |corr| on sin-vocab pool: task2 (cos) = "
              f"{pm['task2']:.3f}, task3 (-sin) = {pm['task3']:.3f}", ""]
    for seed in args.seed_list:
        v = all_results[seed]["verdicts"]
        lines.append(f"- seed {seed}: "
                     + " / ".join(f"{k} {'**PASS**' if ok else '**FAIL**'}"
                                  for k, ok in v.items()))
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_recruit.md", table)
    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args),
               "results": {str(s): all_results[s] for s in all_results}})


if __name__ == "__main__":
    main()
