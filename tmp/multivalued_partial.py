"""
multivalued_partial.py — PoC-M2: 部分的多価性と場の部分的重なり
(docs/idea_multivalued.md §8 PoC-M2)

PoC-M1 は全域で 2 分岐に分かれるデータ（y = ±sin x）を扱った。本 PoC は
**入力領域の一部でだけ多価になる**データを扱う:

    y = sin(x)                      (x <= 0 : 単価値)
    y = sin(x) ± amp * sin(2x)      (x >  0 : 2 分岐)

分岐は背景 sin(x) を共有し、x > 0 でのみ ±amp*sin(2x) だけ違う。M1 との
本質的な差は三つあり、それぞれ新しい検証点になる:

  (i)   多峰性は**入力領域に依存**する。検出は大域的ではなく x の近傍窓
        ごとに走らせる必要がある（局所多峰性プロファイル）。検出された
        領域が真の多価領域 x > 0 と一致するかを測る。
  (ii)  場は**部分的に重なる**べき。共通部分（背景）は共有し、分岐部分
        だけが自前になるのが理想。M1 では第二分岐が 1 ユニットで済んだが、
        ここでは「多価領域の広さに比例したコスト」で済むかが問われる。
  (iii) 割当は**局所的**。単価値領域のデータ点はどちらの場に入れてもよい
        （don't-care）ので、割当精度は多価領域に限って評価する。

Phase 1 (single) : 単一場で学習し、局所多峰性プロファイルを測る
Phase 2 (split)  : 分岐 1 を学習 -> 凍結語彙として分岐 2 を読み出し共有で格納
Phase 3 (eval)   : 分岐 MSE（多価領域 / 単価値領域を分けて）、自前ユニット、
                   重なりの構造、局所割当精度

検証項目:
  V-a 局所的な潰れ: 多価領域の残差が単価値領域より大きく、単価値領域は
      対照と同水準（= 単一場でも共通部分は学べている）
  V-b 局所検出: 局所多峰性の p 値が多価領域で有意、単価値領域で非有意
  V-c 部分的重なり: 第二分岐の自前ユニットが M1 より多いが全体より十分小さく、
      分岐 MSE が単一場の多価領域残差より大幅に低い
  V-d 局所割当: 多価領域での割当精度が高い

生成物 (out/multivalued_partial/):
  fig_partial.png / table_partial.md / results.json

実行例:
  python tmp/multivalued_partial.py --quick
  python tmp/multivalued_partial.py --seeds 0,1,2
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
from consolidation_lib import predict, set_field  # noqa: E402
from consolidation_multiscale import SCALES  # noqa: E402
import multivalued_poc as mv  # noqa: E402


# ============================================================
# 部分的多価データ
# ============================================================
def make_partial(x_raw, amp=0.6):
    """x > 0 でのみ 2 分岐に分かれるデータ.

    返り値: b0, b1 [N]（2 分岐）, multi [N] bool（多価領域のマスク）
    """
    base = np.sin(x_raw).astype(np.float32)
    dev = (amp * np.sin(2.0 * x_raw)).astype(np.float32)
    multi = x_raw > 0.0
    b0 = base + np.where(multi, dev, 0.0).astype(np.float32)
    b1 = base - np.where(multi, dev, 0.0).astype(np.float32)
    return b0, b1, multi


# ============================================================
# 局所多峰性プロファイル
# ============================================================
def local_bimodality(x_raw, resid_b0, resid_b1, multi, args, seed,
                     n_win=8, win_frac=0.25):
    """x の近傍窓ごとに多峰性検定を走らせ、プロファイルを返す.

    各窓の中心 c について |x - c| < win_frac*range/2 の点を集め、その
    残差集合（2 分岐分）に対して bimodality_test を適用する。
    """
    lo, hi = float(x_raw.min()), float(x_raw.max())
    half = win_frac * (hi - lo) / 2.0
    centres = np.linspace(lo + half, hi - half, n_win)
    prof = []
    for c in centres:
        sel = np.abs(x_raw - c) < half
        vals = np.concatenate([resid_b0[sel], resid_b1[sel]])
        s = float(np.std(vals))
        if s < 1e-6 or sel.sum() < 8:
            prof.append({"centre": float(c), "score": 0.0, "p": 1.0,
                         "frac_multi": float(multi[sel].mean()),
                         "n": int(sel.sum())})
            continue
        r = mv.bimodality_test(vals, h=args.bimodal_c * s,
                               n_null=args.n_null, seed=seed)
        prof.append({"centre": float(c), "score": r["score"],
                     "p": r["p_value"],
                     "frac_multi": float(multi[sel].mean()),
                     "n": int(sel.sum())})
    return prof


def output_noise(net, x, passes=64):
    """各 x でのネットワーク自身の出力ゆらぎ σ_out(x)（単発 forward の std）.

    「データの広がりが大きいか」を判断する物差しは、ネットワークがそもそも
    区別できない揺らぎの大きさである。NNN は確率的なのでこれを自前で測れる。
    """
    with torch.no_grad():
        ys = torch.stack([net(x) for _ in range(passes)], dim=0)  # [P,N,1]
    return ys.std(dim=0, unbiased=False).squeeze(1).cpu().numpy()


def conditional_profile(x_raw, y_sets, sig_out, multi, n_win=12,
                        win_frac=0.2):
    """条件付き広がり比のプロファイル（ラベル不要・x で条件付け）.

    各 x で観測された y の集合の広がり spread(x) を、その点でのネットワーク
    出力ゆらぎ σ_out(x) と比べる。多価なら spread >> σ_out、単価値なら
    spread ~ 0 になる。窓内の周辺分布を見る多峰性検定と違い、**x をまたぐ
    変動を混ぜない**ので、系統的なフィット誤差の振動を多価性と誤認しない。

    一般化: 実データで x が厳密に重複しないときは x をビン分けするが、その
    幅は「ビン内での関数の変化 < 分岐の間隔」を満たす必要がある（§9.4 の
    検出限界と同型の分解能条件）。
    """
    spread = y_sets.std(axis=0)                    # [N]  (ラベル不要)
    ratio = spread / (sig_out + 1e-6)
    lo, hi = float(x_raw.min()), float(x_raw.max())
    half = win_frac * (hi - lo) / 2.0
    centres = np.linspace(lo + half, hi - half, n_win)
    prof = []
    for c in centres:
        sel = np.abs(x_raw - c) < half
        prof.append({"centre": float(c),
                     "ratio_med": float(np.median(ratio[sel])),
                     "ratio_p90": float(np.quantile(ratio[sel], 0.9)),
                     "frac_multi": float(multi[sel].mean())})
    return prof, spread, ratio


# ============================================================
# main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="PoC-M2: partial multivaluedness")
    poc.add_poc_args(p)
    p.add_argument("--epochs-task", type=int, default=600)
    p.add_argument("--stop-recovery", type=int, default=8)
    p.add_argument("--bimodal-c", type=float, default=0.35)
    p.add_argument("--n-null", type=int, default=200)
    p.add_argument("--amp", type=float, default=0.6,
                   help="多価領域での分岐の振幅")
    p.add_argument("--n-win", type=int, default=12,
                   help="局所多峰性プロファイルの窓数")
    p.add_argument("--win-frac", type=float, default=0.2,
                   help="局所窓の幅（x 全域に対する比）")
    p.add_argument("--n-points", type=int, default=512,
                   help="x のサンプル点数。局所検定は窓あたりの標本数で"
                        "検出力が決まるため M1 (128) より密に取る")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/multivalued_partial")
    args.l1_scales = SCALES
    if args.quick:
        args.epochs_task = 150
        args.epochs_per_step = 5
        args.stop_recovery = 3
        args.n_null = 100
        args.n_points = 256
        args.n_win = 8

    device = torch.device(args.device)
    N = args.n_points
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, N, dtype=np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    b0, b1, multi = make_partial(x_raw, amp=args.amp)
    H = args.hidden_dim
    uni = ~multi

    all_results = {}
    for seed in args.seed_list:
        print(f"\n===== seed {seed} =====")
        torch.manual_seed(seed)
        np.random.seed(seed)
        res = {}

        # ---------- Phase 1: 単一場 ----------
        print("\n  --- Phase 1: single field on partially multivalued data ---")
        net = poc.build_net(H, args.sigma, args.crossing_h, args.num_samples,
                            device, scales=SCALES)
        set_field(net, {0: list(range(H)), 1: list(range(H))},
                  args.sigma, args.crossing_h)
        t0 = torch.tensor(b0, device=device).unsqueeze(1)
        t1 = torch.tensor(b1, device=device).unsqueeze(1)
        tr0 = poc.CovJacTrainer(net, x, t0, lr=args.pre_lr, opt=args.opt,
                                jac_ema=args.jac_ema)
        tr1 = poc.CovJacTrainer(net, x, t1, lr=args.pre_lr, opt=args.opt,
                                jac_ema=args.jac_ema)
        losses_single = []
        for e in range(args.epochs_task):
            l0, l1 = tr0.step(), tr1.step()
            losses_single.append(0.5 * (l0 + l1))
            if e % max(1, args.epochs_task // 4) == 0:
                print(f"    [single] epoch {e:5d} mse={losses_single[-1]:.5f}")
        tr0.close()
        tr1.close()
        pred_single = predict(net, x, passes=32)
        r0_s, r1_s = b0 - pred_single, b1 - pred_single
        mse_multi = float(np.mean(np.concatenate([r0_s[multi], r1_s[multi]]) ** 2))
        mse_uni = float(np.mean(np.concatenate([r0_s[uni], r1_s[uni]]) ** 2))
        print(f"    single: residual MSE  multi-region={mse_multi:.4f}  "
              f"single-valued region={mse_uni:.5f}")
        print(f"    (predicted floor in multi region = amp^2/2 * <sin^2(2x)> "
              f"= {args.amp ** 2 / 4:.4f})")

        # 局所多峰性プロファイル
        prof = local_bimodality(x_raw, r0_s, r1_s, multi, args, seed,
                                n_win=args.n_win, win_frac=args.win_frac)
        for w in prof:
            tag = "MULTI" if w["frac_multi"] > 0.5 else "uni  "
            det = "DETECT" if w["p"] < 0.05 else "      "
            print(f"      window c={w['centre']:+.2f} [{tag}] "
                  f"score={w['score']:.3f} p={w['p']:.3f} {det}")
        # 条件付き広がり比（x で条件付けた検出器）
        sig_out = output_noise(net, x)
        y_sets = np.stack([b0, b1], axis=0)        # [2, N]
        cprof, spread, ratio = conditional_profile(
            x_raw, y_sets, sig_out, multi, n_win=args.n_win,
            win_frac=args.win_frac)
        print(f"    sigma_out (network's own output std): "
              f"median={np.median(sig_out):.4f}")
        for w in cprof:
            tag = "MULTI" if w["frac_multi"] > 0.5 else "uni  "
            det = "DETECT" if w["ratio_med"] > 1.0 else "      "
            print(f"      window c={w['centre']:+.2f} [{tag}] "
                  f"spread/noise med={w['ratio_med']:.2f} "
                  f"p90={w['ratio_p90']:.2f} {det}")
        cm = [w for w in cprof if w["frac_multi"] > 0.75]
        cu = [w for w in cprof if w["frac_multi"] < 0.25]
        crate_multi = float(np.mean([w["ratio_med"] > 1.0 for w in cm])) \
            if cm else 0.0
        crate_uni = float(np.mean([w["ratio_med"] > 1.0 for w in cu])) \
            if cu else 0.0
        print(f"    conditional detection: multi={crate_multi:.2f} "
              f"uni={crate_uni:.2f}")

        det_multi = [w for w in prof if w["frac_multi"] > 0.75]
        det_uni = [w for w in prof if w["frac_multi"] < 0.25]
        rate_multi = float(np.mean([w["p"] < 0.05 for w in det_multi])) \
            if det_multi else 0.0
        rate_uni = float(np.mean([w["p"] < 0.05 for w in det_uni])) \
            if det_uni else 0.0
        print(f"    local detection: multi-region windows {rate_multi:.2f}, "
              f"single-valued windows {rate_uni:.2f}")

        # 対照: 単価値データ（b0 のみ）を同設定で学習した残差
        torch.manual_seed(seed)
        np.random.seed(seed)
        net_c = poc.build_net(H, args.sigma, args.crossing_h,
                              args.num_samples, device, scales=SCALES)
        set_field(net_c, {0: list(range(H)), 1: list(range(H))},
                  args.sigma, args.crossing_h)
        trc = poc.CovJacTrainer(net_c, x, t0, lr=args.pre_lr, opt=args.opt,
                                jac_ema=args.jac_ema)
        trc.run(args.epochs_task)
        trc.close()
        r_ctrl = b0 - predict(net_c, x, passes=32)
        mse_ctrl = float(np.mean(r_ctrl ** 2))
        print(f"    control (single-valued): MSE={mse_ctrl:.5f}")

        res["single"] = {"mse_multi": mse_multi, "mse_uni": mse_uni,
                         "profile": prof, "rate_multi": rate_multi,
                         "rate_uni": rate_uni,
                         "cond_profile": cprof,
                         "cond_rate_multi": crate_multi,
                         "cond_rate_uni": crate_uni,
                         "sigma_out_med": float(np.median(sig_out))}
        res["control"] = {"mse": mse_ctrl}

        # ---------- Phase 2: 分割 ----------
        print("\n  --- Phase 2: split (branch 2 shares branch 1 as vocab) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        net2 = poc.build_net(H, args.sigma, args.crossing_h, args.num_samples,
                            device, scales=SCALES)
        free = {0: list(range(H)), 1: list(range(H))}
        past = {0: [], 1: []}
        print("    [branch 1]")
        d0, reg0 = mv.learn_branch(net2, x, b0, free, past, args, device,
                                   share=False)
        past = {l: past[l] + reg0[l] for l in (0, 1)}
        free = {l: [k for k in free[l] if k not in reg0[l]] for l in (0, 1)}
        print("    [branch 2] (readout sharing)")
        d1, reg1 = mv.learn_branch(net2, x, b1, free, past, args, device,
                                   share=True)
        descriptors = [d0, d1]

        y_pts = np.concatenate([b0, b1])
        assign_idx, P = mv.assign(net2, x, y_pts, descriptors)
        # 割当は多価領域でのみ評価（単価値領域は don't-care）
        br_true = np.concatenate([np.zeros(N, int), np.ones(N, int)])
        m2 = np.concatenate([multi, multi])
        acc = float((assign_idx[m2] == br_true[m2]).mean())
        acc = max(acc, 1.0 - acc)
        mse_b = [float(np.mean((P[i][br_true == i] - y_pts[br_true == i]) ** 2))
                 for i in range(2)]
        mse_b_multi = [float(np.mean((P[i][(br_true == i) & m2]
                                      - y_pts[(br_true == i) & m2]) ** 2))
                       for i in range(2)]
        own = [len(reg0[0]) + len(reg0[1]), len(reg1[0]) + len(reg1[1])]
        w0 = d0["wout"].squeeze(0)
        w1 = d1["wout"].squeeze(0)
        cos = float((w0 @ w1) / (w0.norm() * w1.norm() + 1e-9))
        print(f"    branch MSE (all) = {[round(m, 5) for m in mse_b]}; "
              f"(multi region) = {[round(m, 5) for m in mse_b_multi]}")
        print(f"    own units = {own}; readout cos = {cos:.3f}; "
              f"local assignment accuracy = {acc:.3f}")
        res["split"] = {"mse_branch": mse_b, "mse_branch_multi": mse_b_multi,
                        "own_units": own, "readout_cos": cos,
                        "assign_acc_multi": acc,
                        "holds": [d0["holds"], d1["holds"]]}

        # ---------- 判定 ----------
        va = mse_multi > 5.0 * max(mse_uni, 1e-9) and mse_uni < 20.0 * mse_ctrl
        vb = crate_multi >= 0.75 and crate_uni <= 0.25
        vc = (own[1] < own[0] and max(mse_b_multi) < 0.3 * mse_multi)
        vd = acc > 0.9
        verdicts = {"Va_local_collapse": bool(va),
                    "Vb_local_detection": bool(vb),
                    "Vc_partial_overlap": bool(vc),
                    "Vd_local_assignment": bool(vd)}
        res["verdicts"] = verdicts
        all_results[seed] = res
        print(f"\n  ===== verdicts: {verdicts} =====")

        # ---------- 図 ----------
        if seed == args.seed_list[0]:
            fig = plt.figure(figsize=(13, 8))
            gs = fig.add_gridspec(2, 3)
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(x_raw, b0, "k--", lw=1.0, label="branch A")
            ax.plot(x_raw, b1, "k:", lw=1.0, label="branch B")
            ax.plot(x_raw, pred_single, lw=1.6, c="tab:red",
                    label="single field")
            ax.axvspan(0, x_raw.max(), color="tab:orange", alpha=0.10)
            ax.set_title(f"Partial multivaluedness (shaded = multi region)\n"
                         f"residual MSE: multi {mse_multi:.3f} / "
                         f"uni {mse_uni:.4f}", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

            ax = fig.add_subplot(gs[0, 1])
            cs = [w["centre"] for w in prof]
            ps = [w["p"] for w in prof]
            cols = ["tab:red" if w["frac_multi"] > 0.5 else "tab:gray"
                    for w in prof]
            ax.bar(cs, [-np.log10(max(p, 1e-3)) for p in ps],
                   width=(cs[1] - cs[0]) * 0.8 if len(cs) > 1 else 1.0,
                   color=cols)
            ax.axhline(-np.log10(0.05), ls="--", c="k", lw=0.9,
                       label="p = 0.05")
            ax.axvline(0, ls=":", c="tab:orange", lw=1.2,
                       label="true boundary")
            ax.set_xlabel("window centre in x")
            ax.set_ylabel("-log10 p (bimodality)")
            ax.set_title("Local bimodality profile", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, axis="y")

            ax = fig.add_subplot(gs[0, 2])
            ax.plot(x_raw, ratio, lw=1.0, c="tab:purple",
                    label="spread / noise")
            ax.axhline(1.0, ls="--", c="k", lw=0.9, label="ratio = 1")
            ax.axvline(0, ls=":", c="tab:orange", lw=1.2)
            ax.axvspan(0, x_raw.max(), color="tab:orange", alpha=0.10)
            ax.set_yscale("symlog", linthresh=0.1)
            ax.set_xlabel("x")
            ax.set_title("Conditional detector: data spread at x\n"
                         "vs network's own output noise", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

            ax = fig.add_subplot(gs[1, 0])
            ax.plot(x_raw, b0, "k--", lw=1.0)
            ax.plot(x_raw, b1, "k:", lw=1.0)
            ax.plot(x_raw, P[0][:N], lw=1.4, c="tab:blue",
                    label=f"field 1 ({mse_b[0]:.4f})")
            ax.plot(x_raw, P[1][:N], lw=1.4, c="tab:green",
                    label=f"field 2 ({mse_b[1]:.4f})")
            ax.axvspan(0, x_raw.max(), color="tab:orange", alpha=0.10)
            ax.set_title(f"After split: own units {own}, cos={cos:.2f}",
                         fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

            ax = fig.add_subplot(gs[1, 1])
            ax.plot(losses_single, lw=0.8, c="tab:red", label="single field")
            ax.axhline(mse_multi, ls="--", c="tab:red", lw=0.8,
                       label="multi-region floor")
            ax.axhline(max(mse_b_multi), ls="--", c="tab:green", lw=0.8,
                       label="after split (multi region)")
            ax.set_yscale("log")
            ax.set_xlabel("epoch")
            ax.set_ylabel("MSE")
            ax.set_title("Loss floor", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, which="both")

            ax = fig.add_subplot(gs[1, 2])
            ax.axis("off")
            txt = "\n".join([f"{k}: {'PASS' if v else 'FAIL'}"
                             for k, v in verdicts.items()])
            extra = (f"\n\nconditional detector\n  multi   : {crate_multi:.2f}"
                     f"\n  uni     : {crate_uni:.2f}"
                     f"\n\npooled bimodality\n  multi   : {rate_multi:.2f}"
                     f"\n  uni     : {rate_uni:.2f} (false pos.)"
                     f"\n\nassignment (multi): {acc:.3f}")
            ax.text(0.0, 0.95, "verdicts\n\n" + txt + extra, fontsize=10,
                    va="top", family="monospace")
            fig.suptitle(f"PoC-M2: partial multivaluedness "
                         f"(amp={args.amp}, H={H}, seed {seed})", fontsize=12)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
            savefig(fig, args.out_dir / "fig_partial.png")

    # ---------- 表 ----------
    seed0 = args.seed_list[0]
    r0 = all_results[seed0]
    lines = [f"**PoC-M2: partial multivaluedness** (H={H}, T={args.num_samples}"
             f", amp={args.amp}, epochs-task={args.epochs_task}, "
             f"seeds={args.seeds})", "",
             "single field:", "",
             "| region | residual MSE | conditional detector | pooled "
             "bimodality (flawed) |",
             "|---|---|---|---|",
             f"| multi-valued (x>0) | **{r0['single']['mse_multi']:.4f}** "
             f"| **{r0['single']['cond_rate_multi']:.2f}** "
             f"| {r0['single']['rate_multi']:.2f} |",
             f"| single-valued (x<=0) | {r0['single']['mse_uni']:.5f} "
             f"| **{r0['single']['cond_rate_uni']:.2f}** "
             f"| {r0['single']['rate_uni']:.2f} (false pos.) |",
             f"| control (fully single-valued) | {r0['control']['mse']:.5f} "
             f"| — | — |", "",
             "after split:", "",
             f"- branch MSE (all / multi region): "
             f"{[round(m, 5) for m in r0['split']['mse_branch']]} / "
             f"{[round(m, 5) for m in r0['split']['mse_branch_multi']]}",
             f"- own units per branch: {r0['split']['own_units']}",
             f"- readout cosine similarity: {r0['split']['readout_cos']:.3f}",
             f"- assignment accuracy (multi region only): "
             f"{r0['split']['assign_acc_multi']:.3f}", "",
             "local bimodality profile (seed 0):", "",
             "| window centre | region | score | p |",
             "|---|---|---|---|"]
    for w in r0["single"]["profile"]:
        reg = "multi" if w["frac_multi"] > 0.5 else "uni"
        lines.append(f"| {w['centre']:+.2f} | {reg} | {w['score']:.3f} "
                     f"| {w['p']:.3f} |")
    lines.append("")
    for seed in args.seed_list:
        v = all_results[seed]["verdicts"]
        lines.append(f"- seed {seed}: "
                     + " / ".join(f"{k} {'**PASS**' if ok else '**FAIL**'}"
                                  for k, ok in v.items()))
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_partial.md", table)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args),
               "results": clean(all_results)})


if __name__ == "__main__":
    main()
