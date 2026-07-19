"""
multivalued_poc.py — PoC-M1: 多価データの検出と場の自己分割
(docs/idea_multivalued.md §8)

多価データ y = ±sin(x)（各 50%）を単一のノイズ場で学習すると、二乗誤差は
条件付き平均 E[y|x] = 0 へ収束し、残差は条件付き分散に下げ止まる。本 PoC は

  Phase 1 (single) : 単一場で学習し、下げ止まりと検出シグナルを測る
      - 既約性 : アンサンブルのパス数 P を増やしても残差が減らない
                 （内在ノイズ由来なら 1/P で減る）
      - 多峰性 : 残差分布が単に広いのでなく多峰である
                 （NNN 的実装 = 残差にしきい値を掃引した交差カウントの
                   差分で密度を読む。分布フリー、forward 統計のみ）
      - 対照   : 空きユニットがあるのに損失が下がらない（容量不足でない）
                 → 全ネットワーク単独学習の単価値タスクとの比較で確認
  Phase 2 (split)  : 分割して 2 つの場へ。第二分岐は既存場を凍結語彙として
                     読み出し共有（案2/3 の規約）で格納する。割当は残差最小
                     の場への EM（E-step = クロス評価、forward のみ）。
  Phase 3 (eval)   : 分割後の各分岐の MSE、自前ユニット数、読み出しの関係
                     （±sin なら符号反転 = コサイン類似度 -1 を予測）

検証項目:
  V-a 単一場は潰れる: single の予測が |ȳ| ~ 0、残差が P に依らず下げ止まる
  V-b 検出: 多峰性スコアが単価値対照より有意に大きい
  V-c 分割が効く: 分割後の各分岐 MSE << 単一場の下げ止まり残差
  V-d 分岐は安い: 第二分岐の自前ユニット数が小さく、読み出しが反相関

生成物 (out/multivalued_poc/):
  fig_multivalued.png  予測曲線 + 残差密度 + P 依存性 + 分割後の分岐
  table_multivalued.md 判定表
  results.json

実行例:
  python tmp/multivalued_poc.py --quick
  python tmp/multivalued_poc.py --seeds 0
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
from consolidation_lib import (  # noqa: E402
    eval_with_descriptor, freeze_masks, predict, set_field,
    zero_cross_columns)
from consolidation_multiscale import SCALES  # noqa: E402


# ============================================================
# 多価データ
# ============================================================
def make_multivalued(x_raw, kind="pm_sin"):
    """(x, y) の多価データ。x は 2 回ずつ現れ、y が 2 分岐を取る.

    返り値: x_pts [2N], y_pts [2N], branches [2N]（真の分岐 id; 評価用のみ）
    """
    if kind == "pm_sin":
        b0 = np.sin(x_raw).astype(np.float32)
        b1 = (-np.sin(x_raw)).astype(np.float32)
    else:
        raise ValueError(kind)
    x_pts = np.concatenate([x_raw, x_raw])
    y_pts = np.concatenate([b0, b1])
    branches = np.concatenate([np.zeros(len(x_raw), dtype=int),
                               np.ones(len(x_raw), dtype=int)])
    return x_pts, y_pts, branches, (b0, b1)


# ============================================================
# 検出シグナル（すべて forward 統計）
# ============================================================
def residuals(net, x, y_pts, n_x, passes):
    """アンサンブル平均予測に対する各データ点の残差 [2N]."""
    pred = predict(net, x, passes=passes)          # [N]
    pred2 = np.concatenate([pred, pred])           # 2 分岐とも同じ x
    return y_pts - pred2, pred


def irreducibility(net, x, y_pts, n_x, pass_list=(1, 2, 4, 8, 16, 32)):
    """残差 RMS の P 依存性。内在ノイズなら 1/sqrt(P) で減る成分を持つ."""
    out = {}
    for P in pass_list:
        r, _ = residuals(net, x, y_pts, n_x, P)
        out[P] = float(np.sqrt(np.mean(r ** 2)))
    return out


def crossing_density(vals, grid, h):
    """交差活性内蔵の密度推定と同型の、しきい値掃引による分布フリー密度.

    各しきい値 c について「|vals - c| < h」の頻度を数える = 幅 2h の窓の
    カウント。NNN の (xor2 - xor1)/(2h) と同じく、しきい値を動かした
    交差カウントの差分から密度を読む操作にあたる。
    """
    v = vals[None, :]
    c = grid[:, None]
    return (np.abs(v - c) < h).mean(axis=1) / (2.0 * h)


def bimodality(vals, h=0.15, n_grid=201):
    """多峰性スコア: 二つの峰に挟まれた最も深い谷の相対深さ.

    分割点 k ごとに「左の最大 / 右の最大 / 谷 = dens[k]」を見て

        score = max_k [ 1 - dens[k] / min(max(dens[:k]), max(dens[k:])) ]

    を返す（prefix/suffix 最大で O(n)）。単峰なら谷の両側の最大の一方が
    谷自身に接するので 0、明瞭な 2 峰なら 1 に近づく。分布フリーで
    forward 統計のみ（密度は交差掃引 crossing_density で読む）。
    """
    # 格子は分位点で刈り込む: 裾の孤立した外れ値の手前では密度が厳密に 0 に
    # なり、score = 1 - 0/peak = 1 という偽の「谷」を作るため（帰無分布が
    # 常に 1.0 に張り付く原因だった）
    lo, hi = (float(np.quantile(vals, 0.02)), float(np.quantile(vals, 0.98)))
    if hi - lo < 1e-6:
        return 0.0, None, None
    grid = np.linspace(lo, hi, n_grid)
    dens = crossing_density(vals, grid, h)
    n = len(grid)
    pre = np.maximum.accumulate(dens)              # max(dens[:k+1])
    suf = np.maximum.accumulate(dens[::-1])[::-1]  # max(dens[k:])
    # 峰は谷から窓幅 h 以上離れていることを要求する（サンプル揺らぎによる
    # 幅の狭い谷を峰の対と誤認しないため。要求しないと一様分布のような
    # 平坦・重裾の分布で偽陽性が出る）
    m = max(1, int(round(h / (grid[1] - grid[0]))))
    peak_lo = np.zeros(n)
    for k in range(n):
        left = pre[k - m] if k - m >= 0 else 0.0
        right = suf[k + m] if k + m < n else 0.0
        peak_lo[k] = min(left, right)
    ok = peak_lo > 1e-9
    score = np.zeros(n)
    score[ok] = 1.0 - dens[ok] / peak_lo[ok]
    return float(max(0.0, score.max())), grid, dens


def bimodality_test(vals, h=0.15, n_null=200, seed=0):
    """多峰性スコアと、その帰無較正（分散を合わせた単峰ガウス）.

    スコアは格子上の最大値統計なので、標本数・窓幅に依存する帰無分布を持つ
    （固定閾値での判定は誤り — 一様分布のような平坦・重裾の標本でも
    揺らぎの谷を拾って高い値が出る）。そこで同じ標本数・同じ分散の単峰
    ガウスからパラメトリック・ブートストラップで帰無分布を作り、
    p 値と 95 パーセンタイル閾値を返す。

    注意: この検定が見ているのは「単峰ガウスからの逸脱」であり、多峰性は
    その一形態にすぎない。分割の最終的な正当化はスコアではなく、分割が
    実際に残差をユニットコスト以上に下げること（MDL 判定）で与えるべきである。
    """
    score, grid, dens = bimodality(vals, h=h)
    rng = np.random.default_rng(seed)
    s = float(np.std(vals))
    null = np.array([bimodality(rng.normal(0.0, s, len(vals)), h=h)[0]
                     for _ in range(n_null)])
    p = float((1 + int((null >= score).sum())) / (1 + n_null))
    return {"score": score, "null_q95": float(np.quantile(null, 0.95)),
            "p_value": p, "grid": grid, "dens": dens}


# ============================================================
# Phase 2: 分割（第二分岐を読み出し共有で格納）
# ============================================================
def assign(net, x, y_pts, descriptors, passes=16):
    """E-step: 各データ点を残差最小の場へ割り当てる（forward のみ）."""
    preds = []
    for r in descriptors:
        p = eval_with_descriptor(net, x, r, passes=passes)
        preds.append(np.concatenate([p, p]))
    P = np.stack(preds, axis=0)                    # [n_fields, 2N]
    err = (P - y_pts[None, :]) ** 2
    return np.argmin(err, axis=0), P


def learn_branch(net, x, y_target, free, past, args, device, share=True):
    """空き領域 + 凍結語彙（past）で 1 分岐を学習し anneal-until-stop で整理."""
    H = args.hidden_dim
    net.fcs[2].weight.data.zero_()
    net.fcs[2].bias.data.zero_()
    if share and (past[0] or past[1]):
        zero_cross_columns(net, past)
        masks = freeze_masks(H, past, device)
        vocab = {l: sorted(past[l]) for l in (0, 1)}
    else:
        masks, vocab = None, {0: [], 1: []}
    mobil = {l: sorted(set(free[l]) | set(vocab[l])) for l in (0, 1)}
    set_field(net, mobil, args.sigma, args.crossing_h)
    t = torch.tensor(y_target, device=device).unsqueeze(1)
    tr = poc.CovJacTrainer(net, x, t, lr=args.pre_lr, opt=args.opt,
                           jac_ema=args.jac_ema)
    tr.grad_masks = masks
    for e in range(args.epochs_task):
        loss = tr.step()
        if e % max(1, args.epochs_task // 3) == 0:
            print(f"      [learn] epoch {e:5d} mse={loss:.5f}")
    tr.close()
    base = float(np.mean(tr.losses[-min(100, len(tr.losses)):]))
    tol = base * args.drift_mult + args.drift_abs
    _, holds = csoft.anneal_until_stop(net, x, y_target, tol, args,
                                       eligible=free, grad_masks=masks,
                                       vocab=vocab)
    region = {l: [k for k in free[l] if float(net.sigma_vecs[l][k]) > 0]
              for l in (0, 1)}
    r = {"name": "branch", "target": y_target, "region": region,
         "support": {l: sorted(region[l] + vocab[l]) for l in (0, 1)},
         "sigma0": float(args.sigma), "h0": float(args.crossing_h),
         "wout": net.fcs[2].weight.data.clone(),
         "b_out": net.fcs[2].bias.data.clone(), "tol": tol, "holds": holds}
    return r, region


# ============================================================
# main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="PoC-M1: multivalued data, "
                                            "field self-splitting")
    poc.add_poc_args(p)
    p.add_argument("--epochs-task", type=int, default=600)
    p.add_argument("--stop-recovery", type=int, default=8)
    p.add_argument("--bimodal-c", type=float, default=0.35,
                   help="残差密度の窓幅（h = c * std）。固定幅では標本の"
                        "広がりに追随できず検出力が消えるため相対指定")
    p.add_argument("--n-null", type=int, default=200,
                   help="多峰性検定の帰無ブートストラップ回数")
    args = fncl.finalize_args(p.parse_args(), default_out="out/multivalued_poc")
    args.l1_scales = SCALES
    if args.quick:
        args.epochs_task = 150
        args.epochs_per_step = 5
        args.stop_recovery = 3

    device = torch.device(args.device)
    N = 128
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, N, dtype=np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    x_pts, y_pts, branches, (b0, b1) = make_multivalued(x_raw)
    H = args.hidden_dim

    all_results = {}
    for seed in args.seed_list:
        print(f"\n===== seed {seed} =====")
        torch.manual_seed(seed)
        np.random.seed(seed)
        res = {}

        # ---------- Phase 1: 単一場 ----------
        print("\n  --- Phase 1: single field on multivalued data ---")
        net = poc.build_net(H, args.sigma, args.crossing_h, args.num_samples,
                            device, scales=SCALES)
        set_field(net, {0: list(range(H)), 1: list(range(H))},
                  args.sigma, args.crossing_h)
        # 多価データ: 同じ x に 2 つの目標 -> 目標は平均 (MSE の最適解と同値)
        # だが学習則には各点を等頻度で見せるため、交互提示で 1 epoch を構成する
        t0 = torch.tensor(b0, device=device).unsqueeze(1)
        t1 = torch.tensor(b1, device=device).unsqueeze(1)
        tr0 = poc.CovJacTrainer(net, x, t0, lr=args.pre_lr, opt=args.opt,
                                jac_ema=args.jac_ema)
        tr1 = poc.CovJacTrainer(net, x, t1, lr=args.pre_lr, opt=args.opt,
                                jac_ema=args.jac_ema)
        losses_single = []
        for e in range(args.epochs_task):
            l0 = tr0.step()
            l1 = tr1.step()
            losses_single.append(0.5 * (l0 + l1))
            if e % max(1, args.epochs_task // 4) == 0:
                print(f"    [single] epoch {e:5d} mse={losses_single[-1]:.5f}")
        tr0.close()
        tr1.close()
        pred_single = predict(net, x, passes=16)
        irr = irreducibility(net, x, y_pts, N)
        r_single, _ = residuals(net, x, y_pts, N, 32)
        bt = bimodality_test(r_single, h=args.bimodal_c * float(np.std(r_single)),
                             n_null=args.n_null, seed=seed)
        bim, grid, dens = bt["score"], bt["grid"], bt["dens"]
        mse_single = float(np.mean(r_single ** 2))
        print(f"    single: |mean pred| = {np.abs(pred_single).mean():.4f}, "
              f"residual MSE = {mse_single:.4f}")
        print(f"    irreducibility (RMS vs passes): "
              f"{ {k: round(v, 4) for k, v in irr.items()} }")
        print(f"    bimodality score = {bim:.3f} "
              f"(null q95 = {bt['null_q95']:.3f}, p = {bt['p_value']:.3f})")

        # 対照: 単価値データ（sin のみ）を同じ設定で学習した残差
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
        pred_c = predict(net_c, x, passes=32)
        r_ctrl = b0 - pred_c
        bt_c = bimodality_test(r_ctrl, h=args.bimodal_c * float(np.std(r_ctrl)),
                               n_null=args.n_null, seed=seed)
        bim_c, grid_c, dens_c = bt_c["score"], bt_c["grid"], bt_c["dens"]
        irr_c = irreducibility(net_c, x, np.concatenate([b0, b0]), N)
        print(f"    control (single-valued sin): residual MSE = "
              f"{float(np.mean(r_ctrl ** 2)):.5f}, bimodality = {bim_c:.3f} "
              f"(p = {bt_c['p_value']:.3f})")

        res["single"] = {"mse": mse_single, "bimodality": bim,
                         "bimodal_p": bt["p_value"],
                         "bimodal_null_q95": bt["null_q95"],
                         "irreducibility": irr,
                         "abs_mean_pred": float(np.abs(pred_single).mean())}
        res["control"] = {"mse": float(np.mean(r_ctrl ** 2)),
                          "bimodality": bim_c, "bimodal_p": bt_c["p_value"],
                          "bimodal_null_q95": bt_c["null_q95"],
                          "irreducibility": irr_c}

        # ---------- Phase 2: 分割 ----------
        print("\n  --- Phase 2: split into two fields ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        net2 = poc.build_net(H, args.sigma, args.crossing_h, args.num_samples,
                            device, scales=SCALES)
        free = {0: list(range(H)), 1: list(range(H))}
        past = {0: [], 1: []}
        descriptors = []
        # 分岐 1: 空き全域（この時点では「多価と分かった後の一方」を学ぶ）
        print("    [branch 1]")
        d0, reg0 = learn_branch(net2, x, b0, free, past, args, device,
                                share=False)
        descriptors.append(d0)
        past = {l: past[l] + reg0[l] for l in (0, 1)}
        free = {l: [k for k in free[l] if k not in reg0[l]] for l in (0, 1)}
        # 分岐 2: 分岐 1 を凍結語彙として読み出し共有
        print("    [branch 2] (readout sharing with branch 1)")
        d1, reg1 = learn_branch(net2, x, b1, free, past, args, device,
                                share=True)
        descriptors.append(d1)

        # 割当（E-step）と分岐ごとの MSE
        assign_idx, P = assign(net2, x, y_pts, descriptors)
        acc = float((assign_idx == branches).mean())
        acc = max(acc, 1.0 - acc)          # ラベル入替は同値
        mse_b = [float(np.mean((P[i][branches == i] - y_pts[branches == i]) ** 2))
                 for i in range(2)]
        w0 = descriptors[0]["wout"].squeeze(0)
        w1 = descriptors[1]["wout"].squeeze(0)
        cos = float((w0 @ w1) / (w0.norm() * w1.norm() + 1e-9))
        own = [len(reg0[0]) + len(reg0[1]), len(reg1[0]) + len(reg1[1])]
        print(f"    branch MSE = {[round(m, 5) for m in mse_b]}; "
              f"own units = {own}; readout cosine = {cos:.3f}; "
              f"assignment accuracy = {acc:.3f}")
        res["split"] = {"mse_branch": mse_b, "own_units": own,
                        "readout_cos": cos, "assign_acc": acc,
                        "holds": [d0["holds"], d1["holds"]]}

        # ---------- 判定 ----------
        va = (res["single"]["abs_mean_pred"] < 0.2
              and irr[32] > 0.5 * irr[1])            # P を増やしても下げ止まる
        vb = (bt["p_value"] < 0.05 and bt_c["p_value"] >= 0.05)
        vc = max(mse_b) < 0.5 * mse_single
        vd = own[1] <= max(2, own[0] // 2) and cos < -0.5
        verdicts = {"Va_collapse": bool(va), "Vb_detection": bool(vb),
                    "Vc_split_helps": bool(vc), "Vd_branch_is_cheap": bool(vd)}
        res["verdicts"] = verdicts
        all_results[seed] = res
        print(f"\n  ===== verdicts: {verdicts} =====")

        # ---------- 図 ----------
        if seed == args.seed_list[0]:
            fig = plt.figure(figsize=(13, 8))
            gs = fig.add_gridspec(2, 3)
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(x_raw, b0, "k--", lw=1.0, label="branch +sin")
            ax.plot(x_raw, b1, "k:", lw=1.0, label="branch -sin")
            ax.plot(x_raw, pred_single, lw=1.6, c="tab:red",
                    label="single field")
            ax.set_title(f"Single field collapses to E[y|x]\n"
                         f"|mean pred|={np.abs(pred_single).mean():.3f}, "
                         f"MSE={mse_single:.3f}", fontsize=9)
            ax.set_ylim(-1.6, 1.6)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

            ax = fig.add_subplot(gs[0, 1])
            if grid is not None:
                ax.plot(grid, dens, lw=1.6, c="tab:red",
                        label=f"multivalued ({bim:.2f}, p={bt['p_value']:.3f})")
            if grid_c is not None:
                ax.plot(grid_c, dens_c, lw=1.6, c="tab:gray",
                        label=f"control ({bim_c:.2f}, p={bt_c['p_value']:.3f})")
            ax.set_xlabel("residual y - ȳ(x)")
            ax.set_ylabel("density (crossing sweep)")
            ax.set_title("Bimodality of the residual", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

            ax = fig.add_subplot(gs[0, 2])
            Ps = sorted(irr)
            ax.plot(Ps, [irr[k] for k in Ps], "o-", c="tab:red",
                    label="multivalued")
            ax.plot(Ps, [irr_c[k] for k in Ps], "o-", c="tab:gray",
                    label="control")
            ref = irr[1] / np.sqrt(np.array(Ps, dtype=float))
            ax.plot(Ps, ref, ":", c="k", label="1/sqrt(P) reference")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("ensemble passes P")
            ax.set_ylabel("residual RMS")
            ax.set_title("Irreducibility: residual vs P", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, which="both")

            ax = fig.add_subplot(gs[1, 0])
            ax.plot(x_raw, b0, "k--", lw=1.0)
            ax.plot(x_raw, b1, "k:", lw=1.0)
            ax.plot(x_raw, P[0][:N], lw=1.4, c="tab:blue",
                    label=f"field 1 (MSE {mse_b[0]:.4f})")
            ax.plot(x_raw, P[1][:N], lw=1.4, c="tab:green",
                    label=f"field 2 (MSE {mse_b[1]:.4f})")
            ax.set_title(f"After split: own units {own}, "
                         f"readout cos={cos:.2f}", fontsize=9)
            ax.set_ylim(-1.6, 1.6)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

            ax = fig.add_subplot(gs[1, 1])
            ax.plot(losses_single, lw=0.8, c="tab:red",
                    label="single field (multivalued)")
            ax.axhline(mse_single, ls="--", c="tab:red", lw=0.8,
                       label="residual floor")
            ax.axhline(max(mse_b), ls="--", c="tab:green", lw=0.8,
                       label="after split (worst branch)")
            ax.set_yscale("log")
            ax.set_xlabel("epoch")
            ax.set_ylabel("MSE")
            ax.set_title("Loss floor of the single field", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, which="both")

            ax = fig.add_subplot(gs[1, 2])
            ax.axis("off")
            txt = "\n".join([f"{k}: {'PASS' if v else 'FAIL'}"
                             for k, v in verdicts.items()])
            ax.text(0.0, 0.9, "verdicts\n\n" + txt, fontsize=10, va="top",
                    family="monospace")
            fig.suptitle(f"PoC-M1: multivalued data y = ±sin(x) "
                         f"(H={H}, seed {seed})", fontsize=12)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
            savefig(fig, args.out_dir / "fig_multivalued.png")

    # ---------- 表 ----------
    seed0 = args.seed_list[0]
    r0 = all_results[seed0]
    lines = [f"**PoC-M1: multivalued data y = ±sin(x)** (H={H}, "
             f"T={args.num_samples}, epochs-task={args.epochs_task}, "
             f"seeds={args.seeds})", "",
             "| quantity | multivalued (single field) | control (single-valued) |",
             "|---|---|---|",
             f"| residual MSE | {r0['single']['mse']:.4f} "
             f"| {r0['control']['mse']:.5f} |",
             f"| \\|mean prediction\\| | {r0['single']['abs_mean_pred']:.4f} | — |",
             f"| bimodality score | **{r0['single']['bimodality']:.3f}** "
             f"| {r0['control']['bimodality']:.3f} |",
             f"| bimodality p-value | **{r0['single']['bimodal_p']:.3f}** "
             f"| {r0['control']['bimodal_p']:.3f} |",
             f"| residual RMS at P=1 | {r0['single']['irreducibility'][1]:.4f} "
             f"| {r0['control']['irreducibility'][1]:.4f} |",
             f"| residual RMS at P=32 | {r0['single']['irreducibility'][32]:.4f} "
             f"| {r0['control']['irreducibility'][32]:.4f} |", "",
             "after split:", "",
             f"- branch MSE: {[round(m, 5) for m in r0['split']['mse_branch']]}",
             f"- own units per branch: {r0['split']['own_units']}",
             f"- readout cosine similarity: "
             f"{r0['split']['readout_cos']:.3f}",
             f"- assignment accuracy (E-step): "
             f"{r0['split']['assign_acc']:.3f}", ""]
    for seed in args.seed_list:
        v = all_results[seed]["verdicts"]
        lines.append(f"- seed {seed}: "
                     + " / ".join(f"{k} {'**PASS**' if ok else '**FAIL**'}"
                                  for k, ok in v.items()))
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_multivalued.md", table)

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
