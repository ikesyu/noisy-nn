"""
consolidation_rho.py — 課題7: 連続的な動員度 rho の効用検証 (§13.5 -> §12.9.15)

§13.5 の「最初に確かめるべき 2 点」を検証する:

exp1 (--exp family): rho 依存の語彙の質。
  タスク sin(x) を獲得した語彙（凍結領域）を rho で部分動員し、読み出し
  のみの再利用（凍結 L2 チューニング曲線への ridge 回帰）の残差が rho で
  どう変わるかを測る。rho は sigma = rho*sigma0 と h = h0/rho を同時に
  動かすため、期待活性のバンプは「縮む」のでなく形が変わる（§13.5(3)）。
  比較: rho=1（現行の二値動員）/ 一様 rho の最良値 / ユニットごとの rho
  （座標探索; 推論時に実現可能な記述子は per-unit rho までなので、
  「複数 rho の重ね合わせ」は測らない）。
exp2 (--exp recruit): §12.9.8 知見4 の取りこぼし回収。
  sin(x) の語彙に対しプローブ相関 |cos 72°| = 0.31 の新タスク
  sin(x+0.4pi) は、二値プローブ（閾値 0.5）では全額自前を払う。
  部分動員（rho* を一様グリッドの ridge 残差最小で選ぶ / rho=score の
  ヒューリスティック）で再利用したときの自前ユニット数・MSE・
  タスク 1 の忘却（厳密ゼロが保たれるか）を二値と比較する。
exp3 (--exp union): 質量/個数の乖離（§13.5(4)）。
  案4 の union アニール（3 タスク共有場）を「段階的 rho」版で走らせる:
  ユニットの巻き戻しを全量（rho=1 へ）でなく直前ステップまでにし、
  終端で rho in (0,1] を許す。二値版（現行）と比較して、
  質量 sum(rho) と個数 |supp| がどう乖離するか、および支持個数への
  圧力（commit パス: 薄いユニットの kill を閉ループで試みる）で
  個数が回復するかを測る。

実行例:
  python tmp/consolidation_rho.py --quick
  python tmp/consolidation_rho.py --exp family,recruit --seeds 0,1,2
  python tmp/consolidation_rho.py --exp union --seeds 0
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
import consolidation_joint as cjoint  # noqa: E402
from consolidation_lib import (  # noqa: E402
    H_DEAD, TaskCtx, alive, any_over_tol, freeze_masks, joint_checkpoint,
    joint_restore, joint_round, kill_everywhere, probe_tuning_corr,
    predict, region_snapshot, region_drift, unit_score, zero_cross_columns)
from consolidation_multiscale import SCALES  # noqa: E402

RHO_GRID = [1.0, 0.8, 0.6, 0.45, 0.3, 0.2]


# ---------------------------------------------------------------- field / fit
def set_field_rho_map(net, own: dict, voc_rho: dict, sigma0, h0):
    """own は全量 (rho=1)、voc_rho = {l: {k: rho}} は per-unit で部分動員。"""
    H = net.sigma_vecs[0].shape[0]
    for l in (0, 1):
        sig = torch.zeros(H)
        hv = torch.full((H,), H_DEAD)
        for k in own[l]:
            sig[k] = sigma0
            hv[k] = h0
        for k, rho in voc_rho[l].items():
            sig[k] = rho * sigma0
            hv[k] = h0 / rho
        net.sigma_vecs[l] = sig.to(net.sigma_vecs[l].device)
        net.h_vecs[l] = hv.to(net.h_vecs[l].device)


def vocab_features(net, x, voc_rho: dict, sigma0, h0, passes=4):
    """語彙のみを voc_rho で動員し、L2 語彙ユニットの zbar を特徴として返す。"""
    keep = ([s.clone() for s in net.sigma_vecs],
            [h.clone() for h in net.h_vecs])
    set_field_rho_map(net, {0: [], 1: []}, voc_rho, sigma0, h0)
    _, zbar = poc.collect_stats(net, x, passes=passes)
    net.sigma_vecs, net.h_vecs = keep
    cols = sorted(voc_rho[1])
    return zbar[1][:, cols].cpu().numpy()


def ridge_res(Z, y, lam=1e-3):
    """切片つき ridge 回帰の相対残差 RMS（std(y) 比）。"""
    Zc = Z - Z.mean(axis=0)
    yc = y - y.mean()
    A = Zc.T @ Zc + lam * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Zc.T @ yc)
    res = yc - Zc @ w
    return float(np.sqrt((res ** 2).mean()) / (yc.std() + 1e-12))


def uniform_map(voc, rho):
    return {l: {k: rho for k in voc[l]} for l in (0, 1)}


# ------------------------------------------------------------- exp1: family
def exp_family(args, device, x, x_raw):
    """語彙の再利用の質（ridge 残差）を rho の関数として測る。"""
    sigma0, h0 = float(args.sigma), float(args.crossing_h)
    t1 = [("sin(x)", np.sin(x_raw).astype(np.float32))]
    targets = {
        "sin(x+0.4pi)": np.sin(x_raw + 0.4 * np.pi).astype(np.float32),
        "sin(x+0.8pi)": np.sin(x_raw + 0.8 * np.pi).astype(np.float32),
        "0.7sin(x)-0.5": (0.7 * np.sin(x_raw) - 0.5).astype(np.float32),
        "sin(2x)": np.sin(2 * x_raw).astype(np.float32)}
    rows = []
    for seed in args.seed_list:
        print(f"\n########## family / seed {seed} ##########")
        out = csoft.run_sequence("hard", seed, args, device, t1, x, x_raw)
        net = out["net"]
        voc = out["registry"][0]["region"]
        print(f"  vocab: L1 {len(voc[0])} + L2 {len(voc[1])}")
        for name, tgt in targets.items():
            # (a) 一様 rho スイープ
            res_u = {}
            for rho in RHO_GRID:
                Z = vocab_features(net, x, uniform_map(voc, rho), sigma0,
                                   h0, passes=args.stat_passes)
                res_u[rho] = ridge_res(Z, tgt)
            best_rho = min(res_u, key=res_u.get)
            # (b) per-unit rho（L2 のみ座標探索、L1 は最良一様値で固定）
            vr = uniform_map(voc, best_rho)
            best_pu = res_u[best_rho]
            for _ in range(2):
                for k in voc[1]:
                    for rho in RHO_GRID:
                        if abs(vr[1][k] - rho) < 1e-9:
                            continue
                        old = vr[1][k]
                        vr[1][k] = rho
                        r = ridge_res(vocab_features(
                            net, x, vr, sigma0, h0,
                            passes=args.stat_passes), tgt)
                        if r < best_pu - 1e-4:
                            best_pu = r
                        else:
                            vr[1][k] = old
            rows.append({"seed": seed, "target": name,
                         "res_rho1": res_u[1.0], "best_rho": best_rho,
                         "res_best_uniform": res_u[best_rho],
                         "res_per_unit": best_pu,
                         "sweep": {str(r): res_u[r] for r in RHO_GRID}})
            print(f"  {name}: rho=1 {res_u[1.0]:.3f} | best uniform "
                  f"rho={best_rho} {res_u[best_rho]:.3f} | per-unit "
                  f"{best_pu:.3f}")
    lines = [f"**連続 rho — exp1 語彙の質（読み出しのみ再利用の相対残差 "
             f"RMS/std）** (vocab = sin(x) hard 獲得領域, H="
             f"{args.hidden_dim} multiscale, grid={RHO_GRID}, "
             f"seeds={args.seeds})", "",
             "| target | rho=1 | best uniform (rho) | per-unit rho |",
             "|---" * 4 + "|"]
    for name in targets:
        sel = [r for r in rows if r["target"] == name]
        r1 = np.mean([r["res_rho1"] for r in sel])
        bu = np.mean([r["res_best_uniform"] for r in sel])
        br = [r["best_rho"] for r in sel]
        pu = np.mean([r["res_per_unit"] for r in sel])
        lines.append(f"| {name} | {r1:.3f} | {bu:.3f} ({br}) | {pu:.3f} |")
    return rows, "\n".join(lines) + "\n"


# ------------------------------------------------------------ exp2: recruit
def eval_rho(net, x, desc, passes=8):
    """rho つき記述子で場を設定し予測（forward 平均）を返す。"""
    keep = ([s.clone() for s in net.sigma_vecs],
            [h.clone() for h in net.h_vecs])
    w_keep = (net.fcs[2].weight.data.clone(), net.fcs[2].bias.data.clone())
    set_field_rho_map(net, desc["region"], desc["voc_rho"], desc["sigma0"],
                      desc["h0"])
    net.fcs[2].weight.data.copy_(desc["wout"])
    net.fcs[2].bias.data.copy_(desc["b_out"])
    preds = [predict(net, x) for _ in range(passes)]
    net.sigma_vecs, net.h_vecs = keep
    net.fcs[2].weight.data.copy_(w_keep[0])
    net.fcs[2].bias.data.copy_(w_keep[1])
    return np.mean(preds, axis=0)


def acquire_with_rho(net, x, target, past, voc_rho, args, device):
    """空き + 部分動員語彙で学習し、自前を anneal-until-stop で整理する。"""
    H = args.hidden_dim
    sigma0, h0 = float(args.sigma), float(args.crossing_h)
    free = {l: [k for k in range(H) if k not in past[l]] for l in (0, 1)}
    net.fcs[2].weight.data.zero_()
    net.fcs[2].bias.data.zero_()
    zero_cross_columns(net, past)
    set_field_rho_map(net, free, voc_rho, sigma0, h0)
    masks = freeze_masks(H, past, device)
    t = torch.tensor(target, device=device).unsqueeze(1)
    trainer = poc.CovJacTrainer(net, x, t, lr=args.pre_lr, opt=args.opt,
                                jac_ema=args.jac_ema)
    trainer.grad_masks = masks
    for e in range(args.epochs_task):
        loss = trainer.step()
        if e % max(1, args.epochs_task // 4) == 0:
            print(f"    [learn] epoch {e:5d} mse={loss:.5f}")
    trainer.close()
    base = float(np.mean(trainer.losses[-min(100, len(trainer.losses)):]))
    tol = base * args.drift_mult + args.drift_abs
    vocab = {l: sorted(voc_rho[l]) for l in (0, 1)}
    csoft.anneal_until_stop(net, x, target, tol, args, eligible=free,
                            grad_masks=masks, vocab=vocab)
    region = {l: [k for k in free[l] if float(net.sigma_vecs[l][k]) > 0]
              for l in (0, 1)}
    desc = {"region": region, "voc_rho": voc_rho, "sigma0": sigma0,
            "h0": h0, "wout": net.fcs[2].weight.data.clone(),
            "b_out": net.fcs[2].bias.data.clone()}
    mse = float(np.mean((eval_rho(net, x, desc) - target) ** 2))
    return desc, mse, region


def exp_recruit(args, device, x, x_raw):
    """知見4 シナリオ: プローブ相関 0.31 のタスクを部分動員で回収する。"""
    sigma0, h0 = float(args.sigma), float(args.crossing_h)
    t1 = ("sin(x)", np.sin(x_raw).astype(np.float32))
    t2 = ("sin(x+0.4pi)", np.sin(x_raw + 0.4 * np.pi).astype(np.float32))
    rows = []
    for seed in args.seed_list:
        print(f"\n########## recruit / seed {seed} ##########")
        # --- arm binary: 現行の案3（プローブ閾値 0.5 の二値動員） ---
        out_b = crc.run_sequence_recruit(seed, args, device, [t1, t2], x,
                                         cleanup_thr=args.mse_star)
        tb = out_b["tasks"][1]
        own_b = tb["n_own"]["0"] + tb["n_own"]["1"]
        # --- arm rho: 同じ土台から部分動員で獲得 ---
        out1 = csoft.run_sequence("hard", seed, args, device, [t1], x, x_raw)
        net = out1["net"]
        r1 = out1["registry"][0]
        voc = r1["region"]
        snap1 = region_snapshot(net, r1["region"])
        scores = probe_tuning_corr(net, x, t2[1], voc, sigma0, h0,
                                   passes=args.stat_passes)
        corr_max = max(scores.values()) if scores else 0.0
        # rho* を一様グリッドの ridge 残差最小で選ぶ（forward 統計のみ）
        res_u = {rho: ridge_res(vocab_features(net, x,
                                               uniform_map(voc, rho),
                                               sigma0, h0,
                                               passes=args.stat_passes),
                                t2[1]) for rho in RHO_GRID}
        rho_star = min(res_u, key=res_u.get)
        print(f"  probe corr max = {corr_max:.2f} (< 0.5 -> 二値では棄却), "
              f"rho* = {rho_star} (残差 {res_u[rho_star]:.3f} vs "
              f"rho=1 {res_u[1.0]:.3f})")
        desc, mse_r, region_r = acquire_with_rho(
            net, x, t2[1], past=voc, voc_rho=uniform_map(voc, rho_star),
            args=args, device=device)
        own_r = len(region_r[0]) + len(region_r[1])
        drift1 = region_drift(net, r1["region"], snap1)
        rows.append({"seed": seed, "own_binary": own_b,
                     "mse_binary": tb["mse_final"], "own_rho": own_r,
                     "mse_rho": mse_r, "rho_star": rho_star,
                     "corr_max": corr_max, "drift_t1": drift1,
                     "res_sweep": {str(r): res_u[r] for r in RHO_GRID}})
        print(f"  binary: own {own_b} MSE {tb['mse_final']:.4f} | "
              f"rho: own {own_r} MSE {mse_r:.4f} | drift(t1) "
              f"{drift1:.2e}")
    lines = [f"**連続 rho — exp2 取りこぼし回収（§12.9.8 知見4 の再現）** "
             f"(t1=sin(x) -> t2=sin(x+0.4pi), プローブ相関 ~0.31 < 閾値 "
             f"0.5, thr={args.mse_star}, seeds={args.seeds})", "",
             "| seed | binary own | binary MSE | rho* | rho own | rho MSE "
             "| drift(t1) |",
             "|---" * 7 + "|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['own_binary']} "
                     f"| {r['mse_binary']:.4f} | {r['rho_star']} "
                     f"| {r['own_rho']} | {r['mse_rho']:.4f} "
                     f"| {r['drift_t1']:.2e} |")
    return rows, "\n".join(lines) + "\n"


# -------------------------------------------------------------- exp3: union
def graded_union_anneal(net, x, ctxs, args, commit_floor=None):
    """union アニールの段階的 rho 版: 巻き戻しを直前ステップまでにする。

    ユニットは「これ以上下げると多目的許容を破る」rho で自然に停止し、
    終端で rho in (0,1] を持つ（質量最小化のみ）。commit_floor を与えると
    第二パスで rho < floor のユニットの完全 kill を閉ループで試み、
    失敗したら元の rho へ戻す（支持個数への圧力）。
    """
    sigma0 = float(args.sigma)
    rho = {l: {k: 1.0 for k in alive(net, l)} for l in (0, 1)}
    settled = {0: set(), 1: set()}
    removed = {0: 0, 1: 0}

    def run_block():
        for _ in range(args.epochs_per_step):
            joint_round(net, ctxs)

    def decay_step(l, k):
        """1 段の rho 減衰 + 閉ループ。戻り値: 'ok'|'fail'|'snapped'。"""
        ck = joint_checkpoint(net, ctxs)
        net.sigma_vecs[l][k] *= args.anneal_alpha
        if l >= 1:
            net.h_vecs[l][k] = net.h_vecs[l][k] / args.anneal_alpha
        run_block()
        h = 0
        while any_over_tol(ctxs) and h < args.max_holds:
            run_block()
            h += 1
        if any_over_tol(ctxs):
            joint_restore(net, ctxs, ck)
            return "fail"
        rho[l][k] *= args.anneal_alpha
        act = float(ctxs[-1].trainer.cap.z[l][:, :, k].mean())
        if act < args.snap_act:
            kill_everywhere(net, ctxs, l, k)
            rec = 0
            while any_over_tol(ctxs) and rec < args.stop_recovery:
                run_block()
                rec += 1
            if any_over_tol(ctxs):
                joint_restore(net, ctxs, ck)
                rho[l][k] /= args.anneal_alpha
                return "fail"
            return "snapped"
        return "ok"

    # --- パス 1: 貪欲 min S_k、per-unit 部分巻き戻し ---
    while True:
        ctxs[0].activate(net)
        _, zbar = poc.collect_stats(net, x, passes=args.stat_passes)
        scored = None
        for l in (0, 1):
            rows_ = alive(net, 1)
            basis_all = alive(net, l)
            for k in basis_all:
                if k in settled[l]:
                    continue
                if l == 1:
                    w2 = sum(float((ctx.wout[:, k] ** 2).sum())
                             for ctx in ctxs)
                else:
                    w2 = (float((net.fcs[1].weight.data[rows_][:, k] ** 2)
                                .sum()) if rows_ else 0.0)
                S = unit_score(zbar[l], k, [j for j in basis_all if j != k],
                               w2, args.ridge)
                if S is not None and (scored is None or S < scored[0]):
                    scored = (S, l, k)
        if scored is None:
            break
        _, l, k = scored
        while True:
            st = decay_step(l, k)
            if st == "fail":
                settled[l].add(k)
                break
            if st == "snapped":
                removed[l] += 1
                break

    # --- パス 2 (commit): 薄いユニットに個数圧力 ---
    committed = {0: 0, 1: 0}
    if commit_floor is not None:
        dims = [(l, k) for l in (0, 1) for k in alive(net, l)
                if rho[l][k] < commit_floor]
        dims.sort(key=lambda lk: rho[lk[0]][lk[1]])
        for l, k in dims:
            ck = joint_checkpoint(net, ctxs)
            old_sig = float(net.sigma_vecs[l][k])
            old_h = float(net.h_vecs[l][k])
            kill_everywhere(net, ctxs, l, k)
            # 生きているユニットの突然の kill なので、EMA を kill 後の
            # 損失で更新してから判定する（stale-EMA の誤受理を防ぐ）
            run_block()
            rec = 0
            while any_over_tol(ctxs) and rec < args.stop_recovery:
                run_block()
                rec += 1
            if any_over_tol(ctxs):
                joint_restore(net, ctxs, ck)
                net.sigma_vecs[l][k] = old_sig
                net.h_vecs[l][k] = old_h
            else:
                committed[l] += 1
                removed[l] += 1
    return rho, removed, committed


def graded_transit_anneal(net, x, ctxs, args, commit_floor=None):
    """段階的 rho 版その 2: binary と同じ「過渡超過の通過」を許す変種。

    ユニットごとに anneal_unit と同じ外側ループ（hold を使い切っても次の
    減衰ステップへ進む）を回し、各ステップ末尾で許容内なら
    その状態を last_good として記録する。経路の終端（活性 < snap_act か
    max_steps）で完全 kill を試み、回復猶予後も超過なら全量巻き戻しでは
    なく last_good（最後に許容内だった途中 rho）へ戻す —
    「捨てずに低 rho で残す」(§13.5(3)) の実装。
    """
    rho = {l: {k: 1.0 for k in alive(net, l)} for l in (0, 1)}
    settled = {0: set(), 1: set()}
    removed = {0: 0, 1: 0}

    def run_block():
        for _ in range(args.epochs_per_step):
            joint_round(net, ctxs)

    while True:
        ctxs[0].activate(net)
        _, zbar = poc.collect_stats(net, x, passes=args.stat_passes)
        scored = None
        for l in (0, 1):
            rows_ = alive(net, 1)
            basis_all = alive(net, l)
            for k in basis_all:
                if k in settled[l]:
                    continue
                if l == 1:
                    w2 = sum(float((ctx.wout[:, k] ** 2).sum())
                             for ctx in ctxs)
                else:
                    w2 = (float((net.fcs[1].weight.data[rows_][:, k] ** 2)
                                .sum()) if rows_ else 0.0)
                S = unit_score(zbar[l], k, [j for j in basis_all if j != k],
                               w2, args.ridge)
                if S is not None and (scored is None or S < scored[0]):
                    scored = (S, l, k)
        if scored is None:
            break
        _, l, k = scored
        ck_full = joint_checkpoint(net, ctxs)
        rho_full = rho[l][k]
        last_good = None
        steps = 0
        while True:
            net.sigma_vecs[l][k] *= args.anneal_alpha
            if l >= 1:
                net.h_vecs[l][k] = net.h_vecs[l][k] / args.anneal_alpha
            rho[l][k] *= args.anneal_alpha
            run_block()
            h = 0
            while any_over_tol(ctxs) and h < args.max_holds:
                run_block()
                h += 1
            steps += 1
            if not any_over_tol(ctxs):
                last_good = (joint_checkpoint(net, ctxs), rho[l][k])
            act = float(ctxs[-1].trainer.cap.z[l][:, :, k].mean())
            if act < args.snap_act or steps >= args.max_anneal_steps:
                break
        kill_everywhere(net, ctxs, l, k)
        rec = 0
        while any_over_tol(ctxs) and rec < args.stop_recovery:
            run_block()
            rec += 1
        if not any_over_tol(ctxs):
            removed[l] += 1
            rho[l][k] = 0.0
        elif last_good is not None:
            joint_restore(net, ctxs, last_good[0])
            rho[l][k] = last_good[1]
            settled[l].add(k)
        else:
            joint_restore(net, ctxs, ck_full)
            rho[l][k] = rho_full
            settled[l].add(k)

    committed = {0: 0, 1: 0}
    if commit_floor is not None:
        dims = [(l, k) for l in (0, 1) for k in alive(net, l)
                if rho[l][k] < commit_floor]
        dims.sort(key=lambda lk: rho[lk[0]][lk[1]])
        for l, k in dims:
            ck = joint_checkpoint(net, ctxs)
            old_sig = float(net.sigma_vecs[l][k])
            old_h = float(net.h_vecs[l][k])
            kill_everywhere(net, ctxs, l, k)
            # 生きているユニットの突然の kill なので、EMA を kill 後の
            # 損失で更新してから判定する（stale-EMA の誤受理を防ぐ）
            run_block()
            rec = 0
            while any_over_tol(ctxs) and rec < args.stop_recovery:
                run_block()
                rec += 1
            if any_over_tol(ctxs):
                joint_restore(net, ctxs, ck)
                net.sigma_vecs[l][k] = old_sig
                net.h_vecs[l][k] = old_h
            else:
                committed[l] += 1
                removed[l] += 1
                rho[l][k] = 0.0
    return rho, removed, committed


def field_stats(net, rho):
    n, mass, bands = 0, 0.0, {"dim": 0, "mid": 0, "full": 0}
    for l in (0, 1):
        for k in alive(net, l):
            n += 1
            r = rho[l][k] if rho is not None else 1.0
            mass += r
            bands["dim" if r <= 0.4 else "mid" if r <= 0.8
                  else "full"] += 1
    return n, mass, bands


def exp_union(args, device, x, x_raw):
    tasks = [("sin(x)", np.sin(x_raw).astype(np.float32)),
             ("cos(x)", np.cos(x_raw).astype(np.float32)),
             ("sin(x+pi)", np.sin(x_raw + np.pi).astype(np.float32))]
    H = args.hidden_dim
    rows = []
    for seed in args.seed_list:
        print(f"\n########## union / seed {seed} ##########")
        res_a = crc.run_sequence_recruit(seed, args, device, tasks, x)
        net, registry = res_a["net"], res_a["registry"]
        union_a = {l: sorted({k for r in registry for k in r["support"][l]})
                   for l in (0, 1)}
        sigma0, h0 = float(args.sigma), float(args.crossing_h)
        for l in (0, 1):
            sig = torch.zeros(H)
            hv = torch.full((H,), H_DEAD)
            for k in union_a[l]:
                sig[k] = sigma0
                hv[k] = h0
            net.sigma_vecs[l] = sig.to(device)
            net.h_vecs[l] = hv.to(device)
        ctxs = [TaskCtx.from_registry(net, x, r, args) for r in registry]
        for _ in range(args.settle_epochs):
            joint_round(net, ctxs)
        for ctx in ctxs:
            eq = float(np.mean(ctx.trainer.losses[-50:]))
            ctx.tol = max(ctx.tol, eq * args.drift_mult + args.drift_abs)
        n0 = len(union_a[0]) + len(union_a[1])
        ck0 = joint_checkpoint(net, ctxs)
        arms = {}
        # --- arm binary（現行） ---
        print(f"  ----- arm binary (union {n0}) -----")
        cjoint.union_anneal(net, x, ctxs, args)
        n, mass, bands = field_stats(net, None)
        arms["binary"] = {"count": n, "mass": mass, "bands": bands,
                          "mse": [ctx.eval(net, x)[0] for ctx in ctxs]}
        # --- arm graded（質量のみ; ステップ内許容超過で即 settle） ---
        joint_restore(net, ctxs, ck0)
        print(f"  ----- arm graded (union {n0}) -----")
        rho, rem, _ = graded_union_anneal(net, x, ctxs, args)
        n, mass, bands = field_stats(net, rho)
        arms["graded"] = {"count": n, "mass": mass, "bands": bands,
                          "mse": [ctx.eval(net, x)[0] for ctx in ctxs]}
        # --- arm transit（binary と同じ通過規則 + last_good へ部分巻き戻し） ---
        joint_restore(net, ctxs, ck0)
        print(f"  ----- arm transit (union {n0}) -----")
        rho, rem, _ = graded_transit_anneal(net, x, ctxs, args)
        n, mass, bands = field_stats(net, rho)
        arms["transit"] = {"count": n, "mass": mass, "bands": bands,
                           "mse": [ctx.eval(net, x)[0] for ctx in ctxs]}
        # --- arm transit+commit（個数圧力） ---
        joint_restore(net, ctxs, ck0)
        print(f"  ----- arm transit+commit (union {n0}) -----")
        rho, rem, com = graded_transit_anneal(net, x, ctxs, args,
                                              commit_floor=0.5)
        n, mass, bands = field_stats(net, rho)
        arms["transit+commit"] = {"count": n, "mass": mass, "bands": bands,
                                  "mse": [ctx.eval(net, x)[0]
                                          for ctx in ctxs],
                                  "committed": com[0] + com[1]}
        for nm, a in arms.items():
            print(f"  [{nm}] count {a['count']} mass {a['mass']:.1f} "
                  f"bands {a['bands']} max MSE {max(a['mse']):.4f}")
        rows.append({"seed": seed, "union_start": n0, "arms": arms,
                     "tols": [float(ctx.tol) for ctx in ctxs]})
    lines = [f"**連続 rho — exp3 union アニールの質量/個数** (3 tasks, "
             f"H={args.hidden_dim}, alpha={args.anneal_alpha}, "
             f"commit floor 0.5, seeds={args.seeds})", "",
             "| arm | 個数 |supp| | 質量 sum rho | dim(<=0.4) | "
             "mid | full | max MSE |",
             "|---" * 7 + "|"]
    r0 = rows[0]
    for nm, a in r0["arms"].items():
        b = a["bands"]
        lines.append(f"| {nm} | {a['count']} | {a['mass']:.1f} "
                     f"| {b['dim']} | {b['mid']} | {b['full']} "
                     f"| {max(a['mse']):.4f} |")
    lines.append("")
    lines.append(f"(開始 union = {r0['union_start']}; "
                 f"tol = {['%.4f' % t for t in r0['tols']]})")
    return rows, "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="continuous mobilisation dial "
                                            "rho: §13.5 PoC")
    poc.add_poc_args(p)
    p.add_argument("--epochs-task", type=int, default=600)
    p.add_argument("--stop-recovery", type=int, default=8)
    p.add_argument("--rho-init", type=float, default=1.0)
    p.add_argument("--recruit-thresh", type=float, default=0.5)
    p.add_argument("--settle-epochs", type=int, default=100)
    p.add_argument("--layer-fails", type=int, default=3)
    p.add_argument("--use-eps", type=float, default=0.02)
    p.add_argument("--mse-star", type=float, default=0.05)
    p.add_argument("--exp", type=str, default="family,recruit,union")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/consolidation_rho")
    args.cleanup = True
    args.l1_scales = SCALES
    if args.quick:
        args.epochs_task = 120
        args.epochs_per_step = 5
        args.stop_recovery = 3
        args.settle_epochs = 20

    device = torch.device(args.device)
    N = 128
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, N, dtype=np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)

    exps = [e.strip() for e in args.exp.split(",")]
    results, tables = {}, []
    if "family" in exps:
        rows, table = exp_family(args, device, x, x_raw)
        results["family"] = rows
        tables.append(table)
        print("\n" + table)
    if "recruit" in exps:
        rows, table = exp_recruit(args, device, x, x_raw)
        results["recruit"] = rows
        tables.append(table)
        print("\n" + table)
    if "union" in exps:
        rows, table = exp_union(args, device, x, x_raw)
        results["union"] = rows
        tables.append(table)
        print("\n" + table)

    tag = "_".join(exps)
    write_text(args.out_dir / f"table_rho_{tag}.md", "\n\n".join(tables))

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

    save_json(args.out_dir / f"results_{tag}.json",
              {"config": fncl.config_dict(args), "results": clean(results)})


if __name__ == "__main__":
    main()
