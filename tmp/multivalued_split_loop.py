"""
multivalued_split_loop.py — PoC-M3: 分岐数未知・非等頻度の多価データからの
場の自己増殖ループ (docs/idea_multivalued.md §8 PoC-M3)

M1/M2 は分岐数 2 を既知として分割した。本 PoC は**分岐数を知らずに**、
場を 1 つずつ生やすループでデータに個数を決めさせる:

  データ: 交差しない 3 分岐（分岐同一性の非識別性問題を切り離すため）
      b0 = sin(x)               重み w0（多数派）
      b1 = 1.5 + 0.5 sin(2x)    重み w1
      b2 = -1.8 - 0.5 sin(x)    重み w2（少数派）
    最小分離 0.3（b0-b2, x=-pi/2 近傍）。設定 main = (5,3,2)/10、
    rare = (6,3,1)/10（10% の稀分岐）。

  ループ（peel 型; §5 の「未説明点の集積 -> 分割」の実装形）:
    while 未説明率 >= min-gain:
      1. 新しい場を生やす。学習対象は**未説明点のみ**。
         - warm 期: 目標 = 各 x の未説明候補の重み付き平均
           （多数派分岐へ寄る保守的バイアス = §9.1 の対処）
         - WTA 期: 目標 = 各 x で現在の予測に最も近い未説明候補
           （サンプルレベルの hard-EM。5 epoch ごとに更新）
      2. 既存場は凍結語彙として読み出し共有（案2 の規約）、
         anneal-until-stop で整理。
      3. 受理判定（MDL）: 新規説明点（|r| < eps-point）の重み付き割合が
         min-gain 未満なら場を棄却（領域を掃除して停止）。
    最大 max-fields で安全停止。

検証項目:
  V-a 個数の自動決定: 両設定で場がちょうど 3 つ生える
  V-b 全分岐の説明: 停止時の未説明率 < 5% かつ全分岐の MSE < 0.05
  V-c 稀分岐の回収: rare 設定で 10% 分岐が自場を得て MSE < 0.05
  V-d 再利用の構造: b2 (= -0.5 sin x - 1.8, b0 の張る空間内) の場の自前
      ユニットが小さい（読み出し再利用）。b1 (sin 2x, 新部分空間) は中程度

生成物 (out/multivalued_split_loop/):
  fig_splitloop.png / table_splitloop.md / results.json

実行例:
  python tmp/multivalued_split_loop.py --quick
  python tmp/multivalued_split_loop.py --seeds 0,1,2
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
    eval_with_descriptor, freeze_masks, kill_unit, predict, set_field,
    zero_cross_columns)
from consolidation_multiscale import SCALES  # noqa: E402


# ============================================================
# 3 分岐の多価データ（交差なし・非等頻度）
# ============================================================
def make_branches(x_raw):
    b0 = np.sin(x_raw).astype(np.float32)
    b1 = (1.5 + 0.5 * np.sin(2.0 * x_raw)).astype(np.float32)
    b2 = (-1.8 - 0.5 * np.sin(x_raw)).astype(np.float32)
    return [b0, b1, b2]


# ============================================================
# 目標の構成（warm = 重み付き平均 / WTA = 最近傍候補）
# ============================================================
def warm_target(ys, weights, unexplained, pred):
    """各 x で未説明候補の重み付き平均。候補が無い x は現在の予測（無勾配）."""
    N = ys[0].shape[0]
    t = pred.copy()
    for n in range(N):
        num, den = 0.0, 0.0
        for b in range(len(ys)):
            if unexplained[b][n]:
                num += weights[b] * ys[b][n]
                den += weights[b]
        if den > 0:
            t[n] = num / den
    return t.astype(np.float32)


def wta_target(ys, unexplained, pred):
    """各 x で現在の予測に最も近い未説明候補（sample-level hard-EM）."""
    N = ys[0].shape[0]
    t = pred.copy()
    for n in range(N):
        best, bd = None, None
        for b in range(len(ys)):
            if unexplained[b][n]:
                d = abs(ys[b][n] - pred[n])
                if bd is None or d < bd:
                    bd, best = d, ys[b][n]
        if best is not None:
            t[n] = best
    return t.astype(np.float32)


# ============================================================
# 1 つの場を生やす（warm -> WTA -> anneal-until-stop）
# ============================================================
def grow_field(net, x, ys, weights, unexplained, free, past, args, device):
    H = args.hidden_dim
    share = bool(past[0] or past[1])
    net.fcs[2].weight.data.zero_()
    net.fcs[2].bias.data.zero_()
    if share:
        zero_cross_columns(net, past)
        # 二層共有 (§12.9.9): 先行する場が抱え込んだ L1 基底（特に細スケール）
        # を後続の場の L2 が読めるようにする。忘却は厳密ゼロのまま。
        masks = freeze_masks(H, past, device, share_l1=True)
        vocab = {l: sorted(past[l]) for l in (0, 1)}
    else:
        masks, vocab = None, {0: [], 1: []}
    mobil = {l: sorted(set(free[l]) | set(vocab[l])) for l in (0, 1)}
    set_field(net, mobil, args.sigma, args.crossing_h)

    pred = predict(net, x, passes=4)
    t_np = warm_target(ys, weights, unexplained, pred)
    tr = poc.CovJacTrainer(net, x,
                           torch.tensor(t_np, device=device).unsqueeze(1),
                           lr=args.pre_lr, opt=args.opt, jac_ema=args.jac_ema)
    tr.grad_masks = masks
    for e in range(args.epochs_task):
        if e % args.wta_every == 0:
            pred = predict(net, x, passes=4)
            t_np = (warm_target(ys, weights, unexplained, pred)
                    if e < args.warm_epochs
                    else wta_target(ys, unexplained, pred))
            tr.t = torch.tensor(t_np, device=device).unsqueeze(1)
        loss = tr.step()
        if e % max(1, args.epochs_task // 3) == 0:
            phase = "warm" if e < args.warm_epochs else "wta"
            print(f"      [{phase}] epoch {e:5d} mse={loss:.5f}")
    tr.close()
    # 精練フェーズ: WTA の目標更新は 5 epoch ごとに目標を揺らし学習予算を
    # 浪費する（M2 の同難度の分岐は固定目標で 0.007 に達していた）。
    # 目標を確定させてから固定で磨く = hard-EM の M-step を正しく行う。
    tr2 = poc.CovJacTrainer(net, x,
                            torch.tensor(t_np, device=device).unsqueeze(1),
                            lr=args.pre_lr, opt=args.opt,
                            jac_ema=args.jac_ema)
    tr2.grad_masks = masks
    for e in range(args.refine_epochs):
        loss = tr2.step()
        if e % max(1, args.refine_epochs // 2) == 0:
            print(f"      [refine] epoch {e:5d} mse={loss:.5f}")
    tr2.close()
    base = float(np.mean(tr2.losses[-min(100, len(tr2.losses)):]))
    tol = base * args.drift_mult + args.drift_abs
    _, holds = csoft.anneal_until_stop(net, x, t_np, tol, args,
                                       eligible=free, grad_masks=masks,
                                       vocab=vocab, share_l1=True)
    region = {l: [k for k in free[l] if float(net.sigma_vecs[l][k]) > 0]
              for l in (0, 1)}
    r = {"name": "field", "target": t_np, "region": region,
         "support": {l: sorted(region[l] + vocab[l]) for l in (0, 1)},
         "sigma0": float(args.sigma), "h0": float(args.crossing_h),
         "wout": net.fcs[2].weight.data.clone(),
         "b_out": net.fcs[2].bias.data.clone(), "tol": tol, "holds": holds}
    return r, region, t_np


# ============================================================
# 自己増殖ループ
# ============================================================
def split_loop(net, x, x_raw, ys, weights, args, device,
               past=None, free=None):
    """場の自己増殖ループ。past/free を渡すと既存ネットワークの状態から
    続きとして走る（オンライン統合 PoC-M4 用）。返り値に更新済みの
    past/free を含むので、呼び出し側は後続タスクへそのまま続けられる."""
    B = len(ys)
    N = ys[0].shape[0]
    total_w = float(sum(weights)) * N
    unexplained = [np.ones(N, dtype=bool) for _ in range(B)]
    free = ({0: list(range(args.hidden_dim)),
             1: list(range(args.hidden_dim))}
            if free is None else {l: list(free[l]) for l in (0, 1)})
    past = ({0: [], 1: []} if past is None
            else {l: list(past[l]) for l in (0, 1)})
    fields, history = [], []

    def frac_unexplained():
        return float(sum(weights[b] * unexplained[b].sum()
                         for b in range(B)) / total_w)

    while len(fields) < args.max_fields:
        fu = frac_unexplained()
        history.append(fu)
        print(f"    [loop] fields={len(fields)} unexplained={fu:.3f}")
        if fu < args.min_gain:
            print("    [stop] 未説明率が min-gain 未満 -> 停止")
            break
        print(f"    [field {len(fields) + 1}]")
        r, region, t_np = grow_field(net, x, ys, weights, unexplained,
                                     free, past, args, device)
        pred = eval_with_descriptor(net, x, r, passes=16)
        newly = [(np.abs(ys[b] - pred) < args.eps_point) & unexplained[b]
                 for b in range(B)]
        gain = float(sum(weights[b] * newly[b].sum() for b in range(B))
                     / total_w)
        by_branch = [float(weights[b] * newly[b].sum() / total_w)
                     for b in range(B)]
        dom = int(np.argmax(by_branch))
        purity = by_branch[dom] / max(gain, 1e-9)
        own = len(region[0]) + len(region[1])
        print(f"      gain={gain:.3f} (branch split {np.round(by_branch, 3)}"
              f", dominant=b{dom}, purity={purity:.2f}); own units={own}")
        if gain < args.min_gain:
            # MDL 棄却: 説明の増分がコストを正当化しない -> 掃除して停止
            for l in (0, 1):
                for k in region[l]:
                    kill_unit(net, l, k)
            print(f"    [stop] gain {gain:.3f} < {args.min_gain} -> "
                  f"場を棄却（掃除）して停止")
            break
        fields.append({"descriptor": r, "region": region, "own": own,
                       "gain": gain, "dominant_branch": dom,
                       "purity": purity, "pred": pred,
                       "holds": r["holds"]})
        for b in range(B):
            unexplained[b] &= ~newly[b]
        past = {l: past[l] + region[l] for l in (0, 1)}
        free = {l: [k for k in free[l] if k not in region[l]]
                for l in (0, 1)}
    history.append(frac_unexplained())
    return fields, history, unexplained, past, free


# ============================================================
# main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="PoC-M3: unknown branch count, "
                                            "field self-splitting loop")
    poc.add_poc_args(p)
    p.add_argument("--epochs-task", type=int, default=600)
    p.add_argument("--stop-recovery", type=int, default=8)
    p.add_argument("--warm-epochs", type=int, default=100,
                   help="重み付き平均目標で学ぶウォーム期")
    p.add_argument("--wta-every", type=int, default=5,
                   help="WTA 目標の更新間隔 (epochs)")
    p.add_argument("--refine-epochs", type=int, default=400,
                   help="WTA 確定後の固定目標での精練 epoch 数")
    p.add_argument("--eps-point", type=float, default=0.14,
                   help="点が説明されたとみなす |残差| 閾値"
                        "（分岐の最小分離 0.3 の半分未満であること)")
    p.add_argument("--min-gain", type=float, default=0.05,
                   help="場の受理に要する新規説明率 / ループ停止閾値")
    p.add_argument("--max-fields", type=int, default=5)
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/multivalued_split_loop")
    args.l1_scales = SCALES
    if args.quick:
        args.epochs_task = 150
        args.refine_epochs = 80
        args.warm_epochs = 40
        args.epochs_per_step = 5
        args.stop_recovery = 3

    device = torch.device(args.device)
    N = 128
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, N, dtype=np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    ys = make_branches(x_raw)
    B = len(ys)
    H = args.hidden_dim
    configs = {"main": (5, 3, 2), "rare": (6, 3, 1)}

    all_results = {}
    for seed in args.seed_list:
        res_seed = {}
        for cname, weights in configs.items():
            print(f"\n===== seed {seed} / config {cname} "
                  f"(weights {weights}) =====")
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = poc.build_net(H, args.sigma, args.crossing_h,
                                args.num_samples, device, scales=SCALES)
            fields, history, unexplained, _, _ = split_loop(
                net, x, x_raw, ys, weights, args, device)

            # ---- 評価 ----
            n_fields = len(fields)
            fu_final = history[-1]
            # 分岐ごとの MSE: その分岐を支配的に説明した場での誤差
            branch_field = {}          # b -> 最初の場（分岐 MSE 用）
            branch_fields = {}         # b -> その分岐を支配する場すべて
            for i, f in enumerate(fields):
                branch_field.setdefault(f["dominant_branch"], i)
                branch_fields.setdefault(f["dominant_branch"], set()).add(i)
            mse_branch = []
            for b in range(B):
                if b in branch_field:
                    pred = fields[branch_field[b]]["pred"]
                    mse_branch.append(float(np.mean((ys[b] - pred) ** 2)))
                else:
                    mse_branch.append(float("inf"))
            # E-step 割当精度（全点; 真の分岐 -> その場、の一致率）
            preds = np.stack([f["pred"] for f in fields], axis=0) \
                if fields else np.zeros((0, N))
            correct, tot = 0.0, 0.0
            for b in range(B):
                if not len(fields):
                    break
                a = np.argmin((preds - ys[b][None, :]) ** 2, axis=0)
                if b in branch_fields:
                    ok = np.isin(a, list(branch_fields[b]))
                    correct += weights[b] * ok.sum()
                tot += weights[b] * N
            acc = float(correct / max(tot, 1e-9))
            owns = [f["own"] for f in fields]
            purities = [f["purity"] for f in fields]
            print(f"\n  ===== [{cname}] fields={n_fields} "
                  f"unexplained={fu_final:.3f} "
                  f"branch MSE={[round(m, 4) for m in mse_branch]} "
                  f"own={owns} purity={[round(p, 2) for p in purities]} "
                  f"acc={acc:.3f} =====")
            res_seed[cname] = {
                "n_fields": n_fields, "history": history,
                "unexplained_final": fu_final,
                "mse_branch": mse_branch, "own_units": owns,
                "purity": purities, "assign_acc": acc,
                "dominant": [f["dominant_branch"] for f in fields],
                "gains": [f["gain"] for f in fields]}
            res_seed[cname]["_fields"] = fields  # 図用（json では除外）

        # ---- 判定 ----
        m, r_ = res_seed["main"], res_seed["rare"]
        va = (m["n_fields"] == B and r_["n_fields"] == B)
        vb = (m["unexplained_final"] < args.min_gain
              and max(m["mse_branch"]) < 0.05
              and r_["unexplained_final"] < args.min_gain)
        rare_b = 2                      # rare 設定で重み最小の分岐
        vc = (rare_b in r_["dominant"]
              and r_["mse_branch"][rare_b] < 0.05)
        # b2 は b0 の張る空間内 -> 自前小 / b1 は新部分空間 -> 中程度
        own_b2 = (m["own_units"][m["dominant"].index(2)]
                  if 2 in m["dominant"] else 999)
        own_b1 = (m["own_units"][m["dominant"].index(1)]
                  if 1 in m["dominant"] else 999)
        vd = own_b2 <= 5 and own_b2 < own_b1
        verdicts = {"Va_field_count": bool(va), "Vb_all_explained": bool(vb),
                    "Vc_rare_recovered": bool(vc),
                    "Vd_reuse_structure": bool(vd)}
        res_seed["verdicts"] = verdicts
        print(f"\n  ===== seed {seed} verdicts: {verdicts} =====")
        all_results[seed] = res_seed

        # ---- 図（先頭 seed / main） ----
        if seed == args.seed_list[0]:
            fields = res_seed["main"]["_fields"]
            fig = plt.figure(figsize=(13, 4.8))
            gs = fig.add_gridspec(1, 3)
            ax = fig.add_subplot(gs[0, 0])
            for b, yb in enumerate(ys):
                ax.plot(x_raw, yb, "k--", lw=0.8)
            for i, f in enumerate(fields):
                ax.plot(x_raw, f["pred"], lw=1.5,
                        label=f"field {i + 1} -> b{f['dominant_branch']} "
                              f"(own {f['own']})")
            ax.set_title(f"{len(fields)} fields grown for "
                         f"{B} branches (unknown count)", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

            ax = fig.add_subplot(gs[0, 1])
            hist = res_seed["main"]["history"]
            ax.step(range(len(hist)), hist, where="post", lw=1.6,
                    c="tab:blue", label="main (5:3:2)")
            hist_r = res_seed["rare"]["history"]
            ax.step(range(len(hist_r)), hist_r, where="post", lw=1.6,
                    c="tab:red", label="rare (6:3:1)")
            ax.axhline(args.min_gain, ls="--", c="k", lw=0.9,
                       label=f"stop threshold {args.min_gain}")
            ax.set_xlabel("fields grown")
            ax.set_ylabel("unexplained fraction (weighted)")
            ax.set_title("The loop stops by itself", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            ax = fig.add_subplot(gs[0, 2])
            ax.axis("off")
            lines_v = [f"{k}: {'PASS' if v else 'FAIL'}"
                       for k, v in verdicts.items()]
            extra = (f"\n\nmain: own={res_seed['main']['own_units']}"
                     f"\n      purity="
                     f"{[round(p, 2) for p in res_seed['main']['purity']]}"
                     f"\n      acc={res_seed['main']['assign_acc']:.3f}"
                     f"\nrare: own={res_seed['rare']['own_units']}"
                     f"\n      acc={res_seed['rare']['assign_acc']:.3f}")
            ax.text(0.0, 0.95, "verdicts\n\n" + "\n".join(lines_v) + extra,
                    fontsize=10, va="top", family="monospace")
            fig.suptitle(f"PoC-M3: field self-splitting loop, branch count "
                         f"unknown (H={H}, seed {seed})", fontsize=12)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
            savefig(fig, args.out_dir / "fig_splitloop.png")

        for cname in configs:
            res_seed[cname].pop("_fields", None)

    # ---- 表 ----
    seed0 = args.seed_list[0]
    r0 = all_results[seed0]
    lines = [f"**PoC-M3: field self-splitting loop** (H={H}, "
             f"T={args.num_samples}, epochs-task={args.epochs_task}, "
             f"eps-point={args.eps_point}, min-gain={args.min_gain}, "
             f"seeds={args.seeds})", "",
             "branches: b0=sin(x), b1=1.5+0.5sin(2x), b2=-1.8-0.5sin(x) "
             "(min separation 0.3)", "",
             "| config | fields grown | unexplained at stop | branch MSE "
             "| own units | purity | assign acc |",
             "|---" * 7 + "|"]
    for cname in configs:
        c = r0[cname]
        lines.append(
            f"| {cname} | **{c['n_fields']}** "
            f"| {c['unexplained_final']:.3f} "
            f"| {[round(m, 4) for m in c['mse_branch']]} "
            f"| {c['own_units']} "
            f"| {[round(p, 2) for p in c['purity']]} "
            f"| {c['assign_acc']:.3f} |")
    lines.append("")
    for seed in args.seed_list:
        v = all_results[seed]["verdicts"]
        lines.append(f"- seed {seed}: "
                     + " / ".join(f"{k} {'**PASS**' if ok else '**FAIL**'}"
                                  for k, ok in v.items()))
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_splitloop.md", table)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()
                    if not str(k).startswith("_")}
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
