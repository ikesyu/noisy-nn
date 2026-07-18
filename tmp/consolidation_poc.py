"""
consolidation_poc.py — noise-field self-consolidation の PoC
(docs/idea_consolidation.md §12 の PoC-1/2/3)

学習済み NNN（sin 回帰, [1, H, H, 1], Gaussian SimpleNNNSample）のノイズ場
非ゼロ領域を各隠れ層 --remove-frac だけ縮小し、以下の 4+1 法を比較する。
ライブラリのモデルは h（交差しきい値）が σ と独立なので、「h 凍結・σ のみ
アニール」（idea_consolidation.md §4.2 の消滅経路 (a)）がそのまま実装できる。

  route_A        : 一括線形吸収 (§6)。期待活性のリッジ回帰 → 次層へ移植 →
                   σ_k = 0。毎削除後に S_k を再計算する貪欲法 (§6.5)。学習なし。
  route_A_budget : route_A + ノイズ予算移転 (§6.4)。層 0 の削除で失われる
                   ゆらぎ分散を次隠れ層の σ_j に付け替える。
  route_A_ft     : route_A の後に cov_jac で fine-tune（epoch 数は route_B と同じ）。
  route_B        : ノイズ・アニーリング自己コンソリデーション (§7)。冗長性
                   スコア S_k で候補を選び、σ_k を幾何アニール（h は凍結）し
                   ながら cov_jac 学習を継続。損失 EMA が許容を超えたら
                   アニールを保留して学習を追加（閉ループ）。credit の自動
                   消灯（Var(z_k)→0）に任せ、移送則・折り畳み・教師コピーは
                   使わない。σ_k が snap-frac 未満で σ_k = 0 へスナップ。
  abrupt_ft      : PoC-3 の対照。同数のユニットを初期 S_k ランキングで一括
                   σ = 0 にし、route_B と同 epoch 数だけ cov_jac で回復させる
                   （generic pruning + fine-tune の NNN 版）。
  route_B_rand   : (--with-random) route_B の削除順ランダム対照。

学習則は cov_jac（重みミラー + Kolen-Pollack 追跡 + KDE スロープ、Adam;
--opt sgd で HW 整合形）。事前学習も同じ cov_jac なので、パイプライン全体が
forward-only で完結する。ライブラリ本体（ConsolidableNNN / CovJacTrainer /
kill_unit / anneal_unit / collect_stats など）は consolidation_lib にあり、
本モジュールは互換のため再エクスポートした上で、実験固有の部分（回帰移植・
削除スコア・route A/B の貪欲スケジュール・図表・main）だけを持つ。

判定:
  V1 (consolidation) : route_B の最終タスク MSE <= 1.5 x 事前学習 MSE
  V2 (annealing)     : route_B の移行中最大 MSE < abrupt の削除直後 MSE、かつ
                       route_B 最終 MSE <= abrupt_ft 最終 MSE x 1.1
  V3 (budget)        : route_A_budget の E_func <= route_A の E_func

生成物 (out/consolidation_poc/):
  table_final.md          最終比較表 + 判定
  fig_deletion_curves.png 削除曲線（タスク MSE / E_func vs 削除数）
  fig_anneal_trace.png    route_B の損失軌跡（snap マーカー付き）vs abrupt
  fig_predictions.png     整理前後の予測曲線
  results.json

実行例:
  python tmp/consolidation_poc.py --quick      # 動作確認 (~1 分)
  python tmp/consolidation_poc.py              # 本実験 (CPU 数分〜十数分)
  python tmp/consolidation_poc.py --opt sgd --ft-lr 0.03   # HW 整合の学習則
  python tmp/consolidation_poc.py --with-random
"""
import argparse

import numpy as np
import torch
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")       # スクリプト実行はヘッドレス; import 時は変更しない
import matplotlib.pyplot as plt  # noqa: E402

import fncl_driver as fncl  # noqa: E402  (make_task / 保存ヘルパを再利用)
from fncl_driver import save_json, savefig, write_text  # noqa: E402
# ライブラリ本体は consolidation_lib に移した。既存の実験スクリプトが
# poc.kill_unit などを参照し続けられるよう、ここから再エクスポートする。
from consolidation_lib import (  # noqa: E402,F401
    H_DEAD, ConsolidableNNN, CovJacTrainer, anneal_unit, checkpoint,
    collect_stats, kill_unit, n_active, noise_budget, restore, ridge_fit)
from nnn.credit import EPS  # noqa: E402,F401


def build_net(H: int, sigma: float, h: float, t: int, device: torch.device,
              out_dim: int = 1):
    net = ConsolidableNNN(structure=[1, H, H, out_dim], std=sigma, h=h, t=t,
                          output_bias=True)
    # fncl_driver.build_model と同じ 1 次元 bump タイリング初期化
    centres = torch.linspace(-2.0, 2.0, H)
    mag = 0.8 + 0.4 * torch.rand(H)
    sign = torch.where(torch.rand(H) < 0.5, -1.0, 1.0)
    w1 = (mag * sign).unsqueeze(1)
    with torch.no_grad():
        net.fcs[0].weight.copy_(w1)
        net.fcs[0].bias.copy_(-(w1.squeeze(1)) * centres)
    net = net.to(device)
    net.sigma_vecs = [torch.full((H,), sigma, device=device) for _ in range(2)]
    net.h_vecs = [torch.full((H,), h, device=device) for _ in range(2)]
    return net


# ============================================================
# 回帰移植と削除スコア (§6)
# ============================================================
def score_layer(net, zbar, l: int, active: list, lam: float):
    """層 l の各 active ユニットの削除コスト S_k (§6.5/§7.5) と回帰結果."""
    Wn = net.fcs[l + 1].weight.data                       # [Ho, H]
    out = {}
    for k in active:
        rest = [j for j in active if j != k]
        a, c, resid = ridge_fit(zbar[l][:, rest], zbar[l][:, k], lam)
        S = (float((Wn[:, k] ** 2).sum())
             * float((resid ** 2).sum())
             / (float((zbar[l][:, k] ** 2).sum()) + EPS))
        out[k] = (S, a, c, rest, resid)
    return out


def absorb(net, z, l: int, k: int, a, c, rest, budget: bool):
    """§6.2 の移植 + σ_k = 0（budget=True なら §6.4 のノイズ予算移転も）."""
    W = net.fcs[l + 1].weight.data
    wk = W[:, k].clone()
    W[:, rest] += wk.unsqueeze(1) * a.unsqueeze(0)
    net.fcs[l + 1].bias.data += c * wk
    kill_unit(net, l, k)
    if budget and l + 1 < len(net.sigma_vecs):
        # 残差ゆらぎの per-sample 分散を次隠れ層の注入ノイズへ付け替える
        # （σ=0 で離脱済みのユニットは復活させない）
        rs = z[l][:, :, k] - torch.einsum("ntj,j->nt", z[l][:, :, rest], a) - c
        v_res = float(rs.var(dim=1, unbiased=False).mean())
        sig = net.sigma_vecs[l + 1]
        alive = (sig > 0).float()
        net.sigma_vecs[l + 1] = torch.sqrt(sig ** 2 + alive * (wk ** 2) * v_res)


# ============================================================
# 計測
# ============================================================
def eval_net(net, x, target, pred0, passes: int = 8):
    pred = fncl.predict(net, x, passes=passes)
    return {"task_mse": float(np.mean((pred - target) ** 2)),
            "e_func": float(np.mean((pred - pred0) ** 2))}


def out_std(net, x, passes: int = 32) -> float:
    """単発 forward の出力ゆらぎ（アンサンブル平均後の per-input std の平均）."""
    with torch.no_grad():
        ys = torch.stack([net(x) for _ in range(passes)], dim=0)  # [P, N, 1]
    return float(ys.std(dim=0, unbiased=False).mean())


# ============================================================
# 各法
# ============================================================
def run_route_A(net, x, target, pred0, quota: dict, args):
    """一括線形吸収の貪欲逐次削除（毎削除後に統計と S_k を再計算）."""
    active = {l: [i for i in range(args.hidden_dim)] for l in (0, 1)}
    curve = [dict(removed=0, **eval_net(net, x, target, pred0))]
    quota = dict(quota)
    removed = 0
    while any(quota[l] > 0 for l in quota):
        z, zbar = collect_stats(net, x, passes=args.stat_passes)
        best = None
        for l in (0, 1):
            if quota[l] <= 0:
                continue
            scores = score_layer(net, zbar, l, active[l], args.ridge)
            k = min(scores, key=lambda kk: scores[kk][0])
            if best is None or scores[k][0] < best[0]:
                best = (scores[k][0], l, k, scores[k])
        _, l, k, (S, a, c, rest, _) = best
        absorb(net, z, l, k, a, c, rest, budget=args._budget)
        active[l].remove(k)
        quota[l] -= 1
        removed += 1
        curve.append(dict(removed=removed, **eval_net(net, x, target, pred0)))
    return curve


def initial_ranking(net, x, quota: dict, args):
    """abrupt 対照用: 事前学習ネットの初期 S_k ランキング（再計算なし・一括）."""
    _, zbar = collect_stats(net, x, passes=args.stat_passes)
    chosen = []
    for l in (0, 1):
        scores = score_layer(net, zbar, l, list(range(args.hidden_dim)), args.ridge)
        order = sorted(scores, key=lambda kk: scores[kk][0])
        chosen += [(l, k) for k in order[:quota[l]]]
    return chosen


def run_abrupt(net, x, target, pred0, chosen, ft_epochs: int, args):
    """PoC-3 対照: 一括 σ=0 -> 同一学習則で fine-tune."""
    for l, k in chosen:
        kill_unit(net, l, k)
    spike = eval_net(net, x, target, pred0)               # 削除直後（回復前）
    trainer = CovJacTrainer(net, x, torch.tensor(target).unsqueeze(1).to(x.device),
                            lr=args.ft_lr, opt=args.opt, jac_ema=args.jac_ema)
    trainer.run(ft_epochs)
    trainer.close()
    return spike, trainer.losses


def run_route_B(net, x, target, pred0, quota: dict, tol: float, args,
                random_order: bool = False, record=None, eligible: dict = None):
    """ノイズ・アニーリング自己コンソリデーション (§7.1).

    record(event, net, trainer, removed, anneal) を渡すと、開始時 ("start")、
    学習チャンクごと ("train"/"hold")、スナップごと ("snap") に呼ばれる
    （可視化用; anneal は現在アニール中の (layer, unit)）。
    eligible={l: [units]} を渡すと、アニール候補をその部分集合に限定する
    （逐次マルチタスク学習で現タスクの領域だけを整理する用途; §12.8）。
    """
    t_target = torch.tensor(target).to(x.device)
    if t_target.dim() == 1:
        t_target = t_target.unsqueeze(1)
    trainer = CovJacTrainer(net, x, t_target, lr=args.ft_lr, opt=args.opt,
                            jac_ema=args.jac_ema)
    rec = record if record is not None else (lambda *a: None)
    active = {l: (list(eligible[l]) if eligible is not None
                  else [i for i in range(args.hidden_dim)]) for l in (0, 1)}
    quota = dict(quota)
    sigma_init = float(args.sigma)
    curve = [dict(removed=0, **eval_net(net, x, target, pred0))]
    snaps, removed, holds_total = [], 0, 0
    rng = np.random.default_rng(12345)
    rec("start", net, trainer, removed, None)
    while any(quota[l] > 0 for l in quota):
        # --- 候補選択: 冗長性スコア S_k（forward 統計のみ; §7.5(ii)） ---
        cand_layers = [l for l in (0, 1) if quota[l] > 0]
        if random_order:
            l = int(rng.choice(cand_layers))
            k = int(rng.choice(active[l]))
        else:
            _, zbar = collect_stats(net, x, passes=args.stat_passes)
            best = None
            for l2 in cand_layers:
                scores = score_layer(net, zbar, l2, active[l2], args.ridge)
                k2 = min(scores, key=lambda kk: scores[kk][0])
                if best is None or scores[k2][0] < best[0]:
                    best = (scores[k2][0], l2, k2)
            _, l, k = best
        # --- 消滅経路のアニール + 閉ループ学習 (§7.1/§7.7; consolidation_lib) ---
        def run_block(kind):
            trainer.run(args.epochs_per_step)
            rec(kind, net, trainer, removed, (l, k))

        holds_total += anneal_unit(
            net, l, k, run_block,
            over_tol=lambda: trainer.ema() > tol,
            read_act=lambda: float(trainer.cap.z[l][:, :, k].mean()),
            alpha=args.anneal_alpha, max_holds=args.max_holds,
            snap_act=args.snap_act, max_steps=args.max_anneal_steps)
        # --- スナップ (§4.4): 厳密離脱（σ=0; 深層は h 番兵; 次層第 k 列 = 0） ---
        kill_unit(net, l, k)
        active[l].remove(k)
        quota[l] -= 1
        removed += 1
        snaps.append(len(trainer.losses))
        rec("snap", net, trainer, removed, (l, k))
        curve.append(dict(removed=removed, **eval_net(net, x, target, pred0)))
    trainer.close()
    return curve, trainer.losses, snaps, holds_total


# ============================================================
# 図・表
# ============================================================
def fig_deletion_curves(curves: dict, points: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, label in (("task_mse", "task MSE"), ("e_func", "E_func (vs pretrained)")):
        ax = axes[0] if key == "task_mse" else axes[1]
        for name, curve in curves.items():
            ax.plot([p["removed"] for p in curve], [max(p[key], 1e-8) for p in curve],
                    marker=".", lw=1.2, label=name)
        for name, (rem, m) in points.items():
            ax.scatter([rem], [max(m[key], 1e-8)], marker="*", s=90, zorder=5,
                       label=name)
        ax.set_yscale("log")
        ax.set_xlabel("units removed")
        ax.set_ylabel(key)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.suptitle("Consolidation deletion curves")
    fig.tight_layout()
    savefig(fig, path)


def fig_anneal_trace(losses_B, snaps, losses_ab, spike_mse, tol, base, path):
    fig = plt.figure(figsize=(9, 4.2))
    plt.plot(losses_B, lw=0.9, label="route_B (anneal + cov_jac)")
    xs = np.arange(len(losses_ab)) + 1
    plt.plot(xs, losses_ab, lw=0.9, label="abrupt_ft (zero-then-finetune)")
    plt.scatter([0], [spike_mse], marker="x", c="r", zorder=5,
                label="abrupt: right after zeroing")
    for i, s in enumerate(snaps):
        plt.axvline(s, color="gray", lw=0.4, alpha=0.5,
                    label="route_B snap" if i == 0 else None)
    plt.axhline(tol, ls="--", c="k", lw=0.8, label="drift tolerance")
    plt.axhline(base, ls=":", c="k", lw=0.8, label="pretrain baseline")
    plt.yscale("log")
    plt.xlabel("epoch (during consolidation)")
    plt.ylabel("eval MSE (log)")
    plt.title("route_B anneal trace vs abrupt control")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3, which="both")
    fig.tight_layout()
    savefig(fig, path)


def fig_predictions(x_raw, target, preds: dict, path):
    fig = plt.figure(figsize=(9, 4.2))
    plt.plot(x_raw, target, "k-", lw=1.0, label="target sin(x)")
    for name, pred in preds.items():
        plt.plot(x_raw, pred, lw=1.0, alpha=0.85, label=name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predictions before/after consolidation")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


# ============================================================
# main
# ============================================================
def add_poc_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """コンソリデーション PoC 固有の引数（可視化スクリプトと共有）."""
    fncl.add_common_args(p, epochs=1500, hidden_dim=32, num_samples=64, seeds="0")
    p.add_argument("--pre-lr", type=float, default=3e-3,
                   help="事前学習 (cov_jac) の lr")
    p.add_argument("--ft-lr", type=float, default=3e-3,
                   help="補償学習 / fine-tune の lr")
    p.add_argument("--opt", choices=("adam", "sgd"), default="adam",
                   help="cov_jac のオプティマイザ (sgd = HW 整合形)")
    p.add_argument("--remove-frac", type=float, default=0.5,
                   help="各隠れ層で削除するユニット比率")
    p.add_argument("--anneal-alpha", type=float, default=0.6,
                   help="σ_k の幾何アニール係数")
    p.add_argument("--epochs-per-step", type=int, default=10,
                   help="アニール 1 段あたりの学習 epoch 数")
    p.add_argument("--snap-frac", type=float, default=0.05,
                   help="(旧) σ_k / σ_0 スナップ比。現在は snap-act を使用")
    p.add_argument("--snap-act", type=float, default=0.01,
                   help="実測平均活動がこの値未満でスナップ")
    p.add_argument("--max-anneal-steps", type=int, default=30,
                   help="1 ユニットのアニール段数上限")
    p.add_argument("--drift-mult", type=float, default=1.5)
    p.add_argument("--drift-abs", type=float, default=0.005,
                   help="許容 = baseline * drift-mult + drift-abs")
    p.add_argument("--max-holds", type=int, default=5,
                   help="1 アニール段あたりの追加学習ブロック上限")
    p.add_argument("--stat-passes", type=int, default=4,
                   help="活性統計の forward パス数")
    p.add_argument("--ridge", type=float, default=0.1)
    return p


def main():
    p = argparse.ArgumentParser(description="noise-field self-consolidation PoC")
    add_poc_args(p)
    p.add_argument("--with-random", action="store_true",
                   help="route_B の削除順ランダム対照も実行")
    args = fncl.finalize_args(p.parse_args(), default_out="out/consolidation_poc")
    if args.quick:
        args.epochs_per_step = 5
        args.remove_frac = 0.25

    device = torch.device(args.device)
    x_raw, target, x, t = fncl.make_task(device)
    H = args.hidden_dim
    quota = {0: int(H * args.remove_frac), 1: int(H * args.remove_frac)}
    n_remove = quota[0] + quota[1]
    log_every = max(1, args.epochs // 5)

    all_results = {}
    for seed in args.seed_list:
        print(f"\n===== seed {seed} =====")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # --- 事前学習 (cov_jac; forward-only パイプライン) --------------------
        net = build_net(H, args.sigma, args.crossing_h, args.num_samples, device)
        trainer = CovJacTrainer(net, x, t, lr=args.pre_lr, opt=args.opt,
                                jac_ema=args.jac_ema)
        for e in range(args.epochs):
            loss = trainer.step()
            if e % log_every == 0 or e == args.epochs - 1:
                print(f"  [pretrain cov_jac_{args.opt}] epoch {e:5d} mse={loss:.5f}")
        trainer.close()
        base = float(np.mean(trainer.losses[-min(100, len(trainer.losses)):]))
        tol = base * args.drift_mult + args.drift_abs
        ckpt = checkpoint(net)
        pred0 = fncl.predict(net, x, passes=16)
        m0 = {"task_mse": float(np.mean((pred0 - target) ** 2)), "e_func": 0.0,
              "n_active": n_active(net), "noise_budget": noise_budget(net),
              "out_std": out_std(net, x)}
        print(f"  pretrained: task MSE={m0['task_mse']:.5f}  baseline(EMA)={base:.5f} "
              f" tol={tol:.5f}")

        res = {"pretrained": m0, "base": base, "tol": tol}
        curves, preds, points = {}, {"pretrained": pred0.tolist()}, {}

        # --- route_B（先に実行して epoch 予算を決める） -----------------------
        print("  --- route_B (noise-annealed self-consolidation) ---")
        restore(net, ckpt)
        args._budget = False
        curve_B, losses_B, snaps_B, holds_B = run_route_B(
            net, x, target, pred0, quota, tol, args)
        epochs_B = len(losses_B)
        final_B = dict(curve_B[-1], n_active=n_active(net),
                       noise_budget=noise_budget(net), out_std=out_std(net, x),
                       epochs=epochs_B, max_drift=float(np.max(losses_B)),
                       holds=holds_B)
        curves["route_B"] = curve_B
        preds["route_B"] = fncl.predict(net, x, passes=16).tolist()
        res["route_B"] = final_B
        print(f"    removed {n_remove} units in {epochs_B} epochs "
              f"(holds={holds_B}); final task MSE={final_B['task_mse']:.5f} "
              f"max drift={final_B['max_drift']:.5f}")

        # --- route_A / route_A_budget ----------------------------------------
        for name, budget in (("route_A", False), ("route_A_budget", True)):
            print(f"  --- {name} ---")
            restore(net, ckpt)
            args._budget = budget
            curve = run_route_A(net, x, target, pred0, quota, args)
            final = dict(curve[-1], n_active=n_active(net),
                         noise_budget=noise_budget(net), out_std=out_std(net, x),
                         epochs=0)
            curves[name] = curve
            res[name] = final
            preds[name] = fncl.predict(net, x, passes=16).tolist()
            print(f"    final task MSE={final['task_mse']:.5f} "
                  f"E_func={final['e_func']:.5f}")
            if name == "route_A":
                ckpt_A = checkpoint(net)

        # --- route_A_ft（route_A の状態から B と同予算の fine-tune） ----------
        print("  --- route_A_ft ---")
        restore(net, ckpt_A)
        tr = CovJacTrainer(net, x, t, lr=args.ft_lr, opt=args.opt,
                           jac_ema=args.jac_ema)
        tr.run(epochs_B)
        tr.close()
        final = dict(eval_net(net, x, target, pred0), n_active=n_active(net),
                     noise_budget=noise_budget(net), out_std=out_std(net, x),
                     epochs=epochs_B)
        res["route_A_ft"] = final
        points["route_A_ft"] = (n_remove, final)
        preds["route_A_ft"] = fncl.predict(net, x, passes=16).tolist()
        print(f"    final task MSE={final['task_mse']:.5f}")

        # --- abrupt_ft（PoC-3 対照: 一括 σ=0 + 同予算 fine-tune） -------------
        print("  --- abrupt_ft (control) ---")
        restore(net, ckpt)
        chosen = initial_ranking(net, x, quota, args)
        spike, losses_ab = run_abrupt(net, x, target, pred0, chosen, epochs_B, args)
        final = dict(eval_net(net, x, target, pred0), n_active=n_active(net),
                     noise_budget=noise_budget(net), out_std=out_std(net, x),
                     epochs=epochs_B, spike_mse=spike["task_mse"])
        res["abrupt_ft"] = final
        points["abrupt_noft"] = (n_remove, spike)
        points["abrupt_ft"] = (n_remove, final)
        preds["abrupt_ft"] = fncl.predict(net, x, passes=16).tolist()
        print(f"    spike (right after zeroing) MSE={spike['task_mse']:.5f}; "
              f"final task MSE={final['task_mse']:.5f}")

        # --- route_B_rand（削除順ランダム対照; 任意） --------------------------
        if args.with_random:
            print("  --- route_B_rand ---")
            restore(net, ckpt)
            args._budget = False
            curve_r, losses_r, _, _ = run_route_B(net, x, target, pred0, quota,
                                                  tol, args, random_order=True)
            final = dict(curve_r[-1], n_active=n_active(net),
                         noise_budget=noise_budget(net), out_std=out_std(net, x),
                         epochs=len(losses_r), max_drift=float(np.max(losses_r)))
            curves["route_B_rand"] = curve_r
            res["route_B_rand"] = final
            print(f"    final task MSE={final['task_mse']:.5f}")

        # --- 判定 --------------------------------------------------------------
        v1 = res["route_B"]["task_mse"] <= 1.5 * m0["task_mse"] + args.drift_abs
        v2 = (res["route_B"]["max_drift"] < res["abrupt_ft"]["spike_mse"]
              and res["route_B"]["task_mse"] <= 1.1 * res["abrupt_ft"]["task_mse"]
              + args.drift_abs)
        v3 = res["route_A_budget"]["e_func"] <= res["route_A"]["e_func"]
        res["verdicts"] = {"V1_consolidation": bool(v1), "V2_annealing": bool(v2),
                           "V3_budget": bool(v3)}
        all_results[seed] = res

        # --- 図表（先頭 seed のみ） --------------------------------------------
        if seed == args.seed_list[0]:
            fig_deletion_curves(curves, points,
                                args.out_dir / "fig_deletion_curves.png")
            fig_anneal_trace(losses_B, snaps_B, losses_ab,
                             spike["task_mse"], tol, base,
                             args.out_dir / "fig_anneal_trace.png")
            show = {k: np.asarray(v) for k, v in preds.items()
                    if k in ("pretrained", "route_A", "route_B", "abrupt_ft")}
            fig_predictions(x_raw, target, show,
                            args.out_dir / "fig_predictions.png")

    # --- 表 ---------------------------------------------------------------------
    seed0 = args.seed_list[0]
    r = all_results[seed0]
    rows = ["| method | N_active | Σσ² | task MSE | E_func | out std | epochs |",
            "|---|---|---|---|---|---|---|"]
    order = ["pretrained", "route_A", "route_A_budget", "route_A_ft",
             "route_B", "abrupt_ft"] + (["route_B_rand"] if args.with_random else [])
    for name in order:
        m = r[name]
        rows.append(f"| {name} | {m.get('n_active', 2 * H)} "
                    f"| {m.get('noise_budget', 2 * H * args.sigma ** 2):.2f} "
                    f"| {m['task_mse']:.5f} | {m['e_func']:.5f} "
                    f"| {m.get('out_std', float('nan')):.4f} "
                    f"| {m.get('epochs', '—')} |")
    v = r["verdicts"]
    lines = [f"**Consolidation PoC** (H={H}, T={args.num_samples}, "
             f"remove {n_remove}/{2 * H} units, opt={args.opt}, "
             f"pre-epochs={args.epochs}, seed={seed0})", "",
             *rows, "",
             f"- pretrain baseline MSE = {r['base']:.5f}, drift tol = {r['tol']:.5f}",
             f"- route_B: 移行中最大 MSE = {r['route_B']['max_drift']:.5f} "
             f"(abrupt の削除直後 = {r['abrupt_ft']['spike_mse']:.5f})",
             "",
             f"- V1 consolidation (route_B final <= 1.5x pretrained): "
             f"{'**PASS**' if v['V1_consolidation'] else '**FAIL**'}",
             f"- V2 annealing (smooth transition & final <= abrupt_ft): "
             f"{'**PASS**' if v['V2_annealing'] else '**FAIL**'}",
             f"- V3 noise-budget (A_budget E_func <= A E_func): "
             f"{'**PASS**' if v['V3_budget'] else '**FAIL**'}"]
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_final.md", table)
    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args),
               "results": {str(s): all_results[s] for s in all_results}})


if __name__ == "__main__":
    main()
