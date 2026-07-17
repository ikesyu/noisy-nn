"""
consolidation_soft.py — ソフト・コンソリデーション第1段: readout-only overlap
(docs/idea_consolidation.md §12.9 案2「浅い共有」)

ハード・コンソリデーション (§12.8, consolidation_solid.py) は関数ごとに互いに
素なノイズ場領域を割り当てる。本実験はその最小の緩和として、過去タスクの
領域を **読み出し専用の共有語彙** として新タスクの forward に参加させる:

  - 過去領域の入力側 (W1 行 / b1 / W2 行 / b2) は勾配マスクで完全凍結
    （CovJacTrainer.grad_masks）。忘却ゼロの論理は h ゲートの物理的隔離から
    明示的な凍結マスクへ移る — これが soft 化の正直な代償。
  - 過去 L1 列 -> 現役 L2 行の交差重みは 0 に固定（語彙ユニットのチューニング
    曲線を changed / mixed にしない = 語彙の純度保持）。共有は最終隠れ層の
    語彙に対する **新タスクの読み出し列** のみ。
  - タスク記述子 = 場マスク + 読み出しベクトル + 出力バイアス。ハード版の
    記述子（場マスク + b_out）の自然な拡張で、推論時にタスクごとに復元する。

各タスクは「空き領域 + 語彙」で学習した後、空き領域のユニットだけを
anneal-until-stop で限界まで整理する: 貪欲 min S_k（L2 候補の回帰基底には
語彙も入れる — 読み出しは語彙へ補償を流せるため。L1 候補の基底は自前のみ —
語彙 L1 への交差列は凍結 0 で補償が流せない）で選び、スナップ後に損失 EMA が
許容へ戻らなければそのユニットを巻き戻して層を打ち切る。ハード対照
（語彙なし・同じ stop 則）との自前ユニット数の差が「重なりによる節約」を与える。

タスク列は sin(x) -> cos(x) -> sin(x+pi)。sin(x+pi) = -sin(x) はタスク1の語彙の
読み出し係数の符号反転だけで表せるため、類似度 -> 重なりの判別実験になる。

検証項目:
  V-a 各タスクが許容内で学習・整理できた（soft）
  V-b 忘却厳密ゼロ: 過去領域の入力側パラメータの max|Δ| = 0（hard / soft とも）
  V-c 節約: soft の自前ユニット総数 < hard
  V-d 類似度→重なり: 語彙寄与率が task3 (-sin) > task2 (cos)

生成物 (out/consolidation_soft/):
  fig_soft.png    予測 + 占有マップ (hard vs soft) + 自前ユニット数 / 語彙寄与率
  table_soft.md   タスク別比較表 + 判定
  results.json

実行例:
  python tmp/consolidation_soft.py --quick
  python tmp/consolidation_soft.py
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


# ============================================================
# ノイズ場・凍結マスク・領域スナップショット
# ============================================================
def set_field(net, active: dict, sigma0: float, h0: float) -> None:
    """active={l:[units]} を動員 (σ0, h0)、他を休眠 (0, H_DEAD) にする."""
    H = net.sigma_vecs[0].shape[0]
    for l in (0, 1):
        sig = torch.zeros(H)
        hv = torch.full((H,), poc.H_DEAD)
        for k in active[l]:
            sig[k] = sigma0
            hv[k] = h0
        net.sigma_vecs[l] = sig.to(net.sigma_vecs[l].device)
        net.h_vecs[l] = hv.to(net.h_vecs[l].device)


def freeze_masks(H: int, past: dict, device):
    """過去領域の入力側を凍結する CovJacTrainer.grad_masks を作る.

    W1: 過去 L1 行を凍結。W2: 過去 L2 行（語彙のチューニング曲線）を丸ごと
    凍結し、さらに過去 L1 列を全行で凍結（現役 L2 行への交差重みは 0 に
    保たれ、共有チャネルは読み出し列だけになる）。読み出し層は共有チャネル
    そのものなのでマスクしない。
    """
    mW0 = torch.ones(H, 1, device=device)
    mb0 = torch.ones(H, device=device)
    mW1 = torch.ones(H, H, device=device)
    mb1 = torch.ones(H, device=device)
    for k in past[0]:
        mW0[k, :] = 0.0
        mb0[k] = 0.0
        mW1[:, k] = 0.0
    for k in past[1]:
        mW1[k, :] = 0.0
        mb1[k] = 0.0
    return {0: (mW0, mb0), 1: (mW1, mb1)}


def zero_cross_columns(net, past: dict) -> None:
    """現役 L2 行に残る過去 L1 列の交差重みを 0 にする.

    これらは過去タスク学習時代の残骸で、過去タスクの推論では相手側が休眠のため
    don't-care だが（region_snapshot の対象外）、soft では過去 L1 が活動する
    ので語彙の純度のため明示的に消す。過去 L2 行は自領域列以外もともと 0。
    """
    H = net.fcs[1].weight.shape[0]
    rows = [r for r in range(H) if r not in past[1]]
    if past[0] and rows:
        ri = torch.tensor(rows).unsqueeze(1)
        ci = torch.tensor(sorted(past[0])).unsqueeze(0)
        net.fcs[1].weight.data[ri, ci] = 0.0


def region_snapshot(net, region: dict):
    """タスクの関数に参加する入力側パラメータ（忘却ゼロ判定用）.

    読み出し（wout / b_out）はタスク記述子として保存・復元されるため対象外。
    """
    i0 = torch.tensor(region[0], dtype=torch.long)
    i1 = torch.tensor(region[1], dtype=torch.long)
    return {
        "w1": net.fcs[0].weight.data[i0].clone(),
        "b1": net.fcs[0].bias.data[i0].clone(),
        "w2": net.fcs[1].weight.data[i1][:, i0].clone(),
        "b2": net.fcs[1].bias.data[i1].clone(),
    }


def region_drift(net, region: dict, snap) -> float:
    now = region_snapshot(net, region)
    diffs = [float((now[k] - snap[k]).abs().max())
             for k in snap if snap[k].numel() > 0]
    return max(diffs) if diffs else 0.0


# ============================================================
# anneal-until-stop（stop 則つき route_B; §12.5 の複合判定の実装形）
# ============================================================
def trainer_state(tr):
    return (copy.deepcopy(tr.W), copy.deepcopy(tr.opt.m),
            copy.deepcopy(tr.opt.v), copy.deepcopy(tr.opt.step),
            len(tr.losses))


def trainer_rollback(tr, st):
    W, m, v, step, nloss = st
    tr.W = copy.deepcopy(W)
    tr.opt.m = copy.deepcopy(m)
    tr.opt.v = copy.deepcopy(v)
    tr.opt.step = copy.deepcopy(step)
    del tr.losses[nloss:]


def anneal_until_stop(net, x, target, tol, args, eligible: dict, grad_masks,
                      vocab: dict):
    """空き領域 eligible を貪欲 min S_k でアニールし、限界で自動停止する.

    1 ユニットごとに (net, trainer) をチェックポイントし、スナップ + 回復猶予
    （stop_recovery ブロック）後も損失 EMA が許容を超えていたら巻き戻して
    その層を打ち切る。両層が閉じたら終了。
    """
    t_target = torch.tensor(target, device=x.device).unsqueeze(1)
    trainer = poc.CovJacTrainer(net, x, t_target, lr=args.ft_lr, opt=args.opt,
                                jac_ema=args.jac_ema)
    trainer.grad_masks = grad_masks
    active = {l: list(eligible[l]) for l in (0, 1)}
    open_layers = {l for l in (0, 1) if active[l]}
    holds_total = 0
    while open_layers:
        # --- 候補選択: 冗長性スコア S_k（基底 = 補償が実際に流せる先） ---
        _, zbar = poc.collect_stats(net, x, passes=args.stat_passes)
        scored = {}
        for l in sorted(open_layers):
            Wn = net.fcs[l + 1].weight.data
            best_l = None
            basis_all = active[l] + (vocab[l] if l == 1 else [])
            for k in active[l]:
                w2 = float((Wn[:, k] ** 2).sum())
                rest = [j for j in basis_all if j != k]
                if w2 == 0.0:
                    S = 0.0            # 生きた消費者がない -> 無償で削除できる
                elif not rest:
                    continue
                else:
                    _, _, resid = poc.ridge_fit(zbar[l][:, rest],
                                                zbar[l][:, k], args.ridge)
                    S = (w2 * float((resid ** 2).sum())
                         / (float((zbar[l][:, k] ** 2).sum()) + poc.EPS))
                if best_l is None or S < best_l[0]:
                    best_l = (S, k)
            if best_l is None:
                open_layers.discard(l)
            else:
                scored[l] = best_l
        if not scored:
            break
        l = min(scored, key=lambda ll: scored[ll][0])
        k = scored[l][1]
        # --- 消滅経路のアニール + 閉ループ (run_route_B と同一の内側ループ) ---
        ck_net, ck_tr = poc.checkpoint(net), trainer_state(trainer)
        steps = 0
        while True:
            net.sigma_vecs[l][k] *= args.anneal_alpha
            if l >= 1:
                net.h_vecs[l][k] = net.h_vecs[l][k] / args.anneal_alpha
            trainer.run(args.epochs_per_step)
            h = 0
            while trainer.ema() > tol and h < args.max_holds:
                trainer.run(args.epochs_per_step)
                h += 1
            holds_total += h
            steps += 1
            act = float(trainer.cap.z[l][:, :, k].mean())
            if act < args.snap_act or steps >= args.max_anneal_steps:
                break
        poc.kill_unit(net, l, k)
        # --- stop 判定: 回復猶予後も EMA > tol なら巻き戻して層を閉じる ---
        rec = 0
        while trainer.ema() > tol and rec < args.stop_recovery:
            trainer.run(args.epochs_per_step)
            rec += 1
        if trainer.ema() > tol:
            poc.restore(net, ck_net)
            trainer_rollback(trainer, ck_tr)
            open_layers.discard(l)
            print(f"    [stop] L{l + 1} unit {k}: 吸収不能 -> 巻き戻し "
                  f"(自前 L{l + 1} = {len(active[l])} で確定)")
        else:
            active[l].remove(k)
            if not active[l]:
                open_layers.discard(l)
    trainer.close()
    return trainer.losses, holds_total


# ============================================================
# 逐次実行（1 モード分）
# ============================================================
def eval_with_descriptor(net, x, r: dict, passes: int = 16):
    """タスク記述子（場 + 読み出し + b_out）を復元して予測する."""
    set_field(net, r["support"], r["sigma0"], r["h0"])
    net.fcs[2].weight.data.copy_(r["wout"])
    net.fcs[2].bias.data.copy_(r["b_out"])
    return fncl.predict(net, x, passes=passes)


def vocab_energy(net, x, r: dict, registry, args):
    """読み出し寄与の分散を自前 / 語彙 / 供給元タスク別に分解する.

    寄与 c_j(x) = wout_j * zbar_j(x)。share = Var_x(語彙寄与和) / Var_x(全寄与和)。
    """
    set_field(net, r["support"], r["sigma0"], r["h0"])
    net.fcs[2].weight.data.copy_(r["wout"])
    net.fcs[2].bias.data.copy_(r["b_out"])
    _, zbar = poc.collect_stats(net, x, passes=args.stat_passes)
    w = r["wout"].squeeze(0)                       # [H]
    c = zbar[1] * w                                # [N, H] 単位寄与
    var = lambda f: float(f.var(unbiased=False))   # noqa: E731

    own2 = r["region"][1]
    voc2 = [j for j in r["support"][1] if j not in own2]
    f_all = c[:, r["support"][1]].sum(dim=1)
    v_all = var(f_all) + poc.EPS
    share = var(c[:, voc2].sum(dim=1)) / v_all if voc2 else 0.0
    by_source = {}
    for s in registry:
        if s is r:
            break
        src2 = [j for j in s["region"][1] if j in voc2]
        by_source[s["name"]] = (var(c[:, src2].sum(dim=1)) / v_all
                                if src2 else 0.0)
    return share, by_source


def run_sequence(mode: str, seed: int, args, device, tasks, x, x_raw):
    """mode in ('hard', 'soft'): 逐次「学習 -> anneal-until-stop -> 解放」."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    H = args.hidden_dim
    net = poc.build_net(H, args.sigma, args.crossing_h, args.num_samples, device)
    past = {0: [], 1: []}
    free = {0: list(range(H)), 1: list(range(H))}
    registry = []

    for ti, (name, target) in enumerate(tasks):
        vocab = ({l: sorted(past[l]) for l in (0, 1)} if mode == "soft"
                 else {0: [], 1: []})
        print(f"\n  ===== [{mode}] task {ti + 1}: {name}  "
              f"(free L1 {len(free[0])} / L2 {len(free[1])}, "
              f"vocab L2 {len(vocab[1])}) =====")
        # --- ワークスペース整備: 読み出しはタスクごとに白紙から学習 ---
        net.fcs[2].weight.data.zero_()
        net.fcs[2].bias.data.zero_()
        if mode == "soft":
            zero_cross_columns(net, past)
        mobil = {l: sorted(set(free[l]) | set(vocab[l])) for l in (0, 1)}
        set_field(net, mobil, args.sigma, args.crossing_h)
        masks = (freeze_masks(H, past, device)
                 if mode == "soft" and (past[0] or past[1]) else None)

        # --- 初期学習（空き + 語彙; 語彙の入力側は凍結） ---
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

        # --- anneal-until-stop（自前 = 空き領域のみが候補） ---
        losses, holds = anneal_until_stop(net, x, target, tol, args,
                                          eligible=free, grad_masks=masks,
                                          vocab=vocab)
        region = {l: [k for k in free[l] if float(net.sigma_vecs[l][k]) > 0]
                  for l in (0, 1)}
        support = {l: sorted(region[l] + vocab[l]) for l in (0, 1)}
        r = {"name": name, "target": target, "region": region,
             "support": support, "sigma0": float(args.sigma),
             "h0": float(args.crossing_h),
             "wout": net.fcs[2].weight.data.clone(),
             "b_out": net.fcs[2].bias.data.clone(),
             "snap": region_snapshot(net, region), "tol": tol, "holds": holds,
             "anneal_epochs": len(losses)}
        registry.append(r)
        pred = eval_with_descriptor(net, x, r)
        r["mse_at_consolidation"] = float(np.mean((pred - target) ** 2))
        print(f"    own region L1 {len(region[0])} + L2 {len(region[1])} "
              f"(holds={holds}); MSE={r['mse_at_consolidation']:.5f} "
              f"(tol={tol:.4f})")
        past = {l: past[l] + region[l] for l in (0, 1)}
        free = {l: [k for k in free[l] if k not in region[l]] for l in (0, 1)}

    # ---------------- 最終評価 ----------------
    K = len(tasks)
    out = {"tasks": [], "cross": np.zeros((K, K))}
    preds = []
    for i, r in enumerate(registry):
        pred = eval_with_descriptor(net, x, r)
        preds.append(pred)
        mse = float(np.mean((pred - r["target"]) ** 2))
        drift = region_drift(net, r["region"], r["snap"])
        for j in range(K):
            out["cross"][i, j] = float(np.mean((pred - registry[j]["target"]) ** 2))
        # 語彙を落とした自前のみの場での MSE（アブレーション）
        r_own = dict(r, support=r["region"])
        mse_own = float(np.mean((eval_with_descriptor(net, x, r_own)
                                 - r["target"]) ** 2))
        share, by_source = ((0.0, {}) if mode == "hard"
                            else vocab_energy(net, x, r, registry, args))
        out["tasks"].append({
            "name": r["name"],
            "n_own": {"0": len(r["region"][0]), "1": len(r["region"][1])},
            "region": {str(l): r["region"][l] for l in (0, 1)},
            "support_l2": len(r["support"][1]),
            "mse_at_consolidation": r["mse_at_consolidation"],
            "mse_final": mse, "mse_own_only": mse_own, "drift": drift,
            "vocab_share": share, "vocab_share_by_source": by_source,
            "holds": r["holds"], "tol": r["tol"],
            "anneal_epochs": r["anneal_epochs"]})
        print(f"  [{mode}] {r['name']}: final MSE={mse:.5f} "
              f"own-only MSE={mse_own:.5f} drift={drift:.2e} "
              f"vocab_share={share:.3f}")
    out["total_own"] = sum(t["n_own"]["0"] + t["n_own"]["1"]
                           for t in out["tasks"])
    out["preds"] = preds
    out["registry"] = registry
    return out


# ============================================================
# 図・表
# ============================================================
def make_figure(tasks, x_raw, res, H, seed, path):
    K = len(tasks)
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(4, K, height_ratios=[2.0, 0.55, 0.55, 1.5])
    for i in range(K):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(x_raw, tasks[i][1], "k--", lw=1.2, label="target")
        ax.plot(x_raw, res["soft"]["preds"][i], lw=1.4, label="soft (own field)")
        t = res["soft"]["tasks"][i]
        ax.set_title(f"task {i + 1}: {t['name']}\nMSE={t['mse_final']:.4f}  "
                     f"own L1+L2={t['n_own']['0']}+{t['n_own']['1']}  "
                     f"vocab share={t['vocab_share']:.2f}")
        ax.set_ylim(-1.6, 1.6)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    for row, mode in ((1, "hard"), (2, "soft")):
        occ = np.zeros((2, H))
        for i, t in enumerate(res[mode]["tasks"]):
            for l in (0, 1):
                for k in t["region"][str(l)]:
                    occ[l, k] = i + 1
        ax = fig.add_subplot(gs[row, :])
        ax.imshow(occ, aspect="auto", cmap=plt.get_cmap("tab10", K + 1),
                  vmin=-0.5, vmax=K + 0.5, origin="lower",
                  extent=[-0.5, H - 0.5, -0.5, 1.5], interpolation="nearest")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["L1", "L2"], fontsize=8)
        ax.set_title(f"{mode}: own-region occupancy "
                     f"(total {res[mode]['total_own']} units)", fontsize=9)
        if row == 1:
            ax.set_xticks([])

    ax = fig.add_subplot(gs[3, :K - 1])
    idx = np.arange(K)
    wd = 0.35
    for off, mode, color in ((-wd / 2, "hard", "tab:gray"),
                             (wd / 2, "soft", "tab:blue")):
        n1 = [t["n_own"]["0"] for t in res[mode]["tasks"]]
        n2 = [t["n_own"]["1"] for t in res[mode]["tasks"]]
        ax.bar(idx + off, n1, wd, color=color, label=f"{mode} L1")
        ax.bar(idx + off, n2, wd, bottom=n1, color=color, alpha=0.5,
               label=f"{mode} L2")
    ax.set_xticks(idx)
    ax.set_xticklabels([t[0] for t in tasks], fontsize=8)
    ax.set_ylabel("own units")
    ax.set_title("Own-region size: hard vs soft (readout-only overlap)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[3, K - 1])
    shares = [t["vocab_share"] for t in res["soft"]["tasks"]]
    ax.bar(idx, shares, 0.5, color="tab:orange")
    ax.set_xticks(idx)
    ax.set_xticklabels([t[0] for t in tasks], fontsize=8)
    ax.set_ylabel("vocab share of output variance")
    ax.set_title("Realized overlap (soft)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Soft consolidation step 1: readout-only overlap "
                 f"(H={H}, seed {seed})", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    savefig(fig, path)


# ============================================================
# main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="soft consolidation step 1: "
                                            "readout-only overlap")
    poc.add_poc_args(p)
    p.add_argument("--epochs-task", type=int, default=1000,
                   help="タスクごとの初期学習 epoch 数")
    p.add_argument("--stop-recovery", type=int, default=8,
                   help="スナップ後の回復猶予ブロック数（超過で巻き戻し停止）")
    args = fncl.finalize_args(p.parse_args(), default_out="out/consolidation_soft")
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
    K = len(tasks)
    H = args.hidden_dim

    all_results = {}
    for seed in args.seed_list:
        print(f"\n===== seed {seed} =====")
        res = {}
        for mode in ("hard", "soft"):
            res[mode] = run_sequence(mode, seed, args, device, tasks, x, x_raw)

        # --- 判定 ---
        soft_t, hard_t = res["soft"]["tasks"], res["hard"]["tasks"]
        va = all(t["mse_final"] <= 2.0 * t["tol"] for t in soft_t)
        vb = all(t["drift"] == 0.0 for t in soft_t + hard_t)
        vc = res["soft"]["total_own"] < res["hard"]["total_own"]
        vd = soft_t[2]["vocab_share"] > soft_t[1]["vocab_share"]
        verdicts = {"Va_within_tol": bool(va), "Vb_zero_forgetting": bool(vb),
                    "Vc_unit_savings": bool(vc), "Vd_similarity_overlap": bool(vd)}
        all_results[seed] = {
            "modes": {m: {"tasks": res[m]["tasks"],
                          "total_own": res[m]["total_own"],
                          "cross": res[m]["cross"].tolist()}
                      for m in ("hard", "soft")},
            "verdicts": verdicts}

        if seed == args.seed_list[0]:
            make_figure(tasks, x_raw, res, H, seed,
                        args.out_dir / "fig_soft.png")

    # --- 表（先頭 seed） ---
    seed0 = args.seed_list[0]
    r0 = all_results[seed0]["modes"]
    lines = [f"**Soft consolidation step 1: readout-only overlap** "
             f"(H={H}, T={args.num_samples}, epochs-task={args.epochs_task}, "
             f"opt={args.opt}, seeds={args.seeds})", "",
             "| mode | task | own L1+L2 | MSE final | MSE own-only "
             "| vocab share | drift | holds |",
             "|---" * 8 + "|"]
    for mode in ("hard", "soft"):
        for t in r0[mode]["tasks"]:
            lines.append(
                f"| {mode} | {t['name']} | {t['n_own']['0']}+{t['n_own']['1']} "
                f"| {t['mse_final']:.5f} | {t['mse_own_only']:.5f} "
                f"| {t['vocab_share']:.3f} | {t['drift']:.2e} | {t['holds']} |")
    lines += ["",
              f"- total own units: hard = {r0['hard']['total_own']}, "
              f"soft = {r0['soft']['total_own']} "
              f"(savings = {r0['hard']['total_own'] - r0['soft']['total_own']})"]
    lines += ["", "vocab share by source (soft):", ""]
    for t in r0["soft"]["tasks"]:
        if t["vocab_share_by_source"]:
            srcs = ", ".join(f"{k}: {v:.3f}"
                             for k, v in t["vocab_share_by_source"].items())
            lines.append(f"- {t['name']}: {srcs}")
    lines += [""]
    for seed in args.seed_list:
        v = all_results[seed]["verdicts"]
        lines.append(f"- seed {seed}: "
                     + " / ".join(f"{k} {'**PASS**' if ok else '**FAIL**'}"
                                  for k, ok in v.items()))
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_soft.md", table)
    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args),
               "results": {str(s): all_results[s] for s in all_results}})


if __name__ == "__main__":
    main()
