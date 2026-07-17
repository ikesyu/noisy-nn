"""
consolidation_solid.py — ハード・コンソリデーションの実証
(docs/idea_consolidation.md §12.8)

ハード・コンソリデーション = 複数の関数を、互いに素なノイズ場領域として
1つのネットワークに逐次格納する方式：

    タスク 1 (sin x)      : 空き全域を動員して学習 -> route_B で keep 領域へ圧縮
    タスク 2 (cos x)      : 空いた領域を動員して学習 -> 圧縮
    タスク 3 (sin(x+pi))  : 同上 ...

推論はタスク i の領域 S_i だけを動員（他は ρ=0: σ=0 かつ h=H_DEAD で沈黙）
して行う。h ゲート（§4.5）により沈黙ユニットは z ≡ 0 なので、

  (i) タスク間の重み保護が**物理的に自動**で成立する：沈黙ユニットは
      KDE スロープ 0・活性 0 のため勾配が厳密に 0 になり、後続タスクの
      学習が過去領域のパラメータに触れない（凍結マスク不要）。
  (ii) 共有される可変量は出力バイアスのみ -> タスクごとに保存・復元する
      （タスク記述子 = 領域マスク + 出力バイアス 1 個）。

検証項目:
  V-a 各タスクの最終 MSE（自領域の場で）が単独学習と同水準
  V-b 忘却ゼロ: タスク i の領域パラメータが後続学習で変化しない
      （max|Δ| = 0 の厳密判定）+ 整理直後と全タスク終了後の MSE 一致
  V-c 場が関数を選ぶ: 交差評価行列（場 i x 目標 j）で対角のみ小さい

生成物 (out/consolidation_solid/):
  fig_solid.png    タスク別予測 + 領域占有マップ + 逐次学習の損失軌跡
  table_solid.md   タスク別 MSE / パラメータ漂移 / 交差評価行列
  results.json

実行例:
  python tmp/consolidation_solid.py --quick
  python tmp/consolidation_solid.py
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


# ============================================================
# ノイズ場の操作（動員 / 休眠）
# ============================================================
def set_field(net, active: dict, sigma0: float, h0: float) -> None:
    """active={l:[units]} を動員 (σ0, h0)、他を休眠 (0, H_DEAD) にする.

    休眠は kill_unit と違い重みに触れない（一時的な沈黙）。h ゲートにより
    休眠ユニットは z=0・スロープ 0 となり、forward からも学習からも消える。
    """
    H = net.sigma_vecs[0].shape[0]
    for l in (0, 1):
        sig = torch.zeros(H)
        hv = torch.full((H,), poc.H_DEAD)
        for k in active[l]:
            sig[k] = sigma0
            hv[k] = h0
        net.sigma_vecs[l] = sig.to(net.sigma_vecs[l].device)
        net.h_vecs[l] = hv.to(net.h_vecs[l].device)


def region_snapshot(net, region: dict):
    """タスク i の関数に参加するパラメータのスナップショット（忘却ゼロ判定用）.

    層間重みは領域サブ行列 W2[region_L2, region_L1] のみを取る。行の残り
    （他タスクの L1 列）はタスク i の推論では相手が休眠 (z=0) のため機能に
    寄与しない「共有 don't-care 成分」であり、後続タスクの kill_unit（列ゼロ化）
    が上書きしてよい。
    """
    i0 = torch.tensor(region[0], dtype=torch.long)
    i1 = torch.tensor(region[1], dtype=torch.long)
    return {
        "w1": net.fcs[0].weight.data[i0].clone(),
        "b1": net.fcs[0].bias.data[i0].clone(),
        "w2": net.fcs[1].weight.data[i1][:, i0].clone(),
        "b2": net.fcs[1].bias.data[i1].clone(),
        "wout": net.fcs[2].weight.data[:, i1].clone(),
    }


def region_drift(net, region: dict, snap) -> float:
    now = region_snapshot(net, region)
    return max(float((now[k] - snap[k]).abs().max()) for k in snap)


def eval_task(net, x, target_1d, registry, i, passes: int = 16) -> float:
    """タスク i の場と出力バイアスを設定して MSE を測る."""
    r = registry[i]
    set_field(net, r["region"], r["sigma0"], r["h0"])
    net.fcs[2].bias.data.copy_(r["b_out"])
    pred = fncl.predict(net, x, passes=passes)
    return float(np.mean((pred - target_1d) ** 2)), pred


# ============================================================
# main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="hard consolidation: sequential "
                                            "multi-function learning")
    poc.add_poc_args(p)
    p.add_argument("--epochs-task", type=int, default=1000,
                   help="タスクごとの初期学習 epoch 数")
    p.add_argument("--keep-l1", type=int, default=8,
                   help="タスクごとに層1へ残すユニット数")
    p.add_argument("--keep-l2", type=int, default=4,
                   help="タスクごとに層2へ残すユニット数")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/consolidation_solid")
    if args.quick:
        args.epochs_task = 120
        args.epochs_per_step = 5
        args.keep_l1 = max(2, args.hidden_dim // 4)
        args.keep_l2 = 2

    device = torch.device(args.device)
    seed = args.seed_list[0]
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = 128
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, N, dtype=np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    tasks = [("sin(x)", np.sin(x_raw).astype(np.float32)),
             ("cos(x)", np.cos(x_raw).astype(np.float32)),
             ("sin(x+pi)", np.sin(x_raw + np.pi).astype(np.float32))]
    H = args.hidden_dim

    net = poc.build_net(H, args.sigma, args.crossing_h, args.num_samples, device)
    free = {0: list(range(H)), 1: list(range(H))}
    registry, all_losses, boundaries = [], [], []

    for ti, (name, target) in enumerate(tasks):
        print(f"\n===== task {ti + 1}: {name}  "
              f"(free: L1 {len(free[0])}, L2 {len(free[1])}) =====")
        t = torch.tensor(target, device=device).unsqueeze(1)

        # --- 空き領域を全動員して学習 -----------------------------------
        set_field(net, free, args.sigma, args.crossing_h)
        trainer = poc.CovJacTrainer(net, x, t, lr=args.pre_lr, opt=args.opt,
                                    jac_ema=args.jac_ema)
        log_every = max(1, args.epochs_task // 4)
        for e in range(args.epochs_task):
            loss = trainer.step()
            if e % log_every == 0 or e == args.epochs_task - 1:
                print(f"  [learn] epoch {e:5d} mse={loss:.5f}")
        trainer.close()
        base = float(np.mean(trainer.losses[-min(100, len(trainer.losses)):]))
        tol = base * args.drift_mult + args.drift_abs
        all_losses += trainer.losses
        pred0 = fncl.predict(net, x, passes=16)

        # --- route_B で keep サイズへ圧縮（対象 = 現タスクの動員領域のみ） ---
        quota = {0: max(0, len(free[0]) - args.keep_l1),
                 1: max(0, len(free[1]) - args.keep_l2)}
        curve, losses, snaps, holds = poc.run_route_B(
            net, x, target, pred0, quota, tol, args, eligible=free)
        all_losses += losses
        boundaries.append(len(all_losses))

        # --- 生存領域の確定・記録 ---------------------------------------
        region = {l: [k for k in free[l] if float(net.sigma_vecs[l][k]) > 0]
                  for l in (0, 1)}
        registry.append({
            "name": name, "target": target, "region": region,
            "sigma0": float(args.sigma), "h0": float(args.crossing_h),
            "b_out": net.fcs[2].bias.data.clone(),
            "snap": region_snapshot(net, region),
            "tol": tol, "holds": holds})
        mse_c, _ = eval_task(net, x, target, registry, ti)
        registry[ti]["mse_at_consolidation"] = mse_c
        print(f"  consolidated to L1 {len(region[0])} + L2 {len(region[1])} "
              f"units (holds={holds}); MSE={mse_c:.5f}")

        # アニールで離脱したユニットは空きプールへ戻す（kill_unit 済み =
        # 読み出し列 0・σ=0; 次タスクの set_field が再動員する）
        free = {l: [k for k in free[l] if k not in region[l]] for l in (0, 1)}

    # ---------------- 最終評価 ----------------
    print("\n===== final evaluation =====")
    K = len(tasks)
    final_mse, drift, cross = [], [], np.zeros((K, K))
    preds = []
    for i in range(K):
        m, pred = eval_task(net, x, registry[i]["target"], registry, i)
        final_mse.append(m)
        preds.append(pred)
        drift.append(region_drift(net, registry[i]["region"],
                                  registry[i]["snap"]))
        for j in range(K):
            cross[i, j] = float(np.mean((pred - registry[j]["target"]) ** 2))
        print(f"  task {i + 1} ({registry[i]['name']}): final MSE={m:.5f} "
              f"(at consolidation {registry[i]['mse_at_consolidation']:.5f}), "
              f"param drift={drift[i]:.2e}")

    # ---------------- 図 ----------------
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, K, height_ratios=[2.2, 0.9, 1.5])
    for i in range(K):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(x_raw, registry[i]["target"], "k--", lw=1.2, label="target")
        ax.plot(x_raw, preds[i], lw=1.4, label="prediction (own field)")
        ax.set_title(f"task {i + 1}: {registry[i]['name']}\n"
                     f"MSE={final_mse[i]:.4f}")
        ax.set_ylim(-1.6, 1.6)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    # 領域占有マップ
    occ = np.zeros((2, H))
    for i, r in enumerate(registry):
        for l in (0, 1):
            for k in r["region"][l]:
                occ[l, k] = i + 1
    ax = fig.add_subplot(gs[1, :])
    im = ax.imshow(occ, aspect="auto", cmap=plt.get_cmap("tab10", K + 1),
                   vmin=-0.5, vmax=K + 0.5, origin="lower",
                   extent=[-0.5, H - 0.5, -0.5, 1.5], interpolation="nearest")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["layer 1", "layer 2"])
    ax.set_xlabel("Neuron index")
    ax.set_title("Region occupancy (0 = released/free, i = task i) — "
                 "disjoint noise-field supports")
    plt.colorbar(im, ax=ax, ticks=range(K + 1), fraction=0.025)

    # 逐次学習の損失軌跡
    ax = fig.add_subplot(gs[2, :])
    ax.plot(all_losses, lw=0.7)
    for i, b in enumerate(boundaries):
        ax.axvline(b, color="r", lw=0.8, ls=":",
                   label="task boundary" if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xlabel("epoch (sequential: learn -> consolidate, per task)")
    ax.set_ylabel("eval MSE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.suptitle(f"Hard consolidation: {K} functions in one network via "
                 f"disjoint noise-field regions (H={H}, seed {seed})",
                 fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    savefig(fig, args.out_dir / "fig_solid.png")

    # ---------------- 表・判定 ----------------
    va = all(final_mse[i] <= 2.0 * registry[i]["tol"] for i in range(K))
    vb = all(d == 0.0 for d in drift)
    vc = all(cross[i, i] < 0.5 * min(cross[i, j] for j in range(K) if j != i)
             for i in range(K))
    lines = [f"**Hard consolidation** (H={H}, keep L1={args.keep_l1}/"
             f"L2={args.keep_l2} per task, epochs-task={args.epochs_task}, "
             f"seed={seed})", "",
             "| task | units (L1+L2) | MSE at consolidation | final MSE | "
             "param drift | holds |",
             "|---" * 6 + "|"]
    for i, r in enumerate(registry):
        lines.append(
            f"| {r['name']} | {len(r['region'][0])}+{len(r['region'][1])} "
            f"| {r['mse_at_consolidation']:.5f} | {final_mse[i]:.5f} "
            f"| {drift[i]:.2e} | {r['holds']} |")
    lines += ["", "cross-evaluation MSE (row = field, col = target):", "",
              "| field \\ target | " + " | ".join(r["name"] for r in registry)
              + " |", "|---" * (K + 1) + "|"]
    for i in range(K):
        lines.append(f"| {registry[i]['name']} | "
                     + " | ".join(f"{cross[i, j]:.4f}" for j in range(K))
                     + " |")
    lines += ["",
              f"- V-a 各タスクが許容内で学習・整理できた: "
              f"{'**PASS**' if va else '**FAIL**'}",
              f"- V-b 忘却ゼロ（領域パラメータの厳密不変）: "
              f"{'**PASS**' if vb else '**FAIL**'}",
              f"- V-c 場が関数を選ぶ（対角優位）: "
              f"{'**PASS**' if vc else '**FAIL**'}"]
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_solid.md", table)
    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args),
               "tasks": [{"name": r["name"],
                          "region": {str(l): r["region"][l] for l in (0, 1)},
                          "mse_at_consolidation": r["mse_at_consolidation"],
                          "final_mse": final_mse[i], "drift": drift[i],
                          "holds": r["holds"], "tol": r["tol"]}
                         for i, r in enumerate(registry)],
               "cross": cross.tolist(),
               "verdicts": {"Va": bool(va), "Vb": bool(vb), "Vc": bool(vc)}})


if __name__ == "__main__":
    main()
