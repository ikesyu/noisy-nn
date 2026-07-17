"""
consolidation_gauge_equiv.py — ゲージ等価性の実測
(docs/idea_consolidation.md §4.6 / §12.7.5 の「残る検証」)

主張（§4.6）:

    (alpha w_k, alpha b_k, alpha sigma_k, h_k)  ==gauge==  (w_k, b_k, sigma_k, h_k / alpha)

すなわち「しきい値を 1/alpha 倍に上げる」（経路 H）と「入力側重み・バイアス・
注入ノイズを一斉に alpha 倍へ縮める」（経路 S）は、ユニットの機能（交差統計）
として同一のはずである。学習は介在させず、事前学習済みネットを凍結して
ユニットごとに forward 測定だけで確認する。

比較する3経路（アニール係数 alpha^t, t = 0..steps）:

    H      : h_k <- h_0 / alpha^t             （w, b, sigma は不変）
    S      : (w_k, b_k, sigma_k) <- alpha^t x （h は不変）    ← H と一致するはず
    sigma  : sigma_k <- alpha^t x sigma_0 のみ （対照; 深い層では上流ゆらぎが
             残るため 0 に落ちず、リーク（§12.7.2）を再現するはず）

測定量: 交差率 nu_k = E[z_k]（N x T x passes の平均）とチューニング曲線
zbar_k(x)。判定: max_t |nu_H - nu_S| がモンテカルロ床（経路 H を独立乱数で
2回測った差）と同水準であること。

生成物 (out/consolidation_gauge_equiv/):
  fig_gauge.png    層別の nu 軌跡（3経路 + MC 床）と |H - S| の比較
  table_gauge.md   ユニットごとの max 偏差 vs MC 床
  results.json

実行例:
  python tmp/consolidation_gauge_equiv.py --quick
  python tmp/consolidation_gauge_equiv.py
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
from nnn.stats import Capture  # noqa: E402


def measure_unit(net, x, l: int, k: int, passes: int):
    """ユニット (l, k) の交差率 nu とチューニング曲線 zbar(x) を測る."""
    cap = Capture(net)
    zs = []
    with torch.no_grad():
        for _ in range(passes):
            net(x)
            zs.append(cap.z[l][:, :, k])
    cap.remove()
    z = torch.cat(zs, dim=1)                       # [N, P*T]
    return float(z.mean()), z.mean(dim=1).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description="gauge equivalence of h-escalation")
    poc.add_poc_args(p)
    p.add_argument("--steps", type=int, default=12, help="アニール段数")
    p.add_argument("--units", type=int, default=4, help="層ごとの被験ユニット数")
    p.add_argument("--meas-passes", type=int, default=8,
                   help="1 測定あたりの forward パス数")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/consolidation_gauge_equiv")
    if args.quick:
        args.steps = 6
        args.units = 2
        args.meas_passes = 4

    device = torch.device(args.device)
    x_raw, target, x, t = fncl.make_task(device)
    H = args.hidden_dim
    seed = args.seed_list[0]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- 事前学習（機能的に意味のある動作点で測るため; 等価性自体は任意点で成立） ---
    net = poc.build_net(H, args.sigma, args.crossing_h, args.num_samples, device)
    trainer = poc.CovJacTrainer(net, x, t, lr=args.pre_lr, opt=args.opt,
                                jac_ema=args.jac_ema)
    log_every = max(1, args.epochs // 5)
    for e in range(args.epochs):
        loss = trainer.step()
        if e % log_every == 0 or e == args.epochs - 1:
            print(f"  [pretrain] epoch {e:5d} mse={loss:.5f}")
    trainer.close()
    ckpt = poc.checkpoint(net)

    # --- 被験ユニット: 各層でベースライン活動の大きい順に --units 個 ---
    cap = Capture(net)
    with torch.no_grad():
        net(x)
    base_act = {l: cap.z[l].mean(dim=(0, 1)) for l in (0, 1)}
    cap.remove()
    subjects = {l: torch.argsort(base_act[l], descending=True)[:args.units].tolist()
                for l in (0, 1)}
    print("subjects:", {f"L{l + 1}": subjects[l] for l in (0, 1)})

    alphas = [args.anneal_alpha ** t for t in range(args.steps + 1)]
    res = {}
    for l in (0, 1):
        res[l] = {}
        for k in subjects[l]:
            w0 = net.fcs[l].weight.data[k].clone()
            b0 = net.fcs[l].bias.data[k].clone()
            s0 = float(args.sigma)
            h0 = float(args.crossing_h)
            nu = {p_: [] for p_ in ("H", "H2", "S", "sigma")}
            curve_dev = []                        # mean_x |zbar_H - zbar_S|
            for a in alphas:
                # 経路 H: h_k <- h0 / a
                poc.restore(net, ckpt)
                net.h_vecs[l][k] = h0 / a
                nH, cH = measure_unit(net, x, l, k, args.meas_passes)
                nH2, _ = measure_unit(net, x, l, k, args.meas_passes)  # MC 床
                # 経路 S: (w, b, sigma) <- a x
                poc.restore(net, ckpt)
                net.fcs[l].weight.data[k] = a * w0
                net.fcs[l].bias.data[k] = a * b0
                net.sigma_vecs[l][k] = a * s0
                nS, cS = measure_unit(net, x, l, k, args.meas_passes)
                # 対照: sigma のみ
                poc.restore(net, ckpt)
                net.sigma_vecs[l][k] = a * s0
                nsig, _ = measure_unit(net, x, l, k, args.meas_passes)
                nu["H"].append(nH)
                nu["H2"].append(nH2)
                nu["S"].append(nS)
                nu["sigma"].append(nsig)
                curve_dev.append(float(np.mean(np.abs(cH - cS))))
            dev_HS = float(np.max(np.abs(np.asarray(nu["H"])
                                         - np.asarray(nu["S"]))))
            floor = float(np.max(np.abs(np.asarray(nu["H"])
                                        - np.asarray(nu["H2"]))))
            res[l][k] = {"nu": {p_: list(map(float, v)) for p_, v in nu.items()},
                         "curve_dev": curve_dev,
                         "dev_HS": dev_HS, "mc_floor": floor}
            print(f"  L{l + 1}#{k:2d}: max|nu_H-nu_S|={dev_HS:.4f} "
                  f"(MC floor {floor:.4f}); "
                  f"sigma-only end nu={nu['sigma'][-1]:.4f} "
                  f"vs H end nu={nu['H'][-1]:.4f}")
    poc.restore(net, ckpt)

    # --- 図 ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    tgrid = np.arange(args.steps + 1)
    for col, l in enumerate((0, 1)):
        ax = axes[0][col]
        for k in subjects[l]:
            r = res[l][k]
            ax.plot(tgrid, r["nu"]["H"], "o-", ms=3,
                    label=f"#{k} H (h/α^t)" if k == subjects[l][0] else None,
                    color="tab:blue", alpha=0.8)
            ax.plot(tgrid, r["nu"]["S"], "s--", ms=3,
                    label=f"#{k} S (αw, αb, ασ)" if k == subjects[l][0] else None,
                    color="tab:orange", alpha=0.8)
            ax.plot(tgrid, r["nu"]["sigma"], ":", lw=1.4,
                    label=f"#{k} σ-only (control)" if k == subjects[l][0] else None,
                    color="tab:red", alpha=0.8)
        ax.set_title(f"layer {l + 1}: crossing rate ν along the anneal")
        ax.set_xlabel("anneal step t  (factor α^t)")
        ax.set_ylabel("ν (mean activity)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1][col]
        devs = [res[l][k]["dev_HS"] for k in subjects[l]]
        floors = [res[l][k]["mc_floor"] for k in subjects[l]]
        xs = np.arange(len(subjects[l]))
        ax.bar(xs - 0.17, devs, width=0.34, label="max |ν_H − ν_S|")
        ax.bar(xs + 0.17, floors, width=0.34, label="MC floor |ν_H − ν_H'|")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"#{k}" for k in subjects[l]])
        ax.set_title(f"layer {l + 1}: deviation vs Monte-Carlo floor")
        ax.set_ylabel("deviation")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Gauge equivalence: h-escalation ≡ input-side scale-down "
                 f"(H={H}, seed {seed})", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    savefig(fig, args.out_dir / "fig_gauge.png")

    # --- 表 ---
    lines = [f"**Gauge equivalence check** (H={H}, steps={args.steps}, "
             f"alpha={args.anneal_alpha}, passes={args.meas_passes}, "
             f"seed={seed})", "",
             "| layer | unit | max nu_H-nu_S | MC floor | curve dev (max) | "
             "sigma-only end nu | H end nu |",
             "|---" * 7 + "|"]
    for l in (0, 1):
        for k in subjects[l]:
            r = res[l][k]
            lines.append(
                f"| L{l + 1} | #{k} | {r['dev_HS']:.4f} | {r['mc_floor']:.4f} "
                f"| {max(r['curve_dev']):.4f} | {r['nu']['sigma'][-1]:.4f} "
                f"| {r['nu']['H'][-1]:.4f} |")
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_gauge.md", table)
    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args),
               "results": {f"L{l + 1}": {str(k): res[l][k] for k in res[l]}
                           for l in (0, 1)}})


if __name__ == "__main__":
    main()
