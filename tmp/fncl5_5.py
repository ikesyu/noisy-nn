"""
fncl5_5.py — 論文 §5.5「読み出し誤差の共分散推定の実証」

(a) 読み出し誤差推定量の比較 (Adam): cov_jac_full の --jac-out 3 変種
      cov    : raw Cov(L,y)/Var(y)。歪度バイアス E[eps^3]/Var(eps) を持ち、
               収束後に Adam がそれを増幅して解から漂流する (ドリフト)。
      cov_m3 : 観測した y の 3 次中心モーメントを共分散から減算 (既定)。
               2 次損失に対して母集団で厳密にバイアスを除去し、backprop 級を維持。
      probe  : 対称ガウスプローブ xi への回帰 Cov(L(y+xi), xi)/Var(xi)。
               E[xi^3]=0 なので任意の滑らかな損失で不偏 (一般解)。
    参照として backprop / cov_jac_adam (解析的 dL/dy) を同一行列で走らせる。
    SGD 変種も既定で実行する (推定量によらず cov_jac_sgd の最適化床に
    留まること = 収束速度は読み出し推定量に依らないことの確認。
    --no-include-sgd で省略可)。

(b) 歪度バイアスの直接検証: cov_jac (adam) で --warmup-epochs だけ学習した
    ネットワーク上で、多数パスにわたる E[g_y] と真値 2(E[y]-t) の差 (観測
    バイアス) を入力ごとに測り、予測項 m3/Var と突き合わせる (散布図と相関)。
    L = (y-t)^2 が y の 2 次関数であることから、母集団で厳密に
      Cov(L,y)/Var(y) = 2(E[y]-t) + E[eps^3]/Var(eps),  eps = y - E[y]
    が成り立つ (このバイアスはサンプル数で消えない)。

生成物 (out/fncl5_5/):
  fig_readout_drift.png  -> 図 (学習曲線: raw のドリフト vs m3/probe の安定, log MSE)
  fig_bias_scatter.png   -> 図 (観測バイアス vs m3/Var の入力別散布図, 相関つき)
  table_readout.md       -> 数表 (seed 別最終 MSE)
  results.json

実行例:
  python tmp/fncl5_5.py                # 既定: H=64, T=64, 1500 epochs, seeds 0,1,2
  python tmp/fncl5_5.py --no-include-sgd
  python tmp/fncl5_5.py --quick
"""
import argparse

import numpy as np
import torch

from fncl_common import (add_common_args, finalize_args, make_task,
                         model_factory, run_matrix, mse_table_md, config_dict,
                         write_text, save_json, fncl)
from fncl5_5_fig import fig_drift, fig_bias_scatter, plot_readout_composite


def bias_verification(args, device, passes: int = 200):
    """(b) 観測バイアス vs 予測 m3/Var を入力ごとに測る.

    cov_jac (adam) を warmup した後のネットワーク (収束付近, ドリフトが問題に
    なる領域) で、E[g_y] を passes 回の独立 forward で推定し、真値 2(E[y]-t)
    との差を予測項 m3/Var と比較する。返り値は per-input の numpy 配列 dict."""
    torch.manual_seed(args.seed_list[0])
    np.random.seed(args.seed_list[0])
    x_raw, target, x, t = make_task(device)
    net = model_factory(args.noise, args, device)()
    fncl.train_cov(net, x, t, args.noise, args.sigma, args.radius, "cov_jac",
                   args.lr, args.warmup_epochs, opt="adam", jac_track=True,
                   jac_ema=args.jac_ema)

    cap = fncl.Capture(net)
    g_est, ys_all = [], []
    with torch.no_grad():
        for _ in range(passes):
            net(x)
            ys = cap.y_samples.squeeze(-1)                   # [N, T]
            L = (ys - t) ** 2                                # [N, T]
            cy = ys - ys.mean(dim=1, keepdim=True)
            cL = L - L.mean(dim=1, keepdim=True)
            g = (cL * cy).mean(dim=1) / ((cy ** 2).mean(dim=1) + fncl.EPS)
            g_est.append(g.cpu().numpy())
            ys_all.append(ys)
        ys_cat = torch.cat(ys_all, dim=1)                    # [N, passes*T]
        mu = ys_cat.mean(dim=1)
        eps = ys_cat - mu.unsqueeze(1)
        var = (eps ** 2).mean(dim=1)
        m3 = (eps ** 3).mean(dim=1)
    cap.remove()
    true_g = (2.0 * (mu - t.squeeze(1))).cpu().numpy()       # 真の 2(E[y]-t)
    bias = np.mean(g_est, axis=0) - true_g                   # 観測バイアス [N]
    skew_term = (m3 / (var + fncl.EPS)).cpu().numpy()        # 予測 m3/Var  [N]
    corr = float(np.corrcoef(bias, skew_term)[0, 1])
    stats = {
        "passes": passes,
        "rms_true_signal": float(np.sqrt(np.mean(true_g ** 2))),
        "rms_observed_bias": float(np.sqrt(np.mean(bias ** 2))),
        "rms_predicted_m3_over_var": float(np.sqrt(np.mean(skew_term ** 2))),
        "corr_bias_vs_m3_over_var": corr,
    }
    print("  bias verification: "
          f"rms(true 2(mu-t))={stats['rms_true_signal']:.4f}  "
          f"rms(bias)={stats['rms_observed_bias']:.4f}  "
          f"rms(m3/Var)={stats['rms_predicted_m3_over_var']:.4f}  "
          f"corr={corr:.3f}", flush=True)
    return {"bias": bias, "skew_term": skew_term, "stats": stats}


# 図の描画関数 (fig_drift / fig_bias_scatter) は fncl5_5_fig.py に一本化。


def main() -> None:
    p = argparse.ArgumentParser(
        description="§5.5 readout error from covariance (cov_jac_full): "
                    "skewness bias, its Adam-amplified drift, and the fixes.")
    add_common_args(p)
    p.add_argument("--noise", choices=("gaussian", "uniform"), default="gaussian")
    p.add_argument("--include-sgd", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="SGD 変種 3 種も行列に追加する (既定 ON; 本文 §5.5 が"
                        "参照するため。--no-include-sgd で省略)")
    p.add_argument("--warmup-epochs", type=int, default=300,
                   help="(b) バイアス検証前の cov_jac(adam) 学習エポック数")
    p.add_argument("--bias-passes", type=int, default=200,
                   help="(b) E[g_y] 推定に使う独立 forward パス数")
    args = finalize_args(p.parse_args(), "out/fncl5_5")
    if args.quick:
        args.warmup_epochs = min(args.warmup_epochs, 30)
        args.bias_passes = min(args.bias_passes, 20)
    device = torch.device(args.device)

    # ---- (a) 読み出し誤差推定量の比較行列 ----
    methods = [
        ("backprop",         {"kind": "backprop"}),
        ("cov_jac_adam",     {"method": "cov_jac", "opt": "adam",
                              "jac_track": True}),
        ("full_cov_adam",    {"method": "cov_jac_full", "opt": "adam",
                              "jac_track": True, "jac_out": "cov"}),
        ("full_cov_m3_adam", {"method": "cov_jac_full", "opt": "adam",
                              "jac_track": True, "jac_out": "cov_m3"}),
        ("full_probe_adam",  {"method": "cov_jac_full", "opt": "adam",
                              "jac_track": True, "jac_out": "probe"}),
    ]
    if args.include_sgd:
        methods += [
            ("cov_jac_sgd",     {"method": "cov_jac", "opt": "sgd",
                                 "jac_track": True}),
            ("full_cov_sgd",    {"method": "cov_jac_full", "opt": "sgd",
                                 "jac_track": True, "jac_out": "cov"}),
            ("full_cov_m3_sgd", {"method": "cov_jac_full", "opt": "sgd",
                                 "jac_track": True, "jac_out": "cov_m3"}),
            ("full_probe_sgd",  {"method": "cov_jac_full", "opt": "sgd",
                                 "jac_track": True, "jac_out": "probe"}),
        ]
    log_every = max(1, args.epochs // 5)
    mse, curves, preds, target, x_raw = run_matrix(
        methods, args.noise, args, log_every)

    out = args.out_dir
    caption = (f"§5.5 readout-error estimators, noise={args.noise}, "
               f"H={args.hidden_dim}, T={args.num_samples}, "
               f"epochs={args.epochs}, lr={args.lr} (Adam rows)")
    table = mse_table_md(mse, args.seed_list, caption)
    print("\n" + table)
    write_text(out / "table_readout.md", table)
    fig_drift(curves, out / "fig_readout_drift.png")

    # ---- (b) 歪度バイアスの直接検証 ----
    print(f"\n[bias verification] cov_jac(adam) warmup {args.warmup_epochs} ep, "
          f"{args.bias_passes} passes")
    bv = bias_verification(args, device, passes=args.bias_passes)
    fig_bias_scatter(bv["bias"], bv["skew_term"],
                     bv["stats"]["corr_bias_vs_m3_over_var"],
                     out / "fig_bias_scatter.png")

    np.savez(out / "fig_data.npz",
             corr=bv["stats"]["corr_bias_vs_m3_over_var"],
             bias=bv["bias"], skew_term=bv["skew_term"],
             **{f"curve_{k}": np.asarray(v) for k, v in curves.items()})
    print(f"  saved {out / 'fig_data.npz'}")
    plot_readout_composite(curves, bv["bias"], bv["skew_term"],
                           bv["stats"]["corr_bias_vs_m3_over_var"],
                           out / "fig_readout.png")
    save_json(out / "results.json",
              {"config": config_dict(args), "noise": args.noise,
               "final_mse": mse, "bias_verification": bv["stats"]})


if __name__ == "__main__":
    main()
