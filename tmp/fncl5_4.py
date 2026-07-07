"""
fncl5_4.py — 論文 §5.4「アブレーション」(Fig.6)

同一 seed 集合・同一初期重みで、以下の対比を最終 MSE で比較する:

  1. credit の中心化:      cov_deriv_kde per_input vs pooled  (決定的な差)
  2. 局所傾き:             cov_deriv kde vs analytic          (一致 -> 解析 phi' 不要)
  3. 傾き因子の寄与:       cov_only vs cov_deriv_kde
  4. mirror 追跡:          cov_jac_adam track vs no-track     (Kolen-Pollack の効果)
  5. 局所オプティマイザ:   cov_jac sgd vs adam / cov_deriv に adam (逆効果の確認)

生成物 (out/fncl5_4/):
  fig_ablation_bar.png   -> Fig.6 (最終 MSE 棒グラフ, log, seed 誤差棒)
  table_ablation.md      -> 数表
  results.json

実行例:
  python tmp/fncl5_4.py
  python tmp/fncl5_4.py --seeds 0,1,2,3,4
  python tmp/fncl5_4.py --quick
"""
import argparse

from fncl_common import (add_common_args, finalize_args, run_matrix,
                         mse_table_md, bar_mse, write_text, save_json,
                         config_dict)

ABLATION_METHODS = [
    # 参照
    ("backprop",                {"kind": "backprop"}),
    # 1. credit の中心化
    ("cov_deriv_kde/per_input", {"method": "cov_deriv", "slope": "kde",
                                 "credit": "per_input"}),
    ("cov_deriv_kde/pooled",    {"method": "cov_deriv", "slope": "kde",
                                 "credit": "pooled"}),
    # 2. 局所傾き (kde vs analytic)
    ("cov_deriv_analytic",      {"method": "cov_deriv", "slope": "analytic"}),
    # 3. 傾き因子なし
    ("cov_only",                {"method": "cov_only"}),
    # 5. cov_deriv に adam (分散増幅で悪化することの確認)
    ("cov_deriv_kde/adam",      {"method": "cov_deriv", "slope": "kde",
                                 "opt": "adam"}),
    # 4+5. cov_jac: track/no-track x sgd/adam
    ("cov_jac_sgd/track",       {"method": "cov_jac", "opt": "sgd",
                                 "jac_track": True}),
    ("cov_jac_adam/track",      {"method": "cov_jac", "opt": "adam",
                                 "jac_track": True}),
    ("cov_jac_adam/no-track",   {"method": "cov_jac", "opt": "adam",
                                 "jac_track": False}),
]


def main() -> None:
    p = argparse.ArgumentParser(
        description="§5.4 ablations: credit centring / slope / optimiser / "
                    "mirror tracking.")
    add_common_args(p)
    p.add_argument("--noise", choices=("gaussian", "uniform"), default="gaussian")
    args = finalize_args(p.parse_args(), default_out="out/fncl5_4")

    log_every = max(1, args.epochs // 5)
    mse, curves, preds, target, x_raw = run_matrix(
        ABLATION_METHODS, args.noise, args, log_every)

    caption = (f"Ablations, noise={args.noise}, H={args.hidden_dim}, "
               f"T={args.num_samples}, epochs={args.epochs}")
    table = mse_table_md(mse, args.seed_list, caption)
    print("\n" + table)
    write_text(args.out_dir / "table_ablation.md", table)
    save_json(args.out_dir / "results.json",
              {"config": config_dict(args), "final_mse": mse})
    bar_mse(mse, args.seed_list, "Ablations (final MSE)",
            args.out_dir / "fig_ablation_bar.png")


if __name__ == "__main__":
    main()
