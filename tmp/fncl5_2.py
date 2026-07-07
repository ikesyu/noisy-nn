"""
fncl5_2.py — 論文 §5.2「主結果 — cov_jac_adam は backprop 級」(ガウスノイズ)

検証 6 手法 (backprop / cov_only / cov_deriv_analytic / cov_deriv_kde /
cov_jac_sgd / cov_jac_adam) を複数 seed で学習し、以下を生成する:

  out/fncl5_2/table_final_mse.md      -> Tab.1 (最終 MSE, seed 別 + mean±std)
  out/fncl5_2/fig_learning_curves.png -> Fig.3 (学習曲線, 先頭 seed)
  out/fncl5_2/fig_fit_check.png       -> Fig.4 (fit check: target vs backprop
                                          vs cov_jac_adam + 残差)
  out/fncl5_2/fig_predictions.png     -> 補助図 (全手法の予測)
  out/fncl5_2/results.json            -> 数値一式 (config + final MSE)

実行例:
  python tmp/fncl5_2.py                 # 3 seeds x 1500 epochs (時間がかかる)
  python tmp/fncl5_2.py --seeds 0,1,2,3,4
  python tmp/fncl5_2.py --quick         # 動作確認
"""
import argparse

from fncl_common import add_common_args, finalize_args, run_verification


def main() -> None:
    p = argparse.ArgumentParser(
        description="§5.2 main result: cov_jac_adam reaches backprop level "
                    "(gaussian noise).")
    add_common_args(p)
    p.add_argument("--noise", choices=("gaussian", "uniform"), default="gaussian")
    args = finalize_args(p.parse_args(), default_out="out/fncl5_2")
    run_verification(args.noise, args)


if __name__ == "__main__":
    main()
