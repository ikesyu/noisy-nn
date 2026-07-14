"""
fncl6_2.py — 論文 §6.2「一様ノイズの決定的利点と実験的裏付け」の実験部分

§5.2 と同一プロトコル (検証 6 手法 x 複数 seed) を --noise uniform
(SimpleNNNUniformSample) で実行し、主結果の 2 点

  (i)  cov_jac_adam が backprop 水準に達すること,
  (ii) 分布フリー kde slope が解析 phi' (一様: -(d-c)/r^2) と一致すること
       (= 一様ノイズの解析式を使わずに同じ精度)

が一様ノイズでも成り立つことを確認する。§6 のデジタル HW 主張の実験的裏付け。

生成物 (out/fncl6_2/):
  (標準出力)              -> Tab.1 の uniform 行の最終 MSE 表
  fig_learning_curves.png -> 補助図 (uniform 学習曲線)
  fig_fit_check.png       -> 補助図 (uniform fit check)
  fig_predictions.png
  results.json

実行例:
  python data_nce/fncl6_2.py
  python data_nce/fncl6_2.py --radius 1.0 --quick
"""
import argparse

from fncl_common import add_common_args, finalize_args, run_verification


def main() -> None:
    p = argparse.ArgumentParser(
        description="§6.2 uniform-noise reproduction of the main result.")
    add_common_args(p)
    args = finalize_args(p.parse_args(), default_out="out/fncl6_2")
    run_verification("uniform", args)


if __name__ == "__main__":
    main()
