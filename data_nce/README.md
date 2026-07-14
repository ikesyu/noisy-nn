# data_nce/ — NCE 論文 (docs/NCE_NNN_CA.pdf) の結果再現コード

論文「Reconstructing Backpropagation from Forward Fluctuations in
Noise-modulated Neural Networks」の図表を再現する実験スクリプト一式。
すべてプロジェクトルートから実行する（`nnn/` パッケージと同じ階層を前提）:

```bash
python data_nce/fncl5_2.py            # 本番 (3 seeds × 1500 epochs)
python data_nce/fncl5_2.py --quick    # 縮小設定での動作確認
```

図と数値データ（results.json / *.npz）は既定でカレントディレクトリの
`out/fncl<sec>_<n>/` に保存される（`--out` で変更可）。**表（Table 2〜5 相当の
数表）は標準出力に表示されるのみ**で、ファイルには保存されない。
`*_fig.py` は対応する本体スクリプトの実行済み出力（results.json / *.npz）から
図だけを再生成するもので、学習の再実行は不要。

## スクリプトと論文の対応

| スクリプト | 論文の節 | 生成する図表 |
|---|---|---|
| `fncl5_2.py` (+`fncl5_2_fig.py`) | §5.2 主結果 (ガウスノイズ) | Table 2, Fig. 2 |
| `fncl5_3.py` (+`fncl5_3_fig.py`) | §5.3 勾配再構成の忠実度 | Fig. 3 |
| `fncl5_4.py` (+`fncl5_4_fig.py`) | §5.4 アブレーション | Fig. 4 |
| `fncl5_5.py` (+`fncl5_5_fig.py`) | §5.5 読み出し誤差の共分散推定・歪度バイアス | Fig. 5 |
| `fncl5_6.py` | §5.6 ベンチマーク (Friedman #1 / Two Moons / Circles) | Table 3 |
| `fncl6_1.py` | §6.1 HW リソース見積もり（標準出力のみ） | Table 4 |
| `fncl6_2.py` (+`fncl5_2_fig.py --data out/fncl6_2`) | §6.2 一様ノイズでの再現 | Table 5 |

## 内部構成

- `fncl/` — 全学習則 (backprop / cov_only / cov_deriv / cov_jac / cov_jac_full)
  の実装本体パッケージ。`nnn/` パッケージに依存（プロジェクトルートを
  sys.path に追加して解決）。
  - `constants.py` — 共有数値定数 (EPS, NUM_POINTS など)
  - `network.py` — モデル構築 (`build_model`)，局所微分 (`phi_prime` /
    `kde_slope`)，forward フック計測 (`Capture`)
  - `perturb.py` — 外部摂動ゲート（`Perturber` など; Appendix B 用）
  - `train.py` — 学習則本体 (`train_cov`, `covariance_credit`, `cov_weight`,
    `ManualOpt`) と backprop 参照 (`train_backprop`)，推論 (`predict`)
  - `viz.py` — 確認用プロット
- `fncl_common.py` — 実験行列 (seed × 手法) の実行、表・JSON・図の保存など、
  各 `fncl5_*.py` / `fncl6_*.py` が共有する薄い基盤。

論文の図表生成から独立した全手法比較の PoC ランナーは
`tmp/forward_noise_covariance_learning.py`（fncl パッケージを import して動く）。

各 (seed, 手法) の直前に torch / numpy を seed し直すため、同一 seed の
全手法は同一の初期重み・同一の乱数系列から開始する。論文の数値は
seeds 0,1,2・H=64・T=64・1500 epochs（各スクリプトの既定値）で得たもの。
