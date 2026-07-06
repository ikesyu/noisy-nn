# Recipe: 確率共鳴（SR）の単峰カーブで「ノイズ場＝機能的資源」を示す

**目的（一言で）**: 神経修飾物質様のノイズ場が単なる“鍵（look-up key）”ではなく、
**最適強度をもつ機能的資源**であることを、確率共鳴（stochastic resonance, SR）の
**逆U字（単峰）カーブ**で示す。ルックアップ鍵には最適強度が存在しないため、内点に
最適ノイズ強度が現れれば、この批判に対する直接的な反証になる。

対象コード: [`examples/sr_separation_curve.py`](../examples/sr_separation_curve.py)
（挙動デモ [`examples/neuromodulated_behavior_modes.py`](../examples/neuromodulated_behavior_modes.py)
の学習・データ・ノイズ場をそのまま再利用する診断スクリプト）。

---

## 1. 前提

- 依存: `numpy`, `torch`, `matplotlib` のみ。
- 出力:
  - 既定は `plt.show()` で図を表示。
  - **`--save PATH`** で掃引データを **CSV 保存**（メタデータは先頭の `#` コメント行、
    データは純数値：列順 `x, separation, signal, task_err`）。
  - **`--no-show`** で図ウィンドウを開かない（`--save` と併用してヘッドレス／バッチ実行）。
  - 読み戻し: `numpy.loadtxt(PATH, delimiter=',')`（`#` 行は自動でスキップ）。
- 実行はリポジトリ直下から。CPU 既定。

環境確認（数十秒の最小実行）:

```bash
python examples/sr_separation_curve.py --epochs 60 --grid-side 13 --s-steps 9
```

---

## 2. 何を測るか（軸と指標）

- **横軸 = ノイズ強度**: 動員された隠れユニットのピーク std（＝神経修飾物質“濃度”の代理）。
  CSV では `--sweep test` なら列名 `test_s`、`--sweep train` なら `sigma_train`。
- **縦軸（3指標、CSV の 2〜4 列）**:
  - `separation` — 3つのノイズ場が生む出力ベクトル場どうしの平均ペア距離（＝**候補2**、
    「各場がどれだけ別々の行動を動員するか」）。
  - `signal` — 出力の**刺激依存成分**（観測ごとの変動、各場の平均を引いた量）。SRで単峰に
    なるべき本命量：`→0`（ユニット未動員）と過大（交差飽和で定数化）で 0 に向かう。
  - `task_err` — 学習目標（近似 one-hot 行動場）への MSE（低いほど良、参照用）。

内点にピーク（`signal`/`separation`）または内点に谷（`task_err`）が出れば SR の単峰。

---

## 3. 2つのスイープ × 3つのモデル（設計）

### スイープ `--sweep`
- `test`（既定）: 1つのネットを `--base-std` で学習し、**テスト時のノイズ強度**を掃引。
  高速だが、学習水準の近傍で `task_err` が最小になるのは構造上当然（**交絡あり**）。
- `train`（**厳密版**）: 各 std ごとに**新規ネットを学習し、その std で採点**。全点が
  「自分の水準で学習・評価」なので、内点最適は**真に最適なノイズ水準**を意味する（交絡なし）。
  1水準あたり1回の学習コスト。

### モデル `--model`
- `analytic`（既定・高速）: ノイズ交差の**期待値**（決定論、閾値 `h` なし）＝平均場近似。
- `sample`: **実ノイズ注入**＋`t` サンプル平均、閾値 `h>0` ＝**本物のSR機構**。やや遅い。
- `statistic`: サンプル応答のモーメント推定（`h`, `t` を使用）。

> **実測済みの重要な区別**: `--sweep train` では、
> - `--model analytic` は**低σ端で最適**（低σでも重み再スケールで学習でき、真のSR障壁なし＝平均場）。
> - `--model sample` は**内点最適**（σ\*≈0.8–1.2）で、低σ側が実際に崩壊（閾値下は伝達不能）。
> → **論文の“厳密なSR図”は `--sweep train --model sample`**、`analytic` はその対照。

---

## 4. データ生成コマンド（保存つき）

`--save` で CSV を残し、後で作図（§6）に使う。バッチは `--no-show` を付ける。
ここでは `data/` にファイル名規約 `sr_<sweep>_<model>_seed<seed>.csv` で保存する。

### R1. 厳密なSR（本命）— sample × 学習水準スイープ
```bash
python examples/sr_separation_curve.py \
    --sweep train --model sample \
    --grid-side 25 --epochs 1500 --train-steps 15 \
    --s-min 0.1 --s-max 4.0 --samples 96 --crossing-h 0.2 --seed 7 \
    --save data/sr_train_sample_seed7.csv --no-show
```
（重い：sample ネットを15個学習。まず軽量プレビュー: `--grid-side 17 --epochs 700 --train-steps 9 --samples 32`）

### R2. 対照（平均場＝SRなし）— analytic × 学習水準スイープ
```bash
python examples/sr_separation_curve.py \
    --sweep train --model analytic \
    --train-steps 15 --s-min 0.1 --s-max 4.0 --seed 7 \
    --save data/sr_train_analytic_seed7.csv --no-show
```

### R3. テスト時スイープ（高速・逆U字の補助図）
```bash
# 平均場（既定・最速）
python examples/sr_separation_curve.py --s-max 4.0 --s-steps 60 --seed 7 \
    --save data/sr_test_analytic_seed7.csv --no-show
# 機構（sample）
python examples/sr_separation_curve.py --model sample \
    --grid-side 21 --epochs 1500 --samples 96 --s-max 4.0 --s-steps 60 --seed 7 \
    --save data/sr_test_sample_seed7.csv --no-show
```

### R4. seed 頑健性（複数シード）— R1 を seed 掃引
```bash
for s in 0 1 2 3 4; do
  python examples/sr_separation_curve.py --sweep train --model sample \
    --grid-side 21 --epochs 1000 --train-steps 11 --s-min 0.1 --s-max 4.0 --samples 64 \
    --seed $s --save data/sr_train_sample_seed$s.csv --no-show
done
```

---

## 5. 標準出力の読み方

```
train-sweep summary (sample):
  stimulus-locked signal peaks at 1.190  (INTERIOR optimum -> SR confirmed)
  signal:   ends=0.063 / 0.790, peak=0.803
  separation at optimum = 1.430
  task err: ends=0.414 / 0.030, min=0.014
  saved curve data -> .../data/sr_train_sample_seed7.csv
```
- `INTERIOR optimum -> SR confirmed` … signal のピークが端点でない＝SR成立。
- `at an endpoint` … 端点最適（analytic の train スイープで想定どおり）。
- `saved curve data -> ...` … `--save` 指定時の保存先。

---

## 6. 保存データからの作図（どれとどれで何を描くか）

各 CSV は列順 `x, separation, signal, task_err`（1列目名は `sigma_train` or `test_s`）。

| 図 | 使う CSV | 描く列（縦軸） vs 横軸 | 主張 |
|---|---|---|---|
| **Fig 1 厳密なSR** | `sr_train_sample_seed7` | `task_err` と `signal` vs `sigma_train` | 内点最適＋低σ崩壊＝真のSR |
| **Fig 2 機構 vs 平均場** | `sr_train_sample_*` ＋ `sr_train_analytic_*` | `signal` vs std を**重ね描き** | sample=内点、analytic=低σ端（障壁なし） |
| **Fig 3 候補2：場の分離度** | 任意の train/test CSV | `separation` vs std | ノイズ強度が場の“区別度”を決め最適点をもつ |
| **Fig 4 テスト時の逆U字** | `sr_test_sample_*`（or analytic） | `separation` と `signal` vs `test_s` | 1ネットでの逆U字（安価な補助図） |
| **Fig 5 seed 頑健性** | `sr_train_sample_seed{0..4}` | `task_err` の **mean±std** vs `sigma_train` | 傾向が seed に依存しない |

### 最小の作図スクリプト例（numpy + matplotlib）
```python
import glob
import numpy as np
import matplotlib.pyplot as plt

def load(path):                     # 列: x, separation, signal, task_err
    d = np.loadtxt(path, delimiter=',')
    return d[:, 0], d[:, 1], d[:, 2], d[:, 3]

# --- Fig 2: 機構(sample) vs 平均場(analytic) : signal vs 学習std を重ね描き ---
xs, _, sig_s, _ = load("data/sr_train_sample_seed7.csv")
xa, _, sig_a, _ = load("data/sr_train_analytic_seed7.csv")
plt.figure()
plt.plot(xs, sig_s, "-o", label="sample (mechanism)")
plt.plot(xa, sig_a, "-s", label="analytic (mean-field)")
plt.xlabel("training noise std"); plt.ylabel("stimulus-locked signal")
plt.legend(); plt.title("Fig 2: mechanism vs mean-field")

# --- Fig 5: seed 頑健性 : task_err の mean±std ---
files = sorted(glob.glob("data/sr_train_sample_seed*.csv"))
cols = [load(f) for f in files]
x = cols[0][0]
err = np.array([c[3] for c in cols])            # [n_seed, n_std]
m, s = err.mean(0), err.std(0)
plt.figure()
plt.plot(x, m, "-o"); plt.fill_between(x, m - s, m + s, alpha=0.3)
plt.xlabel("training noise std"); plt.ylabel("task error (mean ± std)")
plt.title("Fig 5: seed robustness")
plt.show()
```

> スクリプトは論文の作図系（matplotlib / pgfplots / R など）に合わせて自由に。CSV は
> 純数値なので `pandas.read_csv(path, comment='#', header=None,
> names=['x','separation','signal','task_err'])` でも読める。

---

## 7. 再現性・頑健性

- `--seed` で全体を固定（既定 7）。
- 論文用途では **複数 seed（R4）** で内点最適・低σ崩壊が seed に依存しないことを示す（Fig 5）。

---

## 8. 主なパラメータ（抜粋）

| 引数 | 既定 | 意味 / 使いどころ |
|---|---|---|
| `--sweep` | `test` | `test`=テスト時掃引 / `train`=学習水準掃引（厳密） |
| `--model` | `analytic` | `analytic`(平均場) / `sample`(真のSR) / `statistic` |
| `--train-steps` | 13 | 学習水準スイープの点数（train のみ） |
| `--crossing-h` | 0.2 | 交差閾値（sample/statistic）。**SRには `h>0` 必須** |
| `--samples` | 64 | モンテカルロ標本数 `t`。大きいほど滑らか・遅い |
| `--epochs` | 3000 | 1ネットの学習エポック |
| `--grid-side` | 31 | 学習グリッド解像度（大きいほど重い） |
| `--base-std` | 0.8 | test スイープでの学習水準 |
| `--sigma` / `--theta` | 0.22 / 0.15 | ノイズバンプ幅 / 強度切り捨て閾値 |
| `--s-min`/`--s-max`/`--s-steps` | 0.0/3.0/41 | 掃引範囲・点数（横軸） |
| `--seed` | 7 | 乱数シード |
| `--save` | なし | 掃引データを CSV 保存 |
| `--no-show` | off | 図ウィンドウを開かない（バッチ用） |

---

## 9. 既知の注意点・今後の拡張（TODO）

- **交絡**: test スイープの `task_err` は学習水準で最小（構造上当然）。主張は
  `separation/signal`（非学習目標）の内点最適、または `--sweep train` で行う。
- **高σ側の裾**: `--s-max 3.0` だと減衰が穏やか。より明瞭な逆U字の裾には `--s-max 5.0` 程度。
- **Analytic ≈ E[Sample] の重ね描き**: 「平均場 vs 機構」を1枚に。現状は各モデルを別々に
  学習するため厳密な同一重み比較ではない。同一重みで両応答を評価する専用診断を足すとより厳密。
- **行動レベルのSR（候補3）**: 閉ループの採餌効率・到達時間などを `s` の関数にし、
  「行動能力そのもの」に最適ノイズがあることを示す図（特集の主眼に最も近い）。

---

## 10. 参考

- 背景と主張の整理: 過去の議論メモ（SR文脈・多重化・ルックアップ鍵批判への回答）。
- SR の理論的土台: NNN の既発表（Neurocomputing 2報、確率共鳴に基づく NN）をイントロで回収する。
