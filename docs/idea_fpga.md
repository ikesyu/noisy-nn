# NNN のデジタル回路（FPGA）実装 — 検討・実験ログ

NCE 投稿論文（`docs/nce_draft.md`, cov_jac / cov_jac_full）の提案手法を FPGA に
実装し、**学習と推論の両方がオンチップで動く**ことを実証する計画の記録。
最終目標は Lattice Semiconductor 製品をロボットに組み込んだ運動学習デモ。

関連: `docs/nce_draft.md` §6（資源分析）、`docs/idea_duality.md`（続編構想。
§6 の「1 つの場が推論・学習・電力を同時にゲートする」は方向 C として本計画と接続）。

- 2026-07-15 計画立案、Phase 0 完了（両実験 PASS）
- 2026-07-16 Phase 1 着手

---

## 1. 全体工程

| Phase | 内容 | 状態 |
|---|---|---|
| 0 | **アルゴリズム仕様の凍結**（SGD 化・オンライン化の 2 実験） | **完了（PASS）** |
| 1 | **固定小数点ゴールデンモデル**（ストリーミング統計・LFSR ノイズ・ビット幅探索） | **完了（PASS）** |
| 2 | アーキテクチャ設計（演算器共有・メモリマップ・制御 FSM） | **完了** |
| 3 | RTL 実装（Amaranth + cocotb / Verilator でゴールデンモデルと突き合わせ） | 進行中 |
| 4 | 評価ボードで論文の 1 次元回帰をオンチップ学習 | — |
| 5 | ロボット統合（センサ/アクチュエータ IF、運動学習デモ） | — |

デバイス候補: 学習込みプロトタイプは **ECP5**（オープンフロー Yosys/nextpnr）
または **CertusPro-NX**（Radiant、低消費電力）。推論専用に縮退させるなら
iCE40 UP5K。

### 出発点: 論文 §6 の資源分析

- 学習に要る統計量は running sum 4 種（Σz, Σz², Σd, Σdz）+ 除算のみ。
  cov_jac の除算は ~4.2k 回/更新、重みはミラー込み 2 セット ~8.4k words。
- z ∈ {0, ½, 1}（2 ビット）なので MAC はバイナリゲート付き加算に落ちる。
- 一様ノイズなら乱数源は LFSR のみ、期待応答は放物線（コンパレータ+多項式）。

ただし §6 の分析は「Adam を使わない」「バッチ統計をストリーミングに置換できる」
ことが前提。これを検証するのが Phase 0 だった。

---

## 2. Phase 0 実験①: SGD 化（`tmp/fncl_phase0_sgd.py`）

### なぜ Adam ではダメか

Adam はパラメータ毎に sqrt + 除算 + 状態 2 語 + 1/√v̂ の広いダイナミックレンジを
要求し、§6.1 の「累算・積和・ユニットあたり 1 除算」という主張を壊す。
古典的 momentum（sgdm）なら状態 1 語 + 乗算 1 回で、μ = 1−2⁻⁴ = 0.9375 は
`v − (v>>4) + grad` とシフトだけで実装できる。

### 結果（uniform ノイズ, H=64, T=64, 1500 epochs, 3 seeds）

| 構成 | 最終 MSE | vs backprop |
|---|---|---|
| backprop（Adam 参照） | 0.00058 | ×1.00 |
| cov_jac + Adam（論文構成） | 0.00045 | ×0.79 |
| **cov_jac + sgdm(0.9375) + cosine + lr 0.125** | **0.00092 ± 0.00021** | **×1.60 → PASS** |
| cov_jac + 素の SGD（最良 lr） | — | ×6.7（不合格） |

- **momentum が必須**。素の SGD はどの lr でも ×6.7 止まり。
- lr プラトーは狭い: 0.15 で不安定、0.2 以上は未学習水準（~0.496）へ発散。
- 定数は全て 2 の冪で成立: lr = 2⁻³、μ = 1−2⁻⁴。
- lr 減衰は必要。完全シフト実装の step 減衰（epochs/8 毎に半減）は ×2.09 で
  僅かに基準超え → **HW は cosine をサンプルした小さな ROM テーブル**で持つ。

**凍結仕様**: cov_jac + sgdm(1−2⁻⁴) + lr 2⁻³ + cosine ROM + jac_track（KP 追跡）
+ 一様ノイズ + per_input credit + KDE slope。

---

## 3. Phase 0 実験②: オンライン化（`tmp/fncl_phase0_online.py`）

現行 `train_cov` は全入力 N=128 の一括バッチ。ロボットでは入力は 1 個ずつ
時間相関を持って流れる（軌道提示）。総提示回数 192,000 を揃えて比較した。

### 結果（凍結 lr 0.125 のまま）

| config | 最終 MSE | vs batch_ref | mirror r (out / W1) |
|---|---|---|---|
| batch_ref | 0.00092 ± 0.00021 | ×1.00 | — |
| B32_iid | 0.00075 | ×0.81 | 0.775 / 0.994 |
| B8_iid | 0.08353 | ×90.5 | 0.327 / 0.871 |
| B1_iid | 0.49611 | ×537.6 | 0.359 / 0.743 |
| B1_traj | 0.23995 | ×260.0 | 0.270 / 0.773 |
| B32_traj | 0.44397 | ×481.1 | 0.526 / 0.758 |

素朴な B=1 は全滅。原因はオンライン性ではなく**更新分散が 128 倍**になること
（readout mirror の相関 r ごと崩壊する）。

### 結果（lr 線形則: lr = 2⁻³ × B/128）

| config | 最終 MSE | vs batch_ref | mirror r (out / W1) |
|---|---|---|---|
| **B1_iid** | 0.00101 | **×1.09** | 0.991 / 0.975 |
| **B1_traj** | 0.00092 | **×1.00** | 0.969 / 0.965 |
| B8_traj | 0.00287 | ×3.11 | 0.708 / 0.941 |
| B32_traj | 0.01383 | ×15.0 | 0.295 / 0.895 |

**完全オンライン N=1 は lr 線形則だけで成立（PASS）**。軌道（時間相関）提示でも
×1.00。mirror EMA はステップ頻度に合わせ `jac_ema^(B/N)` に自動スケール、
KP 追跡は「適用ステップぶんミラーをずらす」だけなので粒度によらず厳密。

一方、**軌道提示の部分累算窓（B=8, 32）は最悪の中間領域**。窓がグリッドの
局所（B=32 で入力空間の 1/4）に偏った勾配を、窓ぶん大きい lr で踏むのが
二重に悪い。窓=128 は全グリッド一巡と等価なのでバッチに一致し、B=1 は
小さい lr が掃引全体を時間平均する。両端だけが安全。

### FPGA 学習コアの更新粒度仕様（凍結）

- **提示ごと更新（B=1）**。相関入力での部分累算窓は使わない。
- lr = 2⁻³ × 1/128 = **2⁻¹⁰**（バッチ比でシフトを 7 段深くするだけ）。
- mirror EMA 率 β_step = jac_ema^(1/128) = 0.9^(1/128) ≈ 0.99918
  （HW では 1−2⁻ᵏ に丸める。k=10: 0.99902 / k=11: 0.99951 — Phase 1 で決定）。
- sgdm μ = 1−2⁻⁴、cosine は総ステップ数にわたる ROM テーブル。

---

## 4. Phase 1: 固定小数点ゴールデンモデル（進行中）

**目的**: Phase 0 で凍結した B=1 オンライン構成を、RTL にそのまま写せる形
（ストリーミング統計・LFSR 乱数・整数演算）で再現し、ビット幅と除算戦略を
決める。実装は `tmp/fncl_phase1_fxp.py`、結果は `tmp/out/fncl_phase1_*/`。

**判定基準**: LFSR + 固定小数点の B=1 学習が float の batch_ref
（0.00092）の ×2 以内、mirror r が崩壊しないこと。

### Stage A — ストリーミング統計への書き換え（float のまま）

[N,T,H] テンソルを保持する現行実装を、提示ごとに T サンプルを流しながら
running sum（Σz, Σz², Σd, Σd·z, xor カウント）だけで学習する形に書き換え、
Phase 0 の B1 結果と一致することを確認する。T = 64 = 2⁶ なので

    Ŵ_ji = Cov_t(d_j, z_i) / Var_t(z_i) = (T·Σdz − Σd·Σz) / (T·Σz² − (Σz)²)

は**整数（固定小数点）の比**になり、1/T は全てシフト。除算はユニット i あたり
逆数 1 回 + 乗算に落ちる。あわせて KDE slope を「同一 forward の xor カウント
から取る」形（現行 `kde_slope` はノイズを引き直す再パス）に変え、追加パスを
1 本削る。z の交差出力は {0, ½, 1} の 2 ビットなので MAC は §6.1 の想定通り
バイナリゲート付き加算。

### Stage B — LFSR ノイズへの置換

一様ノイズ Uniform(−1, 1) を LFSR 由来の k ビット一様整数に置き換える。
検証する構成:

- `phase`: 全ユニットが同一多項式の m 系列を異なる位相（シード）で共有
  （HW: ユニットごとに小さな LFSR、シードだけ変える）
- `leap` : 1 本の LFSR を時分割で全ユニットに配る（HW 最小構成: LFSR 1 個）
- ビット幅: 16 bit（周期 65535）と 24 bit（周期 16.7M）

リスクは cov_weight のユニット間独立性仮定が系列間相関で壊れること、および
周期の短さによるノイズ再利用（16 bit + leap では H·T=4096 消費/層/提示 →
16 提示で一周）。mirror r と最終 MSE で判定する。

### Stage C — 固定小数点化とビット幅探索

全データパスを整数演算（シフト+加減算+ユニットあたり 1 除算）で実装し、

- 重み・ミラーの小数部ビット Fw（10 / 12 / 14 / 16）
- 前活性 d の小数部ビット Fd（8 / 10 / 12）
- ノイズビット幅（6 / 8 / 12）
- mirror EMA のシフト定数 k と蓄積ガードビット
  （シフト切り捨てによる EMA の不感帯 = 停滞に注意。丸め付きシフトで対策）
- 除算: 整数除算（Num << Fm)/Den + Var 下限クリップ（float 版の eps 相当）

を掃引して、×2 基準を満たす最小ビット幅構成を決める。lr = 2⁻¹⁰ × cosine は
「8 bit ROM 値との乗算 + シフト」で表現する。重みは学習用に広い蓄積幅を持ち、
forward では丸めて使う（勾配が微小なため、更新の underflow が最初の容疑者）。

### 実行報告

#### Stage A（2026-07-16 実施）: **PASS** — ストリーミング化は無損失、EMA はシフト 1 回で凍結

uniform, H=64, T=64, budget=192,000（B=1・軌道提示）, 3 seeds。
比較基準は Phase 0 batch_ref = 0.00092。

| config | 最終 MSE | vs batch_ref | mirror r (out / W1) |
|---|---|---|---|
| stream_exact（EMA 厳密値） | 0.00103 ± 0.00007 | ×1.12 | 0.967 / 0.960 |
| stream_exact_iid | 0.00087 ± 0.00009 | ×0.95 | 0.986 / 0.978 |
| **stream_k10（EMA = 2⁻¹⁰）** | **0.00107 ± 0.00005** | **×1.16** | 0.966 / 0.958 |
| stream_k11（EMA = 2⁻¹¹） | 0.00109 ± 0.00016 | ×1.18 | 0.963 / 0.946 |

- running sum 化・同一パス slope（xor カウント再利用、再パス削除）は
  Phase 0 の B1_traj（×1.00）と同水準。アルゴリズム変更 2 点は無害。
- **mirror EMA は 2⁻¹⁰ シフトで凍結**（厳密値 8.23e-4 との差は誤差の範囲）。

#### Stage B（2026-07-16 実施）: **PASS** — LFSR で成立、24 bit phase が本命

EMA = 2⁻¹⁰ 固定、他は Stage A と同一。

| config | 最終 MSE | vs batch_ref | mirror r (out / W1) |
|---|---|---|---|
| **lfsr_phase24**（ユニット毎 LFSR, 24 bit） | **0.00070 ± 0.00004** | **×0.76** | 0.954 / 0.963 |
| lfsr_leap24（1 本を時分割, 24 bit） | 0.00097 ± 0.00014 | ×1.06 | 0.954 / 0.961 |
| lfsr_leap16 | 0.00102 ± 0.00014 | ×1.11 | 0.956 / 0.957 |
| lfsr_phase16 | 0.00160 ± 0.00026 | ×1.73 | 0.964 / 0.961 |

- **系列間相関は cov_weight を壊さない**（mirror r ≥ 0.95 を全構成が維持）。
  懸念だった独立性仮定はユニット毎の位相差だけで足りる。
- 周期の影響は見える: phase16 は各ユニットが周期 65,535 の系列を約 188 周
  使い回すため ×1.73 まで劣化（それでも基準内）。**24 bit（周期 16.7M、
  1 ユニットの消費 12.3M < 周期）なら劣化なし**。
- HW 選択肢: 品質重視なら **ユニット毎 24 bit LFSR（phase24）**、
  資源最小なら LFSR 1 本の時分割（leap16 でも ×1.11）で良い。

#### Stage C（2026-07-16 実施）: **PASS** — 固定小数点で float を上回る水準

全データパス int64 固定小数点（丸め付きシフト・丸め付き整数除算・9 bit
cosine ROM）、ノイズ源 phase24、EMA 2⁻¹⁰。base = Fw16（重み小数部）/
Fd12（前活性・credit・slope 小数部）/ Nn12（ノイズビット幅）。

| config | 最終 MSE | vs batch_ref | mirror r (out / W1) |
|---|---|---|---|
| **fxp_base**（Fw16/Fd12/Nn12, 3 seeds） | **0.00072 ± 0.00007** | **×0.79** | 0.960 / 0.962 |
| fxp_rand（ノイズだけ torch.rand） | 0.00098 | ×1.07 | 0.971 / 0.953 |
| fxp_fw14 | 0.00088 | ×0.96 | 0.962 / 0.958 |
| fxp_fw12 | 0.00157 | ×1.71 | 0.956 / 0.967 |
| fxp_fd10 | 0.00075 | ×0.82 | 0.948 / 0.962 |
| fxp_fd8 | 0.00070 | ×0.76 | 0.914 / 0.959 |
| fxp_nn8 | 0.00074 | ×0.80 | 0.927 / 0.958 |
| fxp_nn6 | 0.00080 | ×0.87 | 0.948 / 0.963 |

（掃引行は seed 0 単発。fxp_base が float の stream_k10（×1.16）より良いのは
量子化の丸めがディザとして働くためで、固定小数点化のペナルティは実質ゼロ。）

- 単軸の下限: **重み小数部は 14 bit が安全圏**（12 bit は ×1.71 で崩れ際）。
  前活性は 8 bit、ノイズは 6 bit まで単軸では耐える（Fd8/Nn8 で readout
  mirror r が 0.91–0.93 に下がるのが予兆）。
- 実測語長（fxp_base）: 重み 16–18 bit、前活性 16 bit、mirror 分子 24–25 bit、
  除算分母 ≤ 14 bit、EMA 蓄積 20 bit、sgdm 速度 19–20 bit、credit 13 bit。
  いずれも ECP5/CertusPro-NX の 18×18 乗算器と素直に噛み合う。

#### Stage C 複合最小構成（2026-07-16 実施）: **PASS** — Phase 1 完了

単軸で生き残った縮小軸の組み合わせを 3 seeds で最終検証。

| config (Fw/Fd/Nn) | 最終 MSE | vs batch_ref | mirror r (out / W1) |
|---|---|---|---|
| 16 / 10 / 8 | 0.00069 ± 0.00007 | ×0.75 | 0.953 / 0.963 |
| **14 / 10 / 8（凍結仕様）** | **0.00082 ± 0.00015** | **×0.89** | 0.971 / 0.962 |
| 14 / 8 / 6（限界確認） | 0.00083 ± 0.00007 | ×0.90 | 0.971 / 0.964 |

### Phase 1 の結論と凍結仕様

**判定: PASS**。凍結した B=1 オンライン構成は、ストリーミング統計・LFSR 乱数・
固定小数点演算のすべてを重ねても Phase 0 batch_ref の ×0.89（×2 基準に対し
大幅な余裕）で学習し、mirror 回復 r ≈ 0.96–0.97 を維持する。float に対する
ペナルティは実質ゼロ（丸めがディザとして働く）。

**Phase 2 に引き渡す語長仕様**（fxp_min_14_10_8 の実測。カッコ内は小数部）:

| 量 | 語長 | 備考 |
|---|---|---|
| 重み・ミラー使用値 | 16 bit (f14) | 隠れ層 14 bit、readout 16 bit |
| 前活性 d / ys / credit | 14 bit (f10) | ノイズは 8 bit を f10 に整列 |
| 交差出力 z | 2 bit | code {0,1,2}（= 0, ½, 1） |
| mirror 分子 Num | 23 bit (f10) | (T·Σdz − Σd·Σz)、T=2⁶ |
| mirror 分母 Den | 14 bit 非負整数 | 列共有 → 除算はユニット毎 1 回 |
| mirror EMA 蓄積 | 18 bit (f18) | ガード 4 bit、率 2⁻¹⁰（シフト 1 回） |
| sgdm 速度 v | 17 bit (f14) | μ = 1−2⁻⁴ は `v−(v>>4)+g` |
| lr | 2⁻¹⁰ × ROM | 9 bit cosine ROM（1024 エントリ）× シフト |
| 乱数 | 24 bit LFSR × ユニット毎 | phase 配線。leap16（LFSR 1 本）でも ×1.11 |

全演算は「加減算・シフト・18×18 に収まる乗算・ユニットあたり 1 除算」で
閉じており、論文 §6.1 の資源分析が学習系全体で数値的にも裏づけられた。
さらに絞るなら Fw14/Fd8/Nn6（×0.90）まで動作確認済み。

**残課題（Phase 2 へ）**: 演算器の時分割共有と更新レイテンシ（1 提示あたりの
サイクル数）、mirror 除算器の共有度、重み SRAM のポート設計、飽和演算の追加
（ゴールデンモデルは int64 頭打ちなし。実測語長 +1 bit マージンで飽和させる）。

---

## 5. Phase 2: アーキテクチャ設計（完了）

見積りの計算は `tmp/fncl_phase2_arch.py`（数式 = 設計根拠。仕様変更時はここを
更新して表を再生成する）。結果は `tmp/out/fncl_phase2_arch/table_arch.md`。

### 5.1 設計方針: 2 本の 1-D PE アレイ + ブロードキャストバス

学習の演算は「隠れ層間の [H×H] 行列を、行方向（forward・重み更新）と
列方向（mirror 統計・credit 逆伝播・KP 追跡）の両方から舐める」構造を持つ。
これを 2 本の 1-D レーンアレイに分けて解決する。

```
            broadcast bus (code0_i / d1_j / dd1_j / Σd1_j / zbar0_i ...)
   ┌──────────────┴──────────────┐
   ROW アレイ (lane j = 第 2 層ユニット)   COL アレイ (lane i = 第 1 層ユニット)
   - W1[j][:], v_W1[j][:]  (行バンク)     - M1[:][i], vM[:][i], Σdz[:][i] (列バンク)
   - LFSR1_j, 交差 (comp+XOR+カウンタ)     - LFSR0_i, 交差, Σz0_i, Σz²0_i
   - readout 統計, Mout_j, v_Wout_j       - Den_i, recip_i, zbar0_i
   - 64 入力加算木 (ys = Σ Wout·z1)
```

- **forward** は ROW が担当: スロット i で code0_i をブロードキャストし、
  各レーン j が `d1_j += W1[j][i] を code でゲートした加算`（z は 2 bit なので
  乗算器不要）。
- **mirror 統計 / EMA / credit** は COL が担当: d1_j や dd1_j を j 順に
  ブロードキャストし、レーン i がローカルの列スライスに累算する。
  `a0_i = Σ_j M1[j][i]·dd1_j` も同じバスで 64 サイクル。
- **KP 追跡の転置問題**: 重み更新ステップ step_ji は ROW 側（v_W1 と同居)
  で生まれるが、ミラー M1 は COL 側にある。クロスバーを張る代わりに
  **COL 側が sgdm を複製再計算する**: レーン i は vM[:][i]（v_W1 の複製）を
  持ち、dd1_j のブロードキャストとローカルの zbar0_i から同じ step を再計算
  して M1 から引く。コストは +4096×18b RAM と +1 乗算器/レーン のみで、
  配線は増えない（v と vM は同じ入力から同じ演算をするので値は常に一致）。
- z の 2 bit 化により forward・Σdz・readout の MAC は全て「シフト+加算」。
  真の乗算器が要るのは mirror の Num×recip、credit の M×dd、sgdm の v×ROM
  だけで、全てレーンあたり 1–2 個に時分割できる。

### 5.2 提示 1 回の処理スケジュール（制御 FSM のフェーズ）

| フェーズ | 内容 | サイクル (L=64 / 32) |
|---|---|---|
| SAMPLE ×T | ノイズ生成→交差→d1 蓄積（ROW）∥ Σdz 蓄積（COL, 1 サンプル遅れのパイプライン）∥ readout 加算木 | 4352 / 8448 |
| MIRROR | Den・recip（除算 2H 回, パイプライン除算器 1 本）→ Num→Ŵ→EMA（COL）, out 側は ROW ローカル | 282 / 410 |
| CREDIT | e=2(y−t) → a1, dd1（ROW）→ a0 = M1ᵀdd1（COL, j 順ブロードキャスト）| 68 / 136 |
| UPDATE | ROW: i 順に g→v→step→W1、COL: j 順に vM 複製で M1 の KP | 168 / 304 |
| 合計 | | **4934 / 9362** |

75 MHz 想定で **L=64: 15.2 kHz、L=32: 8.0 kHz の更新レート**。budget 192k の
オンチップ学習は 13 s / 24 s。推論のみなら 17.0 / 8.8 kHz。ロボットの制御周期
（100 Hz–1 kHz）に対して 1 桁以上の余裕があり、**L=16（4.1 kHz, ECP5-45 適合）
でも成立する**。

### 5.3 メモリマップと資源

学習コア全体で **410 kb**（W1 72 + v 72 + M1 80 + vM 72 + Σdz 96 + ベクトル 18）。
推論専用に縮退させると 77 kb。バンキングはレーンローカル（ROW 1 本/レーン:
W1+v をパック、COL 2 本/レーン: M1+vM / Σdz）で EBR 3L 本、乗算器 ~3L+4 個。

| デバイス | 適合 | 備考 |
|---|---|---|
| ECP5-85 | **L ≤ 32**（DSP 100/156, EBR 96/208） | 学習込みプロトタイプの本命 |
| ECP5-45 | L ≤ 16（DSP 52/72, EBR 48/108） | オープンフロー最小構成 |
| CertusPro-NX-100 | L ≤ 32 | 低消費電力・ロボット搭載向け |
| iCE40 UP5K | 推論専用 L=8（W1 を SPRAM に配置, ~1.4 kHz） | 学習コアは不可 |

（LUT/EBR/DSP 数は概数。デバイス確定時にデータシートで再確認する。）

### 5.4 除算器と飽和演算

- 除算はユニットあたり 1 回 = **2H 回/更新**（隠れ mirror の recip_i が H、
  out mirror の Den が H）。24 bit 商のパイプライン非回復型 1 本
  （1 結果/サイクル、レイテンシ ~26）で 154 サイクル → ボトルネックでない。
- 全データパスは語長契約（`fncl_phase1_fxp.SAT_WIDTHS`: W 18b / d・ys 16b /
  credit 14b / slope 13b / v 18b / M 蓄積 20b / Num 24b）の**飽和演算**で実装
  する。契約はゴールデンモデルの `--stage fxpsat` で検証する（§5.6）。

### 5.5 I/O と動作モード

- 入力: x（f10, 14 bit）と教師 t（同）。ロボットではセンサ値を正規化して
  SPI/UART で投入。出力: y（f10）と学習中 MSE のテレメトリ。
- モード: TRAIN（提示ごと全フェーズ）/ INFER（SAMPLE のみ）/ FREEZE
  （ミラー・統計は回すが更新しない = デバッグ用）。
- lr ROM（1024×9 bit）は総ステップ数レジスタで正規化して引く。

### 5.6 実行報告

#### 語長契約の飽和演算検証（2026-07-16 実施）: **PASS** — Phase 2 完了

凍結構成（Fw14/Fd10/Nn8, phase24, EMA 2⁻¹⁰）の全データパスを語長契約
（W 18b / d・ys 16b / credit 14b / slope 13b / v 18b / M 蓄積 20b / Num 24b）
の飽和演算に置き換えて 192k 提示 × 3 seeds を学習:

| config | 最終 MSE | vs batch_ref | mirror r (out / W1) |
|---|---|---|---|
| fxp_sat_14_10_8 | 0.00076 ± 0.00015 | **×0.82** | 0.981 / 0.962 |

実測語長は全項目で契約より 2 bit 以上内側に留まり（飽和は実質的に発火せず、
発火しても安全に頭打ちになる保険として機能する）、性能は非飽和版（×0.89）と
同水準。**RTL はこの契約幅で実装してよい**。

これで Phase 2 の設計項目（PE アレイ構成・スケジュール・メモリマップ・
除算器・飽和契約・デバイス適合）はすべて確定。推奨構成は
**ECP5-85 / CertusPro-NX-100 + L=32（更新 8.0 kHz, 学習 24 s）**、
オープンフロー最小は ECP5-45 + L=16（4.1 kHz）。

### 5.7 Phase 3 への引き渡し（Amaranth モジュール分割案）

`lfsr.py`（24 bit Galois, ユニット毎シード）/ `crossing.py`（比較器+巡回 XOR
+カウンタ）/ `pe_row.py` / `pe_col.py`（レーン: RAM スライス+ALU+DSP）/
`divider.py`（非回復型）/ `ctrl.py`（フェーズ FSM + lr ROM）/ `top.py`。
検証は cocotb で `fncl_phase1_fxp.py` の train_stream_fxp をリファレンスに
1 提示単位の完全一致（bit-exact）比較。LFSR・丸めシフト・丸め除算の定義が
ゴールデンモデルと一致していることが前提条件。

## 6. Phase 3: RTL 実装（進行中）

実装は `rtl/fncl_rtl/`（Amaranth 0.5）、検証は `rtl/tests/`。検証方針は
「ゴールデンモデル `tmp/fncl_phase1_fxp.py` の対応関数と **bit-exact 一致**」。
Amaranth 内蔵シミュレータで比較するため cocotb / Verilator は Verilog 一式が
出てから導入すれば足りる。

### Step 1（2026-07-16 実施）: 基本モジュール 4 種 — **全て bit-exact PASS**

| モジュール | 内容 | 突き合わせ先 | 結果 |
|---|---|---|---|
| `lfsr.py` GaloisLfsr | 24 bit Galois LFSR（ユニット毎ノイズ源） | `lfsr_sequence` | 2048 状態一致 |
| `rounding.py` | 丸めシフト / 飽和（組み合わせ式） | `rshift` / `sat` | s=1,4,10 / 8,14,18 bit 一致 |
| `crossing.py` CrossingUnit | 2 値化→巡回 XOR→code {0,1,2}、同一パスで cdiff・Σcode・Σcode² を集計 | `crossing_code` | 20 窓 × T=64 全一致 |
| `divider.py` RoundDiv | 丸め付き符号付き除算（直列 1 bit/cyc、レイテンシ 33） | `rdiv` | 309 ケース一致 |

- 交差ユニットは**巡回項をストリーミングで処理**する（先頭ビットを保持し、
  最終サンプルの次のサイクルに wrap 項を出力）。code 出力は添字順なので
  下流の W1 MAC / Σdz はゴールデンモデルと同じ t 対応で消費できる。
- 除算器 v0 は直列（33 cyc）。2H=128 回/更新を 1 本で回すと mirror フェーズが
  約 +4.2k cyc になるが、それでも L=64 で合計 ~9.2k cyc（8 kHz 相当）に
  収まる。パイプライン化（1 結果/cyc）は後段の最適化とする。
- Verilog 書き出し経路を確認済み（`amaranth-yosys` フォールバック、
  `rtl/out/*.v`: lfsr 48 / crossing 286 / divider 225 行）。

### Step 2（2026-07-16 実施）: 学習コア RTL — **1 提示 bit-exact PASS**

`rtl/fncl_rtl/core.py` FnclCore: §5.1 の ROW/COL アレイ構成（L=H）で
1 提示の全工程（forward T サンプル → mirror 整数比 + EMA → 再帰 credit →
sgdm 更新 + KP 複製）を実行する学習コア。制御 FSM は §5.2 のフェーズを
そのまま実装（SAMPLE は code0 と d1 の 2 本のブロードキャストループ、
交差の巡回項は WRAP0/WRAP1 で明示処理）。レーンローカル記憶はレジスタ配列
（分散 RAM 相当）、除算はレーンあたり直列 RoundDiv 1 基の v0 構成。

**検証**（`rtl/tests/test_core.py`）: H=T=8・凍結語長（Fw14/Fd10/Nn8, 飽和付き）
で、ゴールデンモデル `fxp_step` を 20 提示連続実行し、毎提示後に
**y・重み 6 群・速度・M1s/Mos（EMA 蓄積）・vM（KP 複製）の全状態が bit-exact
一致**。LFSR は各レーンを m 系列オフセット（seq[off]）でシードするだけで
ゴールデンモデルの phase24 ノイズと完全に同期する。1 提示 535 cyc（H=T=8,
直列除算込み）。コアの Verilog 書き出しも確認（`rtl/out/core_h8.v`）。

準備としてゴールデンモデルを 2 点変更（どちらも再検証済み）:
- 1 提示ステップ関数化（`fxp_prepare` / `fxp_step`）。fxpsat seed 0 再実行で
  従前と完全一致。
- slope を `torch.round(cdiff × float)`（round-half-even, HW 非互換）から
  **`rdiv(cdiff << 2Fd, 2·hq·T)`（丸め付き整数除算）へ変更** — ランタイム
  データパスから浮動小数点を一掃。fxpsat 3 seeds 再実行で ×0.82・実測語長
  とも変化なし。

デバッグで得た RTL 側の教訓（Step 3 でも効く）:
- FSM 内の条件付き comb 代入は「非活性状態で init 値に戻る」— `wrap_en` の
  init=1 が保留のつもりの巡回項を即時発火させ、ブロードキャスト中のラッチを
  上書きしていた。ハンドシェイクのデフォルトは明示駆動する。
- 除算器は den をレイテンシ期間ずっと参照していた — start で内部ラッチする
  形に修正（呼び出し側の駆動保持を不要に）。

### Step 3（2026-07-16 実施）: L パラメトリック + Memory 化 — **bit-exact PASS**

`rtl/fncl_rtl/core_l.py` FnclCoreL: core.py の後継で、違いは 2 点。

- **L パラメトリック**: ROW/COL 各 L レーン。レーン l が行/列
  j = l·R+rc（R = H/L）を分担し、H² 配列を舐めるループは (bc, rc) の
  2 重カウンタに一般化。除算器はレーンあたり 1 基（計 2L）。
- **大配列 5 種を Memory に移行**: W1 / v_W1（ROW）、M1s / vM / Σdz（COL）
  はレーンローカルの同期読み出し Memory（深さ R·H、addr = rc·H+相手添字
  = EBR/分散 RAM に対応）。読み出しレイテンシ 1 は 2 相制御（ph）で吸収。
  交差・LFSR はユニット毎並列のまま（コンパレータ+カウンタで安価;
  折り畳みは Phase 4 の最適化）。H 長ベクトルもレジスタのまま。

**検証**（`rtl/tests/test_core_l.py`）: H=T=8・凍結語長で

| 構成 | 結果 | cyc/提示 |
|---|---|---|
| L=8 (R=1) | 6 提示 bit-exact | 703 |
| L=4 (R=2) | 6 提示 bit-exact | 1,373 |
| L=2 (R=4) | 6 提示 bit-exact | 2,713 |
| **L=4 長期** | **200 提示**（50 提示ごと + 先頭 2 で全状態照合）bit-exact | 1,373 |

ゴールデンモデルの軌跡は L に依存しないため同一スナップショット列で全 L を
照合。サイクル数は R にほぼ比例（v0 は 2 相アクセスのため §5.2 見積りの
約 2 倍。実スケール H=64/L=32 換算で ~39k cyc/提示 ≈ 1.9 kHz @75MHz —
ロボット制御には既に足りるが、§5.2 の 8 kHz へは SAMPLE 2 位相の
パイプライン重畳 + 除算器のパイプライン化で回復する = Step 4）。
実スケール H=T=64, L=32 の Verilog 変換も成功（49 万行, 42 s;
`rtl/out/core_l_h64_l32.v`）。長期試験は pysim で足りたため Verilator の
導入は合成後の大規模検証まで保留。

### 残ステップ

1. **Step 4**: Yosys/nextpnr-ecp5 で合成・配置配線試行（LUT/EBR/DSP 実測を
   §5.3 の見積りと突き合わせ、H 長ベクトルの RAM 化や交差ユニット折り畳みの
   要否を判断）+ SAMPLE パイプライン重畳・除算器パイプライン化 → Phase 4 へ。

## 7. 再現コマンド

```bash
cd tmp
# Phase 0 実験① (優勝構成の確認)
../.venv/bin/python fncl_phase0_sgd.py --opts sgdm --decays cosine --lrs 0.125 \
    --momentum 0.9375 --out out/fncl_phase0_sgd_r4
# Phase 0 実験② (B=1 + lr 線形則)
../.venv/bin/python fncl_phase0_online.py --batch-sizes 1 --lr-scale linear \
    --out out/fncl_phase0_online_b1lin
# Phase 1 (固定小数点ゴールデンモデル)
../.venv/bin/python fncl_phase1_fxp.py --stage stream   # Stage A
../.venv/bin/python fncl_phase1_fxp.py --stage lfsr     # Stage B
../.venv/bin/python fncl_phase1_fxp.py --stage fxp      # Stage C
../.venv/bin/python fncl_phase1_fxp.py --stage fxpmin   # Stage C 複合最小構成
# Phase 2
../.venv/bin/python fncl_phase1_fxp.py --stage fxpsat   # 語長契約の飽和演算検証
../.venv/bin/python fncl_phase2_arch.py                 # サイクル・資源見積り
# Phase 3 (リポジトリルートから; 要 pip install amaranth amaranth-yosys)
cd .. && .venv/bin/python rtl/tests/test_units.py       # 基本モジュール突き合わせ
.venv/bin/python rtl/tests/test_core.py                 # 学習コア 1 提示 bit-exact
.venv/bin/python rtl/tests/test_core_l.py               # L=8/4/2 Memory 版
.venv/bin/python rtl/tests/test_core_l.py --long        # 200 提示長期 (L=4)
```
