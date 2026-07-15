# 重みパラメータ θ とノイズ場 P の双対性（構想メモ）

NCE 投稿論文（`docs/nce_draft.md`, "Reconstructing Backpropagation from Forward
Fluctuations in Noise-modulated Neural Networks"）の**続編**として、ノイズ場を
主題にする構想の整理。現状の実装（`examples/dual_*.py` 2 本）が NCE 本体の
主張と噛み合っていない点と、その解消方針を記録する。

関連: `docs/forward_noise_covariance_learning.md`（FNCL の技術ノート）、
`docs/recipe_sr.md`（確率共鳴カーブ、Frontiers 向け）。

---

## 1. 双対性とは何か

NNN の出力を

$$y = f(x; \theta, \mathcal{P})$$

と書いたとき、学習可能な自由度が 2 つの異質な系に分かれている、というのが
出発点である。

- **θ**（重み行列群）: 各ユニットの入出力の**中身**を作る。「どのニューロンが
  何を計算するか」を決める、密で連続な線形写像。
- **P**（ノイズ場、ユニットごとのノイズ強度）: どのニューロンが確率的に
  活性化する、すなわち**機能するか**を決める。交差活性
  $\bar\phi(d;\sigma) = 2F(d;\sigma)(1-F(d;\sigma))$ を通して、$\sigma \to 0$ の
  ユニットは応答が恒等的に 0 になり、適切な $\sigma$ を持つユニットだけが
  確率共鳴で信号を伝える（`nce_draft.md` §3.2）。

θ は「計算の実体」、P は「その実体のどこを起動するかというルーティング／
ゲート」を担う。両者は同じ関数 $f$ を規定するが、役割が直交している。
これが双対性の中身である。

### 双対性を成立させている 3 つの構造的性質

1. **分離可能性（役割の直交）**。θ を固定したまま P を切り替えるだけで、同一
   ネットワークが別のタスクを解ける。`P_A → sin(x)`、`P_B → sin(2x)` を 1 つの
   θ の上に多重化できる。P はシンボリックな「鍵」ではなく、確率共鳴の最適強度を
   持つ機能的資源として振る舞う（`recipe_sr.md` の逆 U 字と整合）。
2. **対称性（どちらも独立に学習信号を運べる）**。θ を凍結して P だけ動かしても
   タスク損失が下がるなら、P それ自体が学習信号を担えることの証拠になる。
3. **相補性（joint 最適化での役割分担）**。θ が局所的な入出力マップを作り、
   P がどのマップを起動するかを選ぶ。鍵となるシグネチャは
   **overlap(P_A, P_B) が下がりながら total loss も下がる**こと。

---

## 2. 現状の実装

| | `examples/dual_node_perturbation_noise_field.py` | `examples/dual_finite_difference_node_perturbation.py` |
|---|---|---|
| θ の更新 | Adam + backprop | 有限差分ノード摂動（出力層のみ解析勾配） |
| P の更新 | SPSA | SPSA（層ごとリスト） |
| モデル | `SimpleNNNStatistic` | 専用 `NNNfd`（`CrossingAnalytic` を利用） |
| rho 初期化 | ランダム | 前半／後半バイアス |
| 追加可視化 | なし | 条件×層ごとの最終ノイズ場 P_A/P_B 棒グラフ |

両者とも 3 条件（theta-only / field-only / dual）を同一の初期 θ・初期 rho から
比較する。P は $P = \mathrm{softplus}(\rho)$ で正値性を保証し、損失は
タスク MSE に sparsity（$\lambda_\text{sparse}\,\mathrm{mean}(P)$）と
overlap（$\lambda_\text{overlap}\,\cos(P_A, P_B)$）の正則化を加えたもの。

### SPSA（Simultaneous Perturbation Stochastic Approximation, Spall 1992）

勾配を計算せず、損失の評価だけから勾配を推定する手法（`nce_draft.md` §2.3 の
関連研究に既出）。パラメータが $d$ 次元のとき素朴な差分は $2d$ 回の損失評価を
要するが、SPSA は全成分を同時にランダム方向 $\delta$ へ振って **2 回**で済ませる：

$$s = \frac{L(\rho + \varepsilon\delta) - L(\rho - \varepsilon\delta)}{2\varepsilon},
\qquad \hat{g} = s\,\delta$$

$s \approx \nabla L \cdot \delta$ より
$\mathbb{E}[\hat g] = \mathbb{E}[\delta\delta^\mathsf{T}]\nabla L = \nabla L$ で、
1 次までの不偏推定量となる（対称差分なので 2 次項が消え、誤差は $O(\varepsilon^2)$）。
分散は大きいが期待値が正しいので、小さい学習率で多数回まわせば降下する。
「$d$ 次元の勾配を 1 個のスカラーとランダム方向の外積で代用する」のが要点。

`dual_finite_difference_node_perturbation.py` の θ-step（ノード摂動）も、
摂動先がパラメータでなくプリ活性 $d$ である点が違うだけで骨格は同一である。

---

## 3. 問題: 現状の実装は NCE 本体の主張と噛み合っていない

NCE 論文の中心的な発見は、**「大域スカラー損失 × 摂動」型の credit
（`cov_only` / `cov_deriv`、ノード摂動の系譜）は消えない系統誤差を持ち、
backprop に届かない**ことだった（§4.2、§5.2）。

| 手法 | 最終 MSE |
|---|---|
| backprop | 0.00057 |
| `cov_only` | 0.09583 |
| `cov_deriv` | 0.03464 |
| `cov_jac` | 0.00056 |

この線形化バイアスは有限サンプル効果ではないため $T$ を増やしても消えない。

**ところが双対性の 2 スクリプトは、θ 側も P 側もまさにこの「負けた側」の
推定量で作られている。** しかも
`dual_finite_difference_node_perturbation.py:279` の `s_A` は
`F.mse_loss` が全 $N$ を平均済みのため**バッチ全体で 1 個のスカラー**であり、
per-input ですらない完全な pooled 版になっている。NCE のアブレーション
（per-input credit が決定的）からすると、最も弱い変種を使っていることになる。

すなわち、NCE 本体で「これでは駄目だ」と論証した道具立てで、その続編を
書こうとしている状態にある。

---

## 4. 方向 A（本命）: ノイズ場の勾配を共分散フレームワークの内側から出す

### 恒等式

任意のスケール族 $p(\xi) = \frac{1}{\sigma}q(\xi/\sigma)$ について、

$$\boxed{\;\frac{\partial \bar\phi(d;\sigma)}{\partial \sigma}
= -\frac{d}{\sigma}\,\bar\phi'(d)\;}$$

が成り立つ。導出は $F(d;\sigma) = Q(d/\sigma)$ より

$$\frac{\partial F}{\partial \sigma} = q(d/\sigma)\cdot\left(-\frac{d}{\sigma^2}\right)
= -\frac{d}{\sigma}\,p(d)$$

これを $\partial\bar\phi/\partial\sigma = 2(1-2F)\,\partial F/\partial\sigma$ に
代入すると、$\bar\phi'(d) = 2(1-2F)p(d)$ より直ちに得られる。

**数値検証済み**（`scratchpad/check_sigma_identity.py`、2026-07-15）:
ガウス $\sigma=0.5$ で残差 3.0e-11、一様 $r=1.0$ で 1.3e-10（量のスケールは
0.6〜0.8）。ガウス・一様の双方で機械精度で一致する。

### なぜ効くか

右辺の $\bar\phi'(d)$ は、**論文が §3.1 で既に分布フリーに推定している当の量
$\phi_T'$（KDE slope = 交差計数）**である。$-d/\sigma$ は各ユニットが自分で
持っている量。したがってノイズ場の勾配は、credit $\hat g$ を `cov_jac` から
そのまま流用して

$$\frac{\partial L}{\partial \sigma^{(l)}_k} \approx
\Bigl\langle\,
\underbrace{\hat g^{(l)}_{n,k}}_{\text{第 3 因子（credit）}}
\cdot
\underbrace{\Bigl(-\tfrac{d^{(l)}_{n,k}}{\sigma^{(l)}_k}\Bigr)\phi_T'\bigl(d^{(l)}_{n,k}\bigr)}_{\partial z_k/\partial \sigma_k\ \text{（局所感度）}}
\,\Bigr\rangle_{m,n}$$

と書ける。これは `nce_draft.md` 末尾の三要素則の式

$$\frac{\partial L}{\partial W^{(l)}_{ij}} \approx
\bigl\langle \hat g^{(l)}_{n,i}\cdot\phi_T'(d^{(l)}_{n,i})\cdot z^{(l-1)}_{n,j}\bigr\rangle_{m,n}$$

と**まったく同じ形**で、前シナプス因子 $z^{(l-1)}_j$ が $-d_k/\sigma_k$ に
置き換わっただけである。新しい統計量も、専用の摂動フェーズも、SPSA も要らない。
$\phi_T'$ が分布フリーなので**この $\sigma$ 勾配も分布フリー**であり、一様ノイズでも
同じ回路がそのまま動く（§6.2 の主張がそのまま継承される）。

### 主張の書き換え

> **θ と P は別々の学習機構で動く 2 つの系ではない。同一の credit $\hat g$ を、
> 外積の相手として $z_\text{prev}$ に当てれば θ の勾配、$-d/\sigma$ に当てれば
> P の勾配が出る。1 つの前向き共分散 credit に、2 人の消費者がいる。**

SPSA／ノード摂動版は**手法ではなくベースライン・アブレーション**に格下げし、
「摂動ベースでもできるが、共分散 credit を使えば桁で良い」と示すのが正しい配置。
NCE 本体の `cov_deriv` vs `cov_jac` の対比が P 側でも再現するはずで、これ自体が
結果になる。

---

## 5. 方向 B: NCE 本体には入れず、続編として書く（既に予告済み）

ドラフトは双対性を入れる場所を自分で空けている：

- **§3.2（240 行目）**: 「$s_k = 0$ のユニットは共分散 credit も厳密に 0 となって
  更新から自動的に外れる。これは **credit assignment と recruitment が同一の場を
  共有する**ことを意味し…」
- **§6.2 末（583〜585 行目）**: 「ノイズは分布だけでなく**空間的な割り当ても
  設計変数**である。…これらの設計とその評価は future work とする。」
- **§7（601〜603 行目）**: 「ノイズ場は、**学習資源の動員と消費電力の双方を
  空間的に制御する変数**であり…既存研究にない新しい視点を与える。」

NCE 本体は「backprop の再構成」で主張が既に一杯であり、双対性を足すと焦点が
ぼやける。**予告を回収する続編**という位置づけが素直で、§3.2 の
「credit と recruitment が同一の場を共有する」という一文が続編の主張そのものの
種になっている。方向 A を採れば、続編は「NCE で作った credit をノイズ場に
向けるだけで、場も学習できる」という最小の増分で最大の主張になる。

---

## 6. 方向 C: 「1 つの場が推論・学習・電力を同時にゲートする」を工学側の見出しに

方向 A から自動的に出る系として、$\sigma_k$ が学習可能になると $s_k \to 0$ は
**学習の結果として**到達できる。そのとき §3.2 の性質により、そのユニットは
推論からも学習からも厳密に外れる。さらに §6.2 の**一様ノイズのコンパクト
サポート**（$|d-c| \ge r$ で発火も学習統計も厳密に 0）と組み合わせると：

> ノイズ場という単一の変数が、(i) どのサブネットワークが計算するか、
> (ii) どこに credit が流れるか、(iii) どこが電力を食うか、を同時に決める。
> しかもそれ自身が同じ credit で学習される。

抽象的な「双対性」より、この具体的な帰結を前面に出すほうが査読で強い。
§7 が期待する FPGA + 小型ロボットの筋にそのまま接続する。

---

## 7. 方向 D: Frontiers 投稿との棲み分け

Frontiers「Computational Models of Neuromodulation」特集（締切 2026-09-11）も
同じノイズ場を主役にする計画であり、放置すると 2 本が競合する。切り分け案：

- **Frontiers** = 神経修飾の枠組み。主張は「ノイズ場が共有重みに多重化された
  方策を*アドレスする*、最適強度を持つ機能的資源（SR）」。主図は
  `examples/sr_separation_curve.py --sweep train --model sample` の逆 U 字。
  **場は与えるもの**で、学習則の話はしない。
- **NCE 続編** = 学習則の枠組み。主張は「その場自身が、backprop を再構成したのと
  同じ前向き共分散 credit で学習される」。多重化（P_A/P_B）は場が学習可能で
  あることの**デモ**であって、主張は credit の側にある。

この切り分けなら、Frontiers が「場は資源である」を、続編が「場は学習される」を
担い、相互に引用できて競合しない。

---

## 8. 方向 E: 現行スクリプトへの具体的な修正

1. **θ-step をノード摂動から `cov_jac` credit へ**（方向 A）。
   `data_nce/fncl/train.py` の `train_cov` / `covariance_credit` / `cov_weight` が
   そのまま使える。
2. **P-step を SPSA から $\sigma$ 勾配へ**（方向 A）。SPSA は比較対象として残す。
3. **pooled スカラー credit をやめる**。現状 `s_A` はバッチ全体で 1 スカラー。
   NCE のアブレーションは per-input が決定的だと示している。
4. **`SimpleNNNStatistic` → `SimpleNNNSample`**。Analytic/Statistic は平均場
   （理論）、Sample が機構。NCE 本体も §3.2 で「本研究の対象は Sample-level」と
   明示しており、続編もそこに揃えるべき。
5. **`softplus(rho)` パラメータ化の再検討**。$\sigma$ 勾配が直接出るなら rho を
   経由する必然性は薄い。一様ノイズなら半径 $r$ を直接クリップするほうが §6.2 の
   ハードウェア議論と整合する。

---

## 9. 次のステップ

方向 A の PoC: `cov_jac` credit + $\sigma$ 勾配で P_A/P_B を学習させ、SPSA 版と
比較する。既存の `data_nce/fncl/` を import すれば大きな作業にはならない。

確認したい仮説：

- (H1) $\sigma$ 勾配版が SPSA 版より桁で速く／低い損失に収束する。
- (H2) `cov_deriv` credit で P を学習すると `cov_jac` credit より系統的に劣る
  （P 側でも本体と同じ序列が再現する）。
- (H3) 学習された P で overlap が下がり、サブネットワーク分化が起きる。
- (H4) 一様ノイズでも同一の結論（分布フリー性の継承）。

---

## 付録: 恒等式の数値検証スクリプト

```python
# scale family p(xi) = (1/s) q(xi/s):  d phi_bar/d s == -(d/s) * phi_bar'(d)
# gaussian (s=0.5): max|num - pred| = 3.0e-11
# uniform  (r=1.0): max|num - pred| = 1.3e-10
import numpy as np
from scipy.stats import norm

def phibar_gauss(d, s):
    F = norm.cdf(d / s); return 2 * F * (1 - F)

def phibar_prime_gauss(d, s):
    F = norm.cdf(d / s); p = norm.pdf(d / s) / s
    return 2 * (1 - 2 * F) * p

d, s0, h = np.linspace(-0.45, 0.45, 9), 0.5, 1e-6
num  = (phibar_gauss(d, s0 + h) - phibar_gauss(d, s0 - h)) / (2 * h)
pred = -(d / s0) * phibar_prime_gauss(d, s0)
assert np.abs(num - pred).max() < 1e-9
```
