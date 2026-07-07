# NCE 投稿論文 構成案（ドラフト）

投稿先: **Neuromorphic Computing and Engineering (IOP)**

ソース:
- `docs/forward_noise_covariance_learning.md`（アルゴリズム・検証結果の詳細）
- `tmp/forward_noise_covariance_learning.py`（検証スクリプト）

主張の二本柱:
1. **主結果** — ノイズを利用する Noise-modulated Neural Network (NNN) では、forward パスの統計量のみから重みミラーを構成して誤差を再帰的に伝播でき（`cov_jac`）、局所的な Adam 更新と組み合わせる（`cov_jac_adam`）と **backprop と同水準の最終精度**（toy 回帰で MSE ~0.0012 vs backprop ~0.0011）に到達する。重み転置（weight transport）も backward データパスも一切使わない。
2. **副次的主結果** — ノイズ分布を**一様分布**にすると、交差活性・期待応答・局所勾配のすべてが多項式／比較器レベルの演算になり、系全体（forward・学習則とも）が**デジタル回路 / FPGA 実装に適した構成**になる。

---

## 論文タイトル（決定）

# **Reconstructing Backpropagation from Forward Noise in Noise-Modulated Neural Networks**

- cov_jac の本質 — forward ノイズの共分散統計から backprop の構造（weight mirror ＋ 再帰的誤差伝播）を「再構成」する — を最も正確に言い当てるタイトル。コロンなし・宣言的。
- "Reconstructing"（再構成）は "replacing"（置換）より正直な語であり、readout が局所（厳密）勾配であること・near-exact であることとも整合する。Abstract / Introduction でもこの動詞に揃えて主張を統一する。

<details>
<summary>検討したその他の候補（記録）</summary>

- "Forward-Only Covariance Learning Achieves Backpropagation-Level Accuracy in Noise-Modulated Neural Networks"（宣言文型・情報量最大）
- "Forward-Only Covariance Learning in Noise-Modulated Neural Networks"（最短）
- "Covariance-Based Weight Mirrors for Forward-Only Learning in Noise-Modulated Neural Networks"（機構重視）
- "Exploiting Intrinsic Stochasticity for Forward-Only Learning in Noise-Modulated Neural Networks"
- "Hardware-Friendly Forward-Only Covariance Learning in Noise-Modulated Neural Networks"（HW 訴求）
- コロン付き案: "Forward-Only Learning in Noise-Modulated Neural Networks: Covariance-Based Weight Mirrors Reach Backpropagation-Level Accuracy" ほか

</details>

---

## 構成案（節・小節）

### Abstract

（日本語たたき台）

誤差逆伝播法（backpropagation）は、転置重み行列を介した backward パス — いわゆる重み輸送（weight transport）— を必要とするため、ニューロモルフィックハードウェアへの実装が本質的に困難である。本論文では、ノイズを計算資源として利用する Noise-modulated Neural Network（NNN）において、backpropagation の構造を forward パスの統計量のみから**再構成**できることを示す。NNN の交差活性化関数は本質的に確率的であり、1 回の推論で多数の確率的 forward サンプルを生成する。前活性がactivationに対して線形であることを利用すると、この揺らぎの共分散 $\mathrm{Cov}(d^{(l+1)}, z^{(l)})/\mathrm{Var}(z^{(l)})$ から順方向重み行列そのものを推定できる（weight mirror）。推定した重みと、交差活性が内蔵する分布フリーな微分推定器を用いて、厳密な出力誤差を計算グラフに沿って再帰的に伝播することで、転置重みの読み出しも backward データパスも一切用いずに backpropagation の重み更新を再構成する。提案する学習則の勾配は真の勾配とコサイン類似度 $\approx 1.0$ で一致する経験的に不偏な推定量であり、局所的な重みごとの Adam 更新と組み合わせることで、toy 回帰タスクにおいて backpropagation と同水準の最終精度（MSE $1.2 \times 10^{-3}$ vs $1.1 \times 10^{-3}$）に到達する。さらに、注入ノイズを一様分布にすると、期待応答は 2 次多項式、局所微分は一次式に退化し、乱数生成は LFSR、activationは比較器と XOR で構成できるため、学習則を含む系全体がデジタル回路・FPGA 実装に適した構成となる。これらの結果は、NNN においてノイズが推論の資源であるだけでなく、backpropagation を担う学習の資源でもあることを示している。

（英文化時の注意: 主張は "reconstruct"（再構成）で統一し、"replace"/"exact" は避ける。readout 層は局所厳密勾配であることを本文で明示。）

### 1. Introduction

（日本語たたき台。段落構成は 背景 → NNN の位置づけ → 着想 → 貢献 → 論文構成。）

深層学習の成功を支える誤差逆伝播法（backpropagation）は、しかしながらハードウェア実装の観点からは扱いにくいアルゴリズムである。隠れ層の credit assignment

$$
\boldsymbol{\delta}^{(l)} = \bigl(W^{(l+1)\mathsf{T}} \boldsymbol{\delta}^{(l+1)}\bigr) \odot \bar\Phi'\bigl(\mathbf{d}^{(l)}\bigr)
$$

は、誤差信号を**転置重み行列** $W^{\mathsf{T}}$ に乗じて逆方向に伝播することを要求する。専用ハードウェア上でこれを実現するには、forward パスとは別の第二のデータパス、転置アクセス可能な重みメモリ（あるいは重みの複製）、そして forward/backward 位相の厳密な同期が必要となる — いわゆる重み輸送（weight transport）問題である [refs]。エッジデバイスやニューロモルフィックチップ上でのオンチップ学習への要求が高まる中 [refs]、この問題を回避する学習則として、feedback alignment とその変種 [refs]、target propagation [refs]、ノード摂動・重み摂動 [refs]、Forward-Forward をはじめとする forward-only 学習則 [refs] などが提案されてきた。しかしこれらの多くは、backpropagation の勾配を別の目的関数や固定ランダム重みで**置き換える**ものであり、勾配との一致は保証されず、精度ギャップが残ることが知られている [refs]。一方、activationの揺らぎの相関から転置重みを推定する weight mirror [Akrout et al. 2019] や Kolen–Pollack 法 [1994] は勾配そのものへの収束を狙うが、学習用の摂動を注入する専用の位相（mirror mode）を必要とする。

本論文は、この問題を Noise-modulated Neural Network（NNN）[refs] の上で考える。NNN は、ノイズを除去すべき外乱ではなく計算資源として利用するネットワークであり、その基本要素である交差活性化関数は、注入ノイズによって入力が閾値をまたいだときにのみ発火する確率的な二値ユニットである。確率共鳴により、二値ユニットの集合統計が滑らかな関数近似を与えること [refs]、ノイズの空間分布（noise field）が学習・推論に動員される部分ネットワークを規定すること [refs] が示されてきた。重要なのは、NNN の推論そのものが多数の確率的 forward サンプルの生成と平均化で構成されている、という点である。

本研究の着想は単純である: **その forward サンプルの揺らぎは、credit assignment に必要な情報を既に含んでいる**。各ユニットのactivationの揺らぎ $z - \mathbb{E}[z]$ は、学習のために注入される摂動なしで、そのユニットの変動が損失に与える影響を運んでいる（暗黙的ノード摂動）。さらに、前活性 $\mathbf{d}^{(l+1)} = W^{(l+1)} \mathbf{z}^{(l)} + \mathbf{b}^{(l+1)}$ がactivationに対して線形であることから、揺らぎの共分散は順方向重みそのものを開示する:

$$
\frac{\mathrm{Cov}\bigl(d^{(l+1)}_j,\, z^{(l)}_i\bigr)}{\mathrm{Var}\bigl(z^{(l)}_i\bigr)} \;\approx\; W^{(l+1)}_{ji}.
$$

すなわち、weight mirror が必要とする摂動は NNN では**すべての forward パスに無料で内在**しており、専用の摂動位相は不要である。共分散から推定した重み $\hat W$ と、交差活性が内蔵する分布フリーな微分推定器（第 3.4 節）を用いれば、厳密な出力誤差を計算グラフに沿って再帰的に伝播でき、転置重みの読み出しも backward データパスも用いずに backpropagation の重み更新を**再構成**できる。これが提案手法 `cov_jac` の中核である。

本論文の貢献は以下の通りである。

1. **Forward-only な共分散 credit assignment の族の提案**: 損失とactivationの共分散に基づく単純な学習則（`cov_deriv`）から出発し、その推定量バイアスの解析を経て、共分散 weight mirror による誤差の再帰伝播（`cov_jac`）に至る系統的な構成を与える。
2. **backpropagation 水準の学習の実証**: `cov_jac` の勾配が真の backpropagation 勾配とコサイン類似度 $\approx 1.0$ で一致する経験的に不偏な推定量であることを直接検証し、局所的な重みごとの Adam 更新（重み輸送を要しない）と組み合わせることで、toy 回帰タスクで backpropagation と同水準の最終精度（MSE $1.2 \times 10^{-3}$ vs $1.1 \times 10^{-3}$）に到達することを示す。
3. **有益な負の結果**: 二値の交差ユニットに対しては、外部からのノード摂動は共通乱数による分散低減を施しても原理的に機能しない（パスワイズ応答の退化）ことを示し、交差活性自身の密度推定こそが正しい局所勾配の推定器であることを明らかにする。
4. **デジタルハードウェア親和性**: 注入ノイズを一様分布にすると、期待応答が 2 次多項式、局所微分が一次式となり、乱数生成（LFSR）・activation（比較器＋XOR）・学習統計（カウンタと積和）のすべてが乗算器を最小限しか用いないデジタル回路で構成できることを示し、backpropagation との定性的リソース比較を与える。

以下、第 2 節で関連研究を整理し、第 3 節で NNN と交差活性化関数を定式化する。第 4 節で forward ノイズ共分散に基づく学習則を構成し、第 5 節で実験的検証を与える。第 6 節で一様ノイズ NNN のハードウェア適性を論じ、第 7 節の議論を経て第 8 節で結論を述べる。

（執筆時の注意: 貢献 2 の数値は seed 複数の統計を取ってから mean±std に差し替える。weight mirror / KP との差分（「摂動が無料」）は査読上の新規性の核なので、関連研究節でも同じ言葉で反復する。）

### 2. Related Work

（日本語たたき台）

#### 2.1 重み輸送問題と backpropagation 代替

backpropagation の隠れ層 credit assignment が転置重み行列 $W^{\mathsf{T}}$ を要求すること — 重み輸送問題 — は、生物学的妥当性とハードウェア実装の両面から長く指摘されてきた [refs]。feedback alignment は誤差の逆伝播を固定ランダム行列 $B$ で代替できることを示し [Lillicrap et al.]、direct feedback alignment は出力誤差を各層へ直接投影する [Nøkland]。しかしこれらの手法では順方向重みが $B$ に「揃う」ことに依存するため勾配との一致は保証されず、タスクや深さによる精度ギャップが報告されている [refs]。target propagation 系 [refs] は逆写像の学習によって誤差信号を置き換えるが、逆写像自体の学習コストを伴う。これに対し、順方向重みの転置を活動の相関から**推定**して真の勾配への収束を狙うのが Kolen–Pollack 法 [Kolen & Pollack 1994] と weight mirror [Akrout et al. 2019] である。weight mirror は各層にノイズを注入して活動の相関から $W^{\mathsf{T}}$ を推定するが、そのために推論とは別の専用位相（mirror mode）と摂動注入回路を必要とする。**本研究との差分**: NNN では weight mirror が必要とする摂動が全 forward パスに内在しており、推論そのものが mirror の測定を兼ねる。すなわち専用位相なしで、重み推定（第 4.4 節の $\hat W$）・Kolen–Pollack 型の追跡・誤差の再帰伝播が forward 統計のみから構成できる。

#### 2.2 摂動ベース学習

摂動を注入して損失の変化と相関を取る学習則には、ノード摂動 [Widrow & Lehr; Fiete & Seung ほか]、重み摂動 / SPSA [refs]、REINFORCE 型の方策勾配 [Williams]、および近年の forward gradient / zeroth-order 法 [Baydin et al.; refs] がある。これらは backward パスを要しない反面、推定分散が次元とともに増大することが本質的な制約である [refs]。本研究の共分散 credit（第 4.2 節）は、activationの自発的揺らぎ $z - \mathbb{E}[z]$ を摂動と見なして損失を回帰する**暗黙的ノード摂動**であり、摂動注入のための追加チャネルを持たない。さらに本研究は、二値の交差ユニットに対しては**外部**ノード摂動が — 共通乱数による分散低減を施してさえ — 原理的に機能しないこと（パスワイズ応答の退化、第 5.6 節）を示し、確率的二値ユニットにおける摂動ベース学習の適用限界を明らかにする点でもこの系譜に位置づく。

#### 2.3 Forward-only 学習則

backward パス自体を排する学習則として、正例・負例の局所的な goodness を対比する Forward-Forward [Hinton 2022]、誤差で変調した二度目の forward パスを用いる PEPITA [Dellaferrera & Kreiman 2022] などが提案されている [refs]。これらは大域的な損失勾配を**局所目的関数で置き換える**アプローチであり、backpropagation の解に一致する保証はない。本研究も forward パスのみで学習する点は共通するが、目的関数を置き換えるのではなく、**backpropagation の勾配構造そのもの（重み・局所微分・誤差の再帰）を forward 統計から再構成する**点で立場が異なる。実際、提案則の更新は真の勾配とコサイン類似度 $\approx 1.0$ で一致し（第 5.3 節）、最終精度も backpropagation と同水準に達する（第 5.2 節）。

#### 2.4 スパイキングネットワークの surrogate gradient と確率的ニューロン

微分不可能な二値・スパイク活性の学習では、階段関数の微分を平滑な代理関数で置き換える surrogate gradient [Neftci et al.; refs] や straight-through estimator [Bengio et al.]、確率的ニューロンによる平滑化 [refs] が標準的である。これらの代理微分は手設計であり、forward の計算とは独立に選ばれる。対照的に NNN の交差活性では、ノイズが forward の応答（期待値 $\bar\phi$）と backward の微分（$\bar\phi' = 2(1-2F)p$）の**双方を同一の統計から**定め、さらに微分はユニット自身のサンプルから分布フリーに推定できる（第 3.4 節）— 推論と学習の間に代理関数由来の不整合が生じない [arXiv:2606.24588]。ハードウェア面では、確率的ビット列で演算を構成する stochastic computing [refs]、FPGA 上のオンチップ学習・SNN 学習則の実装 [refs] が関連する。本研究の学習統計（カウンタ・積和・ユニットあたり 1 除算）はこれらの実装様式と親和的であり、特に一様ノイズ変種では乱数生成から局所微分までが基本的なデジタル素子に退化する（第 6 節）。

### 3. Noise-Modulated Neural Networks

（構成: 3.1 交差活性化関数 / 3.2 ネットワーク構造とアンサンブル平均読み出し / 3.3 期待応答と雑音誘起局所微分 / 3.4 分布フリーな局所傾き / 3.5 ノイズ場。以下、日本語たたき台。記法は先行論文 [Ikemoto, final_preprint; Ikemoto & DallaLibera, arXiv:2606.24588] に合わせる。）

#### 3.1 交差活性化関数（crossing activation function）

Noise-modulated Neural Network（NNN）の基本要素は、ノイズが存在するときにのみ発火する「交差（crossing）」活性化関数である [refs]。入力 $d \in \mathbb{R}$ に対し、独立同分布のノイズ $\eta_1, \eta_2 \overset{\mathrm{i.i.d.}}{\sim} p(\xi)$ を用いて

$$
z = \phi(d) =
\begin{cases}
1 & \text{if } (d \ge \eta_1) \,\dot\vee\, (d \ge \eta_2) \\
0 & \text{otherwise}
\end{cases}
$$

と定義される。ここで $\dot\vee$ は排他的論理和（XOR）である。直観的には、この活性化関数はノイズによって入力が閾値を「またいだ」瞬間にのみ 1 を出力する単純なスパイキングニューロンであり、ノイズが無ければ（$p(\xi) = \delta(\xi)$ ならば）定数入力に対して交差事象は起こり得ず、出力は恒等的に 0 となる。この「ノイズが無ければ出力も勾配も消える」という性質が、ノイズを計算資源として扱う NNN の出発点である。

ノイズの累積分布関数を

$$
F(d) = P(d \ge \eta) = \int_{-\infty}^{d} p(\xi)\, d\xi
$$

とおくと、交差活性の期待値は

$$
\mathbb{E}[z] = \bar\phi(d) = 2F(d)\bigl(1 - F(d)\bigr)
$$

で与えられる。$F$ は単調非減少なので $\bar\phi : \mathbb{R} \to [0, 0.5]$ は $F(d) = 0.5$ で最大となる単峰のベル型応答であり、radial basis function に類似した滑らかな基底として機能する。二値の $z$ 自体は連続入力の情報をほとんど失うが、確率的発火の集合統計 $\bar\phi(d)$ は連続情報を保持する — これは最も単純な確率共鳴（stochastic resonance）の形である [refs]。

#### 3.2 ネットワーク構造とアンサンブル平均読み出し

本研究で扱う NNN は、交差活性を要素ごとに適用する全結合フィードフォワード網である。入力を $\mathbf{z}^{(0)} = \mathbf{x}$ とし、隠れ層 $l = 1, \dots, L-1$ について

$$
\mathbf{d}^{(l)} = W^{(l)} \mathbf{z}^{(l-1)} + \mathbf{b}^{(l)}, \qquad
\mathbf{z}^{(l)} = \Phi^{(l)}\!\bigl(\mathbf{d}^{(l)}\bigr),
$$

出力層は線形読み出し

$$
\mathbf{y} = W^{(L)} \mathbf{z}^{(L-1)} + \mathbf{b}^{(L)}
$$

とする。$\Phi^{(l)}$ は交差活性 $\phi$ の要素ごとの適用であり、学習対象パラメータは $\theta = \{W^{(l)}, \mathbf{b}^{(l)}\}$ のみである（活性化層自体は学習パラメータを持たず、ノイズはハイパーパラメータとして与えられる）。

先行研究 [arXiv:2606.24588] で整理したように、交差活性には (i) 二値の確率的スパイクを出力する **Sample-level**（$\phi$ そのもの）、(ii) $T$ 個のサンプル平均で確率的連続値を出力する **Statistical-level**、(iii) 期待値 $\bar\phi$ を決定論的に出力する **Analytical-level** の 3 つの実装水準がある。本研究の対象は Sample-level のネットワークである: 1 回の推論において各ユニットは独立なノイズを $T$ 回引き、二値activationのサンプル列 $z^{(l)}_{(m)}\ (m = 1, \dots, T)$ を生成し、ネットワーク出力はその線形読み出しのアンサンブル平均

$$
\bar{\mathbf{y}} = \frac{1}{T} \sum_{m=1}^{T} \mathbf{y}_{(m)}
$$

として得る（期待値読み出し）。重要なのは、この $T$ 個の確率的 forward サンプルが**推論のために元々計算されるものであり**、後述する学習則（第 4 節）はそれを追加コストなしに credit assignment の統計量として再利用する、という点である。また、サンプルごとの損失 $L_{(m)} = \|\mathbf{y}_{(m)} - \mathbf{t}\|^2$ と各ユニットの activationの揺らぎ $z_{(m)} - \bar z$ が、第 4 節の共分散推定の生の材料となる。

#### 3.3 期待応答と雑音誘起局所微分

Analytical-level の期待応答 $\bar\phi$ を $d$ で微分すると、

$$
\bar\phi'(d) = 2\bigl(1 - 2F(d)\bigr)\, p(d)
$$

が得られる。これは、二値の階段関数の集まりであるにもかかわらず、ノイズが誘起する統計的な傾き $\mathrm{d}\mathbb{E}[z]/\mathrm{d}d$ が解析的に定まることを意味する。$\bar\phi'$ は $F(d) < 0.5$（閾値の下側）で正、上側で負、閾値から遠ざかると消失する。スパイキングネットワークの surrogate gradient が手設計の平滑関数で階段関数の微分を置き換えるのに対し、NNN では forward の応答そのものとその微分が同一のノイズ統計から導かれるため、推論と学習の間に不整合がない [arXiv:2606.24588]。

本研究では 2 種類のノイズ分布を扱う。

**(a) ガウスノイズ** $p(\xi) = \mathcal{N}(0, \sigma^2)$ のとき、

$$
F(d) = \frac{1}{2}\left(1 + \operatorname{erf}\!\frac{d}{\sqrt{2}\,\sigma}\right), \qquad
\bar\phi'(d) = 2\bigl(1 - 2F(d)\bigr)\,
\frac{1}{\sqrt{2\pi}\,\sigma} e^{-d^2 / 2\sigma^2}.
$$

**(b) 一様ノイズ** $p(\xi) = \frac{1}{2r}\,\mathbf{1}\{|\xi - c| \le r\}$（中心 $c$、半幅 $r$）のとき、$u = d - c$ として $|u| < r$ の範囲で $F(d) = (u + r)/(2r)$ であり、期待応答は**厳密な放物線**

$$
\bar\phi(d) = \frac{1}{2}\left[1 - \left(\frac{d - c}{r}\right)^{2}\right]_{+}, \qquad
\bar\phi'(d) = -\frac{d - c}{r^{2}} \quad (|d - c| < r,\ \text{それ以外は } 0)
$$

となる。ガウスの場合に誤差関数と指数関数の評価を要するのに対し、一様ノイズでは期待応答が 2 次多項式、局所微分が**一次式**に退化する。この事実は第 6 節で述べるデジタルハードウェア実装（一様乱数は LFSR で直接生成でき、$\bar\phi'$ は LUT すら要しない）の基礎となるため、ここで強調しておく。

#### 3.4 分布フリーな局所傾き（一様カーネル密度推定）

$\bar\phi'(d)$ の解析形はノイズ分布 $p(\xi)$ の事前知識を要するが、交差活性はその**内部に分布フリーな微分推定器を持つ**。先行研究 [final_preprint, Appendix A; arXiv:2606.24588] に従い、同一のノイズサンプルに対して入力を $\pm h$ だけずらした 2 つの交差出力

$$
z^{\pm} = \phi(d \pm h)
$$

を考え、それぞれの $T$ サンプル平均を $\bar z^{+}, \bar z^{-}$ とすると、

$$
\phi_T(d) = \frac{1}{2}\bigl(\bar z^{+} + \bar z^{-}\bigr) \approx \bar\phi(d), \qquad
\phi_T'(d) = \frac{\bar z^{+} - \bar z^{-}}{2h} \approx \bar\phi'(d)
$$

が成り立つ。$\phi_T'$ は一様カーネル（帯域幅 $h$）によるカーネル密度推定に他ならず、$T \to \infty,\ h \to 0$ で $\bar\phi'$ に収束する。

ここで統計的に重要なのは、$z^{+}$ と $z^{-}$ が**共通のノイズサンプル**上で評価される点である。閾値を $\pm h$ ずらすことは前活性 $d$ を $\mp h$ ずらすことと等価なので、$\phi_T'$ は共通乱数（common random numbers）の下での対称有限差分 $[z(d+h) - z(d-h)]/(2h)$ に相当し、共有されたノイズが差分で相殺されるため低分散である。すなわち交差活性は、antithetic/CRN 型の分散低減を**自身のサンプル上で無料で**実行する微分推定器を内蔵している。この傾き推定は $p(\xi)$ を一切参照しない（分布フリー）ため、ガウス・一様いずれのノイズに対しても同一の回路・同一のコードで動作する。第 4 節の学習則ではこの $\phi_T'$ を局所感度 $\partial z / \partial d$ として用い、第 5 節で解析形 $\bar\phi'$ を用いた場合と精度が一致することを確認する（したがって解析形は不要である）。

#### 3.5 ノイズ場（noise field）

各ユニットに与えるノイズ分布は一様である必要はない。第 $l$ 層第 $k$ ユニットのノイズ分布を $p^{(l,k)}$ とし、その集合

$$
\mathcal{P} = \bigl\{\, p^{(l,k)} \,\bigr\}_{l,k}
$$

を**ノイズ場**と呼ぶ [arXiv:2606.24588]。$p^{(l,k)} = \delta$（ノイズなし）のユニットは出力も勾配も恒等的に 0 となり、推論・学習の双方から完全に切り離される。したがってノイズ場は、どの部分ネットワークを計算に動員（recruit）するかを定めるトポロジー決定ハイパーパラメータとして機能する。本研究では、各ユニットのノイズ強度（ガウスの $\sigma$ または一様の $r$）をスカラー場 $s_{k} \in [0, 1]$ で変調する最小構成を用い、第 4 節の学習則がこの場と自然に整合すること（$s_k = 0$ のユニットは共分散 credit も厳密に 0 となり、更新から自動的に外れること）を示す。これは credit assignment と recruitment が同一の場を共有することを意味し、第 6 節で述べる消費電力ゲーティングへの含意を持つ。

### 4. Forward-Noise Covariance Learning

（日本語たたき台）

#### 4.1 問題設定と記法

データセット $\mathcal{D} = \{\mathbf{x}_n, \mathbf{t}_n\}_{n=1}^{N}$ に対し、第 3.2 節の Sample-level NNN は入力ごとに $T$ 個の確率的 forward サンプルを生成する。入力 $n$、サンプル $m$ における第 $l$ 層の前活性・activationを $\mathbf{d}^{(l)}_{(m,n)},\ \mathbf{z}^{(l)}_{(m,n)}$、読み出し層のサンプル出力を $\mathbf{y}_{(m,n)}$ とし、サンプルごとの損失を

$$
L_{(m,n)} = \bigl\|\mathbf{y}_{(m,n)} - \mathbf{t}_n\bigr\|^2
$$

と定義する。以下で提案する学習則は、これらの forward 量のみから重み更新を構成する — 転置重み行列の読み出し、backward データパス、autograd はいずれも用いない。

読み出し層についてはあらかじめ明確にしておく。ネットワーク出力 $\bar{\mathbf{y}}_n$ は $W^{(L)}$ に対して線形なので、アンサンブル平均activation $\bar{\mathbf{z}}^{(L-1)}_n = \frac{1}{T}\sum_m \mathbf{z}^{(L-1)}_{(m,n)}$ 上の読み出し勾配

$$
\Delta W^{(L)} = -\eta\, \frac{1}{N} \sum_{n} \frac{\partial L}{\partial \bar{\mathbf{y}}_n} \bar{\mathbf{z}}^{(L-1)\mathsf{T}}_n,
\qquad
\frac{\partial L}{\partial \bar{\mathbf{y}}_n} = 2(\bar{\mathbf{y}}_n - \mathbf{t}_n)
$$

は**局所かつ厳密**であり、隠れ層への誤差伝播を一切必要としない。したがって本研究が backward パスを除去するのは隠れ層の credit assignment であり、問題は「$\partial L / \partial z^{(l)}_i$ を forward 統計からどう推定するか」に帰着する。

#### 4.2 スカラー共分散 credit（`cov_deriv`）

最も単純な構成は、各ユニットの自発的揺らぎに対する損失の回帰である。入力 $n$ ごとに $T$ サンプル上で中心化した共分散を用いて、

$$
g^{(l)}_{n,i} = \frac{\mathrm{Cov}_T\bigl(L_n,\, z^{(l)}_{n,i}\bigr)}{\mathrm{Var}_T\bigl(z^{(l)}_{n,i}\bigr) + \epsilon}
\;\approx\; \frac{\partial L_n}{\partial z^{(l)}_{n,i}}
$$

を activity 側の credit とする。これは backpropagation の $W^{(l+1)\mathsf{T}} \boldsymbol{\delta}^{(l+1)}$ 項を forward 統計で置き換えるものである。これに第 3.4 節の分布フリーな局所傾き $\phi_T'$ を乗じて擬似誤差

$$
\hat\delta^{(l)}_{n,i} = g^{(l)}_{n,i}\, \phi_T'\bigl(d^{(l)}_{n,i}\bigr),
\qquad
\Delta W^{(l)} = -\eta\, \Bigl\langle \hat{\boldsymbol\delta}^{(l)} \mathbf{z}^{(l-1)\mathsf{T}} \Bigr\rangle_{m,n}
$$

を得る。$g \cdot \phi_T' \approx (\partial L/\partial z)(\partial z/\partial d) = \partial L/\partial d$ なので、更新は backpropagation と同じ外積形式 $\boldsymbol{\delta} \mathbf{z}^{\mathsf{T}}$ を持つ。実装上の要点は共分散の**入力ごと（per-input）中心化**である: 全サンプルを一括して中心化すると（pooled）、入力間の損失変動という交絡が支配的になり credit が入力依存性を失う。入力内の $T$ サンプルで中心化することでこの交絡が除かれ、局所勾配のはるかに偏りの小さい推定となる（第 5.4 節のアブレーションで pooled との差を示す）。必要な統計量はユニットあたり $\overline{L},\ \overline{z},\ \overline{Lz},\ \overline{z^2}$ の 4 つの累算のみであり、オンラインで蓄積できる。

#### 4.3 スカラー credit の限界 — 消えない推定量バイアス

`cov_deriv` は実際に学習する（第 5 節）が、backpropagation との間に**サンプル数を増やしても消えない残差床**を残す。原因は推定量の構造にある: $\mathrm{Cov}(L, z_i)/\mathrm{Var}(z_i)$ は、ユニット $i$ の下流に広がるネットワーク全体の効果を**スカラーの損失 $L$ ひとつに潰した**単変量回帰であり、複数ユニットが同時に揺らぐときの交互作用を線形化して無視する。この線形化誤差は真の共分散の性質であって有限サンプル効果ではないため、$T$ を増やしても減少しない（実験的にも学習率減衰・サンプル増で床が下がらないことを確認する; 第 5 節）。そこで問うべきは「**何に対して共分散を取るか**」である。スカラー $L$ ではなく、計算グラフの構造に沿った量と相関を取れば、この潰れは回避できるはずである — これが次項の提案の動機となる。

#### 4.4 構造化共分散 credit（`cov_jac`）— backpropagation の再構成

**(a) 共分散 weight mirror.** 前活性はactivationに対して厳密に線形（$\mathbf{d}^{(l+1)} = W^{(l+1)} \mathbf{z}^{(l)} + \mathbf{b}^{(l+1)}$）なので、入力を固定したときの $T$ サンプル揺らぎの単回帰係数は順方向重みそのものを与える:

$$
\hat W^{(l+1)}_{ji}
= \frac{1}{N} \sum_{n}
\frac{\mathrm{Cov}_T\bigl(d^{(l+1)}_{n,j},\, z^{(l)}_{n,i}\bigr)}{\mathrm{Var}_T\bigl(z^{(l)}_{n,i}\bigr) + \epsilon}
\;\approx\; W^{(l+1)}_{ji}.
$$

各ユニットの交差ノイズは入力を固定すれば独立に揺らぐため、この推定は経験的にほぼ厳密である（Pearson $r \approx 1.0$; 第 5.3 節）。2 つの実装上の要点がある。第一に、相関は**連続量の前活性 $d^{(l+1)}$ と取る**こと — 二値のactivation $z^{(l+1)}$ と取ると推定は大きく劣化する（$r \approx 0.14$; 二値ユニットのパスワイズ退化と同根、第 5.6 節）。第二に、重みは（勾配と異なり）入力に依存しないので、per-input 中心化を保ったまま分子・分母を入力について**プールして**から除算でき（$\hat W = \sum_n \mathrm{Cov}_n / \sum_n \mathrm{Var}_n$）、全 $NT$ サンプルが 1 つの推定に寄与する。

**(b) Kolen–Pollack 型の mirror 追跡.** 重み更新は学習則自身が適用するため、その増分 $\Delta W^{(l)}$ は**既知**である。これを mirror に直接積分する（$\hat W \leftarrow \hat W + \Delta W$、predict ステップ）ことで、mirror は動く重みを追いかける必要がなくなり、共分散測定は初期オフセットの推定（correct ステップ、EMA 平滑化）だけを担えばよい。これは Kolen–Pollack 法 [1994] の機構を forward 統計側で実現したものである。

**(c) 誤差の再帰伝播.** 推定した mirror $\hat W$ と局所傾き $\phi_T'$ を用いれば、厳密な出力誤差から backpropagation とまったく同形の再帰

$$
\hat{\boldsymbol\delta}^{(L-1)}_n
= \Bigl(\hat W^{(L)\mathsf{T}}\, 2(\bar{\mathbf{y}}_n - \mathbf{t}_n)\Bigr) \odot \phi_T'\bigl(\mathbf{d}^{(L-1)}_n\bigr),
\qquad
\hat{\boldsymbol\delta}^{(l)}_n
= \Bigl(\hat W^{(l+1)\mathsf{T}} \hat{\boldsymbol\delta}^{(l+1)}_n\Bigr) \odot \phi_T'\bigl(\mathbf{d}^{(l)}_n\bigr)
$$

を実行でき、重み更新は同じ外積 $\Delta W^{(l)} = -\eta \langle \hat{\boldsymbol\delta}^{(l)} \mathbf{z}^{(l-1)\mathsf{T}} \rangle$ である。第 1 節の backpropagation の式と比べると、唯一の違いは $W^{\mathsf{T}}$ が forward ノイズから測定された $\hat W^{\mathsf{T}}$ に置き換わっている点だけである — この意味で `cov_jac` は backpropagation の**再構成**である。credit はスカラー $L$ に潰されることなく計算グラフに沿って流れるため、第 4.3 節の線形化バイアスを持たず、その誤差は有限サンプル分散（低減可能）のみとなる。統計量は層あたり $O(H^2)$（`cov_deriv` の $O(H)$ に対して）だが、依然として forward-only かつオンライン蓄積可能である。

**(d) 単変量 mirror の条件付きバイアス（注意）.** $\hat W$ の単回帰は層内のactivationが無相関であることを仮定した対角近似であり、層内相関が強い領域では近傍重みの漏れ込み（サンプル数で消えない方向性バイアス）が生じ得る。本研究のレジームでは各ユニット固有の交差ノイズが支配的なため影響は小さい（$r \ge 0.99$）が、厳密な多変量 mirror $\hat W = \mathrm{Cov}(\mathbf{d}, \mathbf{z})\, \mathrm{Cov}(\mathbf{z}, \mathbf{z})^{-1}$ との使い分けを第 7 節で論じる。

#### 4.5 局所オプティマイザ

以上の学習則は勾配推定を与えるものであり、パラメータ更新自体には任意の**重みごとの局所則**を組み合わせられる。SGD（$\theta \leftarrow \theta - \eta \hat g$）はもちろん、Adam の一次・二次モーメント $m, v$ も重みごとに閉じた量なので、Adam を用いても重み輸送は発生せず、ハードウェア親和性（第 6 節）は保たれる。ここで非自明なのは勾配分散との相互作用である: Adam の $1/\sqrt{v}$ 正規化は高分散の勾配を増幅するため、スカラー credit（`cov_deriv`）には有害である一方、`cov_jac` の低分散・準不偏な勾配に対しては backpropagation と同様に有効に働く。実際、`cov_jac` の SGD における残差（$\approx 0.03$）は推定量バイアスではなく最適化の産物であり、局所 Adam に置き換えるだけで backpropagation 水準まで消失する（第 5.2 節）。この対比自体が、`cov_deriv` と `cov_jac` の誤差の性質（バイアス vs 分散）の違いを示す証拠となっている。

### 5. Experimental Results
- **5.1 セットアップ**: sin(x) 回帰、構造 1–64–64–1、t=64、1500 epoch、全手法同一初期重み、cov 系は autograd 完全不使用（no_grad の手動更新）。比較 6 手法（backprop / cov_only / cov_deriv_analytic / cov_deriv_kde / cov_jac_sgd / cov_jac_adam）。
- **5.2 主結果 — `cov_jac_adam` は backprop 級**: 最終 MSE 表（0.0012 vs 0.0011）、学習曲線、fit-check 図（残差付き重ね描き）。cov_jac_sgd の ~0.03 は推定量バイアスではなく最適化アーチファクトであることを sgd/adam 対比で示す。
- **5.3 勾配の不偏性の直接検証**: cov_jac の層別更新と autograd 厳密勾配の cosine 類似度 ≈ 0.9996–1.0000、magnitude ratio ≈ 1.0（未学習・学習途中の両方）。weight mirror の回復精度（Pearson r ≈ 1.0）。
- **5.4 アブレーション**:
  - per_input vs pooled credit（0.03 vs 0.4 — 中心化の決定的重要性）。
  - KDE slope vs 解析的 φ′（一致 → ノイズ分布の解析モデル不要。特に一様ノイズで −(d−c)/r² を使わず一致することを強調）。
  - cov_only vs cov_deriv（局所傾きの寄与）。
  - jac-ema / jac-track の効果。
- **5.5 一様ノイズでの再現**: `--noise uniform` で同じ結論（cov_jac_adam ≈ backprop、KDE=analytic）が成り立つこと。§6 の HW 主張の実験的裏付け。
- **5.6 負の結果 — 外部ノード摂動は二値交差に効かない**: `cov_deriv_gate`（不偏だが高分散）と `cov_deriv_gate_crn`（RNG 状態リセットによりビット厳密な CRN でも改善僅少）。二値活性のパスワイズ微分の退化が原因で、交差の内部密度推定（crossing counting）こそが正しい傾き推定器であること。分量次第で本文短縮＋詳細は Appendix/Supplement。

### 6. Hardware-Oriented Analysis: Why Uniform Noise Makes the System Digital-Friendly
（副次的主結果の節。実装なしでも成立する定性的・アーキテクチャ的議論に限定する）
- **6.1 演算プリミティブの棚卸し**: 学習に必要な統計量は per-unit の running sum 4 種（L, z, Lz, z²）＋ミラー用の Cov(d, z)。必要演算は累算・乗減算・ユニットあたり 1 除算・MAC のみで、転置重みメモリ／第二データパス／位相同期が不要。
- **6.2 一様ノイズの決定的利点**:
  - ノイズ生成: 一様乱数は LFSR で直接生成可能（ガウスは Box–Muller / CLT 近似等の追加回路が必要）。
  - 交差活性: 比較器 2 個＋XOR（二値、乗算不要）。
  - 期待応答は厳密な放物線、局所勾配は**一次式** −(d−c)/r² — LUT すら不要のレベル。さらに KDE slope を使えば解析式自体が不要（xor カウントの差分＝カウンタ演算のみ）。
  - 活性 z が二値 → 外積 δ·zᵀ の乗算が AND/選択に退化、共分散統計もカウンタで実装可能。
- **6.3 定性的リソース比較表**: backprop（転置行列積・重複メモリ・同期）vs 本手法（forward サンプリング＋アキュムレータ）。M サンプルとのトレードオフを正直に記述。
- **6.4 局所性の限界と設計余地**: 単変量ミラーは対角近似（層内相関で方向性バイアス、ただし本レジームでは r≥0.99 と小さい）。多変量ミラー Cov(d,z)Cov(z,z)⁻¹ は厳密だが O(H³) で局所性を損なう。中間案（ブロック対角、RLS＋KP predict、ノイズ場による whitening-by-design）。noise field / recruitment（`cov_deriv_field_gate`）が「更新のゲーティング＝消費電力ゲーティング」に直結する見通し。
- ※ 実 FPGA 実装・電力/面積の定量値は**主張しない**（future work として明示）。

### 7. Discussion
- **7.1 何が新しいか**: weight mirror / KP に必要な摂動が NNN では**毎 forward パスに無料で内在**する — 既知メカニズムの「ネイティブ実現」という位置づけ（過剰主張の回避と新規性の両立）。
- **7.2 生物学的妥当性への含意**（簡潔に）: 転置重みなし・局所統計のみ・確率的発火からの credit assignment。
- **7.3 限界**:
  - toy 回帰 1 タスク・2 隠れ層のみ（分類・深いネットは未検証。再帰でのミラー誤差の複利は開リスク）。
  - readout は局所（厳密）勾配であり backward 除去は隠れ層に限る。
  - 1 更新に t 個の forward サンプルが必要（計算量とのトレード）。
  - 単変量ミラーの条件付き方向性バイアス（弱ノイズ・強相関レジーム）。
  - HW 適性は構成論であり実装実証ではない。
- **7.4 今後の展開**: 分類ベンチマーク、深層化、学習可能な noise field（recruitment と credit の統合）、FPGA プロトタイプ、多変量/RLS ミラーのクロスオーバー診断（層内相関 or Cov(z,z) の条件数を指標に）。

### 8. Conclusion
- forward ノイズの共分散統計だけで backprop の構造を再構成でき、局所 Adam で backprop 級精度に到達。一様ノイズ化により全体がデジタル HW 親和構成になる。ノイズは NNN において推論だけでなく**学習の資源**でもある、で締める。

### Appendices / Supplementary
- **A. 導出**: 一様/ガウス期待応答 φ̄ と φ̄′、KDE slope の antithetic-CRN 解釈、共分散 credit の線形化バイアスの式、単変量ミラーのバイアス項 Σ_{k≠i} W_{jk}Cov(z_k,z_i)/Var(z_i)。
- **B. 負の結果の詳細**: gate / gate_crn の全数値、α 掃引、二値退化の解析（本文 §5.6 を短くする場合の受け皿）。
- **C. 多変量ミラーの standalone 検証**: 弱ノイズ readout・3 隠れ層での r 比較（0.994 で頭打ち vs 1.000）、サンプル数不変性（バイアスの実証）。
- **D. 再現性**: 全ハイパーパラメータ、seed 別結果、コード公開（GitHub リポジトリ）。

---

## 図・表プラン（案）

| # | 内容 | 出典/生成方法 |
|---|---|---|
| Fig.1 | NNN と学習則の概念図: 交差活性、forward サンプル、共分散統計 → weight mirror → 再帰 credit（backprop との対比パス図） | 新規作図 |
| Fig.2 | 交差活性の応答: サンプル z、期待応答 φ̄（ガウス/一様）、KDE slope vs 解析 φ′ | 新規（式から） |
| Fig.3 | 学習曲線（6 手法、log MSE） | `tmp/fncl5_2.py` → `fig_learning_curves.png` |
| Fig.4 | fit-check: sin(x) vs backprop vs cov_jac_adam ＋残差 | `tmp/fncl5_2.py` → `fig_fit_check.png` |
| Fig.5 | 勾配不偏性: 層別 cosine 類似度・magnitude ratio、ミラー回復散布図（Ŵ vs W、二値対照つき） | `tmp/fncl5_3.py` → `fig_mirror_scatter.png`, `fig_grad_cosine.png` |
| Fig.6 | アブレーション棒: per_input vs pooled、kde vs analytic、sgd vs adam、track/no-track | `tmp/fncl5_4.py` → `fig_ablation_bar.png`（uniform 対比は `fncl5_5.py`） |
| Tab.1 | 最終 MSE 一覧（seed 複数、mean±std; gaussian＋uniform） | `tmp/fncl5_2.py` / `fncl5_5.py` → `table_final_mse.md` |
| Tab.2 | HW 比較: backprop vs 提案法（演算・メモリ・データパス・同期）＋一様/ガウスのプリミティブ比較 | 定性表は docs §7 を拡張、定量支援は `tmp/fncl6_1.py` → `table_ops.md`, `table_primitives.md` |
| Fig.7 (opt) | 一様ノイズ HW ブロック図: LFSR → 比較器×2 → XOR → カウンタ/アキュムレータ | 新規作図 |
| Fig.8 (§5.6) | 負の結果: MSE vs α（gate/CRN/基準線）＋CRN 損失差の厳密ゼロ割合 vs α | `tmp/fncl5_6.py` → `fig_gate_mse_vs_alpha.png`, `fig_crn_degeneracy.png` |

## 執筆上の注意（docs §11 を継承）

- **安全な主張**: 「転置重みを介した明示的 backward なしに隠れ層 credit を再構成」「forward サンプルの暗黙的ノード摂動としての再利用」「cov_jac の credit は経験的に不偏（cosine ≈ 1.0）で、局所 Adam により toy タスクで backprop 級の最終精度」。
- **避ける主張**: 「backprop を完全代替」「厳密に backprop 勾配を計算」（near-exact であって bit-exact ではない；ミラーの条件付きバイアスあり）「FPGA での優位性を証明」（実装なし）。
- 投稿前に必要な追加実験は `tmp/fncl5_2.py`〜`fncl5_6.py`, `fncl6_1.py` として整備済み（seed 複数の統計、uniform 再現、勾配不偏性、アブレーション、負の結果、HW 演算計数）。各スクリプトは `--quick` で動作確認可能、生成物は `out/fncl<節>_<小節>/` に保存される。可能なら小規模分類タスク 1 つ追加すると査読が大幅に楽になる。
