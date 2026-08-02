# Noise-modulated Neural Network に基づく報酬変調型強化学習とノイズ場による行動モード形成

## 0. 概要（このレポートの読み方）

本レポートは、Noise-modulated Neural Network（NNN）を強化学習（RL）へ **自然に**（外付けの方策分布・探索スケジュール・転置重み backward・外部 RL アルゴリズムに頼らずに）統合する構想と、その最小実証の記録である。「性能で既存手法に勝つ」ことは目的ではない（§20.14）。核心は、RL の構成要素（方策・探索・局所感度・層間 credit・時間方向 credit・サブネットワーク動員）が NNN の同一ノイズ機構から立ち上がることを示すことにある。

**構成の層**（読む順序）:

- **Part I ― 構想（§1–16）**: 元の設計・理論。forward fluctuation を方策・探索・credit・eligibility に一貫利用し（§2–6）、ノイズ場を行動モード/option として用いる（§7–12）。統合アルゴリズムと実験計画、中心的主張（§13–16）。
- **Part II ― 評価と位置づけ（§17–19）**: 査読的フィードバック（§17）、既存研究との位置づけ（§18）、最大の未定義部だった「場生成パラメータへの credit 経路」の具体化（§19）。
- **Part III ― 実装と実測（§20–21）**: 学習則の自然統合を CartPole で検証（§20：Step A–C ＋ Task #1 の critic 統一）。ノイズ場を行動モードとして検証（§21：L1 addressing・報酬による自律選択・L2 多重化・連続場形成・per-unit σ 検証）。実装はすべて `tmp/`（`tmp/rl/` 共通、`tmp/rl_*` 検証）。
- **結論（§22）**: §20–21 時点の総括と、**最も NNN-native な RL 方式**の同定と詳説。
- **補遺（§23）― 実タスクへの展開（2026-07-20〜23、完結）**: swing-up + balance を題材に、(i) full balance の達成（§23.1–23.3）、(ii) ρ/h ゲートによる漏れなしスキル分離・保護（§23.4/23.7）、(iii) critic の完全 NNN 化（§23.5/23.8–23.9）、(iv) PPO 統合（§23.10）、(v) SAC 統合と PPO との比較（§23.11–23.12）。本補遺で「backprop ゼロの単一 NNN が主要 RL アルゴリズム群を実タスクで駆動できる」ことの検証は一区切りした。
- **付録（§24）― 試行手法一覧**: 本レポートで試したすべての手法を、手法の説明・結果・判定つきの一覧表に整理した早見表。個別の詳細へ入る前の全体把握、または特定の実験の逆引きに使う。

**主要な到達点**（詳細は各節と §22・§23.12）:

1. 転置重み backward・外部 critic・外部 RL アルゴリズムなしに、**forward fluctuation の credit だけで CartPole を学習**（Step A–B, §20.12–13；単一 NNN 統合版 Task #1, §20.17）。credit は autograd の $\nabla_W\log\pi$ を cosine ~0.95 で復元（§20.12）。
2. **単一のノイズ強度 $\sigma$ は計算・探索・制御を同時最大化しない**（SR 対立、§20.16）が、RL は低忠実度 credit に頑健。
3. **ノイズ場が行動をアドレスし**（L1, §21.2）、**報酬で場を自律選択**でき（§21.3）、**共有ユニットに 2 行動が多重化**され（L2, §21.4）、**連続場座標の補間で未学習の中間行動**が現れる（§21.5）。
4. **per-unit σ eligibility は policy-score では ill-posed**（$\mathbb E[\partial\log\pi/\partial\sigma]\approx0$）＝場は低次元 recruitment 座標として学習するのが正しい（§21.6）。
5. **完全 backprop フリー・全特徴学習の単一 NNN が CartPole swing-up + full balance を解く**（actor §23.1、critic 込み §23.9、PPO で安定化 §23.10、SAC でも到達 §23.11）。鍵は EMA mirror + Kolen–Pollack の RL 移植（§23.9）と、「アンサンブル平均 μ の確率性」の統計的会計（§23.10 処方 (i)–(v)）。
6. **ノイズ場によるスキルの分離・保護・合成は ρ/h ゲート（consolidation の動員ダイヤル）で深い層でも厳密に成立**（§23.7：保持 1.000 構成的・drift 0、olap 場で獲得速度も両立）。σ-only recruitment の漏れ（§21.4/§23.4）は機構の限界ではなくゲートの選び方の問題だった。
7. **アルゴリズム選択指針：backprop フリー制約下では PPO 系が構造的に有利**（§23.12：SAC の効率の源泉 reparameterization が制約で禁じられるため。SAC も replay 係留 + noise-deadband 比で実現可能だが到達は約 4 倍遅い）。

否定的結果（§20.14 の較正、§21.1 の SR 対立、§21.6 の σ ill-posed、§23.8 の body 共有 critic、§23.10 v1–v3・§23.11 v1–v4 の設計反復）も含めて正直に記録している。なお §1–16 は当初の構想であり、§17 以降の評価・実測で修正・限定された箇所は各節に明記した（例：§18-C の分散優位の主張は §20.14 で撤回、§23.4 の「漏れ recruitment の限界」は §23.7 で解決済みに更新）。

---

## 1. 目的

Noise-modulated Neural Network（NNN）を強化学習へ接続する際の中心的な狙いは、NNNを既存の actor–critic や PPO における単なる関数近似器として用いることではない。NNN内部で本来生じる確率的揺らぎを、

- 行動生成
- 探索
- 局所感度推定
- 層間 credit assignment
- 時間方向の eligibility trace
- サブネットワークの動員

に一貫して利用することで、強化学習機構そのものをNNNの内部原理として構成することである。

この構想では、NNNのforward fluctuationが瞬間的な行動をsampleすると同時に、その行動に関与した重みとノイズ場要素にeligibilityを残す。後から到来する報酬またはTD誤差が、そのeligibilityを変調する。さらに、ノイズ場は低速な内部状態として、どのサブネットワークが行動生成と学習に参加するかを決定し、その持続と遷移が行動モードまたはoptionを形成する。

役割分担を要約すると、次のようになる。

- **重み**：現在動員されたモードの内部で、どのような状態–行動写像を実現するかを決める。
- **ノイズ場**：どのサブネットワークを機能化し、どの行動モードを実体化するかを決める。
- **eligibility**：直近の行動に各局所要素がどの程度関与したかを保持する。
- **報酬またはTD誤差**：その関与を強化するか抑制するかを決める。

**※ 上の「重み＝写像／ノイズ場＝モード選択」という直交 2 系統の描像は撤回済みである**（`idea_duality.md` §12.3、`docs/idea_core.md` §4.5）。正しくは、ユニット $k$ の $(a_k,\sigma_k)$ は「重みの方向 $\hat a_k$」「共通スケール（ゲージ自由度）」「文脈間の相対 $\sigma$」の 3 成分に分解され、ノイズ場が動かせるのは各ユニットの**動径 1 次元だけ**であって、これは $\theta$ の作用の部分空間である。したがって両者は直交しない。正しい統一的言明は「$\theta$ と $\mathcal P$ は別々の学習機構で動く 2 つの系ではなく、**同一の前向き共分散 credit $\hat g$ に 2 人の消費者がいる**」（外積の相手が $z_{\mathrm{prev}}$ なら $\theta$ の勾配、$-d/\sigma$ なら場の勾配）である。本文書では §10.2 が既にこの正しい形で書かれている。上の箇条書きは、機能面の役割記述としてのみ読むこと。

## 2. NNNのforward fluctuationを方策とする

### 2.1 方策分布を外付けしない

通常の連続制御では、決定論的なニューラルネットワークから平均と分散を出力し、Gaussian policyを別途定義することが多い。NNNでは、このような外付けの確率分布を必須としない。

状態 $s_t$ に対してSample-level NNNを $T$ 個の内部sampleで動作させると、行動readoutから

$$
a_t^{(1)},a_t^{(2)},\ldots,a_t^{(T)}
$$

が得られる。このsample集合そのものを方策分布とみなす。

連続行動では、sample平均とsample共分散を

$$
\mu_t=\frac{1}{T}\sum_m a_t^{(m)}
$$

$$
\Sigma_t=
\frac{1}{T}\sum_m
\left(a_t^{(m)}-\mu_t\right)
\left(a_t^{(m)}-\mu_t\right)^\mathsf{T}
$$

として推定できる。実環境へ適用する行動は、そのうち一つのsample

$$
a_t=a_t^{(m^*)}
$$

とする。残りのsampleは実環境では実行せず、内部統計とcredit推定に利用する。

したがって、同一状態から複数の行動を環境上で試行する必要はない。NNN内部では複数sampleを生成するが、環境へ送る行動は一つだけである。

### 2.2 NNN内部ノイズの複数の役割

この構成では、同じforward fluctuationが同時に以下を担う。

- 行動分布の生成
- 行動探索
- crossing activationの局所微分推定
- covariance weight mirrorの推定
- 行動に対する各層の感度推定
- eligibilityの生成

これは、探索ノイズ、方策分布、勾配推定用摂動を別々に設計する通常の強化学習と異なる。NNNでは、計算を成立させるために必要なノイズが、そのまま探索と学習信号の担体になる。

**※ §20.16・§25.1 の実測により限定される。** (i) 単一の内部ノイズ強度 $\sigma_k$ は、credit 忠実度（低 $\sigma$ を好む）と制御性能（中〜高 $\sigma$ を好む）を同時最大化しない（§20.16）。(ii) §23.1 以降の実タスク実装では、「行動分布の生成」と「行動探索」を実際に担っているのは内部ノイズ $\sigma_k$ ではなく**外付けの探索ノイズ $\sigma_e$**（$a\sim\mathcal N(\mu,\sigma_e^2)$）であり、本項の一体化は行動レベルでは未実現のまま残っている（§25.1 が「最後の外部 scaffolding」として明記）。$\sigma_k$（内部注入ノイズ）と $\sigma_e$（外付け探索ノイズ）の区別は `docs/idea_core.md` §1.2 を参照。

## 3. 方策eligibilityの生成

### 3.1 方策scoreを出力creditとする

方策勾配は一般に

$$
\nabla_\theta J
=
\mathbb{E}\left[
A_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right]
$$

と書ける。

連続行動について、NNNのsample集合を局所的にGaussianで近似し、共分散のパラメータ依存性をいったん無視すると、出力平均に関するscoreは

$$
u_t=\Sigma_t^{-1}(a_t-\mu_t)
$$

となる。

この $u_t$ は教師あり学習における出力誤差に相当するが、この時点では報酬を含まない。意味としては、「今回選ばれた行動が、NNNの平均的な行動からどの方向にどれだけずれていたか」を表す。

### 3.2 forward covarianceによる層間credit再帰

出力score $u_t$ を、NNNのforward-only credit assignmentの始点として用いる。

$$
\delta_t^{(L)}=u_t
$$

$$
\delta_t^{(l)}=
\left(
\widehat W^{(l+1)\mathsf T}\delta_t^{(l+1)}
\right)
\odot
\phi_T'\left(d_t^{(l)}\right)
$$

ここで、

- $\widehat W^{(l+1)}$ はforward fluctuationの共分散から推定したweight mirror
- $\phi_T'$ はcrossing activation自身のnoise sampleから推定した局所微分
- $d_t^{(l)}$ は層 $l$ のpre-activation
- $\delta_t^{(l)}$ は行動scoreがその層へ与える局所credit

である。

この再帰は、転置重みの直接読出しを必要とせず、NNN内部のforward statisticsから構成できる。

### 3.3 瞬間的な重みeligibility

重み $W_{ij}^{(l)}$ に対する瞬間eligibilityは

$$
\psi_{W,t,ij}^{(l)}
=
\delta_{t,i}^{(l)}
 z_{t,j}^{(l-1)}
$$

となる。

より局所的に書けば、

$$
\psi_{W,t,ij}^{(l)}
=
g_{t,i}^{(l)}
\phi_T'\left(d_{t,i}^{(l)}\right)
 z_{t,j}^{(l-1)}
$$

である。ここで $g_{t,i}^{(l)}$ は上位層から再帰的に得られるunit creditである。

これは三要素学習則として解釈できる。

1. 前シナプス活動 $z_{t,j}^{(l-1)}$
2. crossing activationの局所感度 $\phi_T'(d_{t,i}^{(l)})$
3. 行動scoreから得られる大域的credit $g_{t,i}^{(l)}$

ただし、この段階では将来報酬はまだ利用していない。ここで生成されるのは、「現在の行動にこの重みがどの程度関与したか」という痕跡である。

## 4. 時間方向のeligibility traceと報酬変調

### 4.1 遅延報酬への対応

強化学習では、時刻 $t$ の行動が適切だったかどうかが、後の時刻で得られる報酬によって判断される。そのため、瞬間eligibilityを時間方向に蓄積する。

$$
e_{W,t}^{(l)}
=
\gamma\lambda_W e_{W,t-1}^{(l)}
+
\psi_{W,t}^{(l)}
$$

ここで、

- $\gamma$ は割引率
- $\lambda_W$ はtraceの持続性
- $\psi_{W,t}$ は現在の行動に対する瞬間eligibility
- $e_{W,t}$ は過去の行動関与を保持するtrace

である。

### 4.2 TD誤差による変調

価値関数 $V(s)$ を用いる場合、TD誤差は

$$
\Delta_t^R
=
r_t+\gamma V(s_{t+1})-V(s_t)
$$

である。

重み更新は

$$
\Delta W^{(l)}
=
\alpha_W\Delta_t^R e_{W,t}^{(l)}
$$

とする。

この過程は、

$$
\text{NNN fluctuation}
\rightarrow
\text{local eligibility}
\rightarrow
\text{temporal trace}
\rightarrow
\text{reward modulation}
$$

と整理できる。

誤差を即座に逆伝播するのではなく、各局所要素が自分の最近の行動関与を保持し、後から到来した報酬によって可塑化される。

## 5. 離散行動への適用

最初の検証対象としては、二値行動が最も明快である。

行動 $a_t\in\{0,1\}$ とし、NNN sampleから発火確率を

$$
p_t=\frac{1}{T}\sum_m a_t^{(m)}
$$

として推定する。Bernoulli方策の出力scoreは

$$
u_t=
\frac{a_t-p_t}
{p_t(1-p_t)+\epsilon}
$$

である。

したがって、

$$
\nabla_W\log\pi(a_t\mid s_t)
=
u_t\frac{\partial p_t}{\partial W}
$$

となる。$\partial p_t/\partial W$ は、crossing slopeとcovariance weight mirrorを用いてforward statisticsから構成できる。

この場合、Gaussian policy headや明示的な標準偏差パラメータは不要である。crossing activityの発火確率がそのまま方策になる。

CartPoleの左右行動、T-mazeの二者択一、contextual banditなどが初期検証に適している。

## 6. 単一NNNによる行動と価値の共有表現

actorとcriticを完全に別のニューラルネットワークとして構成する必要はない。一つの共有NNNに、行動readoutと価値readoutを設ける。

$$
z_t^{(L-1)}
\longrightarrow
\begin{cases}
a_t & \text{action readout}\\
V(s_t) & \text{value readout}
\end{cases}
$$

行動readoutは、policy-score eligibilityと報酬変調で更新する。

価値readoutは、

$$
L_V=
\left[
V(s_t)-
\left(r_t+\gamma V(s_{t+1})\right)
\right]^2
$$

を対象とする二乗誤差学習として更新できる。

この構成では、一つのNNN内部で、

- action sampling
- value prediction
- exploration
- hidden-layer credit assignment
- eligibility生成

を共有できる。

## 7. ノイズ場を行動モードとして用いる

### 7.1 ノイズ場によるサブネットワーク動員

NNNでは、各ユニットへ与えるノイズ強度の空間分布が、どのユニットを推論と学習へ参加させるかを決める。ノイズがゼロのユニットは出力と局所微分がゼロとなり、実効的にネットワークから切り離される。

**※ この無条件形は §21.4・§23.4 の実測により限定される（`docs/idea_core.md` §2.1・§3.4・§3.5）。** 「$\sigma_k=0$ でユニットが切り離される」は、**理想形**（同一の $d$ に対する 2 回の独立抽選 $z=\mathbf 1[(d\ge\eta_1)\ \dot\vee\ (d\ge\eta_2)]$）では任意の層で厳密に正しい（$p=\delta$ なら 2 つの比較が恒等的に一致するので $z\equiv0$）。しかし実装は 2 回の抽選を「$T$ 軸上の隣接サンプル」で実現する巡回 XOR であり、比較されるのは異なる 2 つの前活性 $d_t,d_{t+1}$ になるため、理想形と一致するのは **$d$ が $T$ 方向に定数である第 1 隠れ層に限られる**。層 $l\ge2$ では上流サンプルゆらぎが $\pm h_k$ をまたぐので $\sigma_k=0$ でも交差が続く（実測 $\nu=0.11$–$0.17$ のプラトー、§21.4／§23.4）。本レポートの実験はすべてサンプルレベルなので、**運用上の切り離し規則は動員ダイヤル $\rho_k$** である（$\sigma_k=\rho_k\sigma_0$、$h_k=h_0/\rho_k$、$\rho_k\in[0,1]$。完全な退役は $\sigma_k\leftarrow0$・$h_k\leftarrow H_{\mathrm{DEAD}}=10^6$・$W^{(l+1)}_{:,k}\leftarrow\mathbf 0$ の三点操作）。ρ/h ゲートなら任意の層で厳密沈黙が成立する（§23.7）。また参加度は $\sigma_k$ の絶対値ではなく**交差率 $\nu_k=\mathbb E[z_k]$**（ゲージ不変）で測る（`docs/idea_core.md` §3.3・§4.3）。

したがって、ノイズ場 $P$ を切り替えると、重み集合 $W$ を共有したまま異なるサブネットワークを動員できる。

$$
\pi_o(a\mid s)
=
\pi(a\mid s;W,P_o)
$$

ここで $P_o$ は行動モード $o$ に対応するノイズ場である。

重みが各モード内の技能を保持し、ノイズ場が現在利用する技能の組合せを選択する。ただしこれは機能面の分業の記述であって、$\theta$ と $\mathcal P$ が直交する 2 系統だという意味ではない（§1 の※、`docs/idea_core.md` §4.5）。

### 7.2 離散的なノイズ場prototype

最も単純な構成は、複数のノイズ場prototype

$$
P_1,P_2,\ldots,P_K
$$

を持つことである。時刻 $t$ では内部モード $o_t$ が選択され、

$$
P_t=P_{o_t}
$$

としてNNNを動作させる。

ただし、各prototypeに「探索」「回避」などの意味ラベルを事前に与えると、単なる手設計optionになる。研究上はprototype番号に意味を与えず、報酬学習後に、各場でどのような行動様式が生じたかを解析する方が強い。

### 7.3 連続的なfield state

よりNNNらしい構成は、離散option番号ではなく、低次元の連続field座標

$$
c_t\in[0,1]^V
$$

を内部状態として持たせることである。

仮想ノイズ空間におけるユニット位置を $u_k^{(l)}$ とし、field座標 $c_t$ から各ユニットのノイズ強度を

$$
\sigma_{t,k}^{(l)}
=
\sigma_{\max}
G\left(u_k^{(l)};c_t,\tau\right)
$$

として生成する。

ここで、

- $G$ はGaussian bumpなどの局所場
- $\tau$ は場の広がり
- $c_t$ は現在の場の中心

である。

$c_t$ を連続的に動かすと、使用するサブネットワークも連続的に変化する。field空間で近い位置は部分的に重なるサブネットワークを動員するため、近い行動モード間の滑らかな遷移を表現できる。

## 8. field stateの持続とoption形成

行動モードとして機能させるには、ノイズ場が毎時刻無秩序に切り替わらず、一定期間持続する必要がある。

例えば、

$$
c_{t+1}
=
(1-\kappa)c_t
+
\kappa\widetilde c_t
+
\xi_t
$$

とする。

ここで、

- $c_t$ は現在のfield state
- $\widetilde c_t$ は現在の観測から提案された次のfield
- $\kappa\ll1$ は変化速度
- $\xi_t$ は小さな探索揺らぎ

である。

この低速ダイナミクスにより、fieldは行動より長い時間スケールで持続する。結果として、サブネットワークの動員状態が一定期間維持され、optionに相当する時間的まとまりが形成される。

option終了は、例えば、

- 現在のモードのadvantageが低下したとき
- field座標の移動量が閾値を超えたとき
- 内部状態または環境イベントが終了条件を満たしたとき

と定義できる。

ただし、明示的な終了判定器を外付けすると通常のhierarchical RLに近づく。最初はfieldの持続ダイナミクス自体からモード切替を定義する方が、NNN固有の構成として明確である。

## 9. 外付けgating networkを避ける構成

状態からノイズ場を生成するために別のMLPを設けると、mixture-of-expertsのgating networkに見えやすい。これを避けるには、同一NNN内部を機能的に二つに分ける。

### 9.1 Tonic modulatory core

少数のユニットには常に基礎ノイズを与える。この部分は常時動作し、観測と現在のfield stateから次のfield座標候補を生成する。

**※ 「常時動作」の条件は $\sigma_k$ の絶対値の小ささではない**（`docs/idea_core.md` §3.3・§4.3・§4.8）。$\sigma_k$ の絶対値はゲージ依存量であり、$(w_k,b_k,\sigma_k,h_k)\mapsto\alpha_k(w_k,b_k,\sigma_k,h_k)$ で出力を変えずに任意に伸縮できるので、それ自体は参加・不参加の根拠にならない。core が常時動作するための条件は**交差率 $\nu_k>0$**、すなわち $h_k/\sigma_k$ が不感帯に入らないことである。したがって core は「小さな $\sigma$ を持つユニット」ではなく「$\rho_k$ を高く保ち $\nu_k>0$ を維持するユニット」として指定するのが正しい。

$$
(s_t,c_t)
\longrightarrow
\widetilde c_t
$$

### 9.2 Field-recruited policy body

modulatory coreが生成したfieldによって、大部分のユニットへのノイズ強度を決定する。

$$
c_t
\longrightarrow
P_t
\longrightarrow
\text{recruited policy subnetwork}
\longrightarrow
a_t
$$

全体は一つのNNNとして構成する。

因果順序を明確にするため、時刻 $t$ で生成したfieldを次時刻に適用する。

$$
c_{t+1}=f_{\mathrm{mod}}(s_t,c_t)
$$

この構成により、同一時刻内でfieldがfield自身を生成する循環を避けられる。

## 10. ノイズ場自身のeligibility

**※ 本節の学習則（per-unit $\sigma_k$ を policy score で credit する）は §21.6 の実測により棄却されている。** 単一 pass の $\partial\log\pi/\partial\sigma$ はノルム ~60 と大きいが独立 2 pass 間の cosine ≈ $-0.04$、200 pass 平均でノルムはほぼ 0、すなわち $\mathbb E[\partial\log\pi/\partial\sigma]\approx0$ であり、$-d/\sigma$ 形も forward 推定形もこの gold に対し cosine ≈ 0 だった。動作点近傍では per-unit の $\sigma_k$ は平均方策を動かさず、分散（探索幅）だけを変えるため尤度スコアに映らない（§17.2-2 の指摘の実測版）。**正しくは、場は per-unit $\sigma_k$ ではなく低次元の recruitment 座標として学習する**（§19・§21.5 で成立、`docs/idea_core.md` §4.5 の「$\mathcal P$ が動かせるのは動径 1 次元だけ」と整合）。以下は棄却された定式の記録として残す。ただし §10.2 の「同一 credit の 2 消費者」という構造そのものは正しく、正典側でもそのまま採用されている（`docs/idea_core.md` §5.8）。

### 10.1 ノイズ強度に対する局所感度

ノイズ強度 $\sigma_{t,k}^{(l)}$ が行動分布を変えるなら、重みと同様にノイズ場にもpolicy eligibilityを定義できる。

$$
\psi_{\sigma,t,k}^{(l)}
=
g_{t,k}^{(l)}
\frac{\partial \bar\phi
\left(d_{t,k}^{(l)};\sigma_{t,k}^{(l)}\right)}
{\partial \sigma_{t,k}^{(l)}}
$$

スケール族のnoise distribution $p(\xi)=\frac1\sigma q(\xi/\sigma)$ では、

$$
\frac{\partial\bar\phi}{\partial\sigma}
=
-\frac{d}{\sigma}\bar\phi'(d)
$$

を利用できる。

**※ 成立条件を明示する（`docs/idea_core.md` §5.8・§4.7）。** この恒等式が**厳密に**成立するのは、交差半幅を $h_k=c_h\sigma_k$ とノイズ scale へ連動させた体制（体制 (a)。応答が $\bar z=g(d/\sigma)$ となり $c_h$ の値によらず厳密）である。$h$ を大域ハイパーパラメータとして固定する体制 (b) では厳密には成り立たず近似にとどまる（§17.2-7 と同じ論点）。さらに体制 (b) には**構造バイアス**がある：推定量 $\langle\hat g\,\phi_T'\,(-d/\sigma)\rangle$ は $h$ 固定でも Euler 恒等式 $\partial L/\partial\log\sigma_k+(w_k^{\mathsf T}\partial L/\partial w_k+b_k\partial L/\partial b_k)=0$ を**恒等的に満たしてしまう**ため、モデルが持たない対称性を推定量が強制することになり、このバイアスはサンプル数を増やしても消えない。§20 以降の実測はすべて $h=0.15$ 固定（体制 (b)）で走っているので、本節の $\psi_\sigma$ を定量的に使う場合はこの点を明示的に扱う必要がある。

したがって、

$$
\psi_{\sigma,t,k}^{(l)}
=
g_{t,k}^{(l)}
\left(
-\frac{d_{t,k}^{(l)}}
{\sigma_{t,k}^{(l)}}
\right)
\phi_T'\left(d_{t,k}^{(l)}\right)
$$

となる。

### 10.2 重みeligibilityとの共通構造

重みeligibilityは

$$
\psi_{W,t,ij}^{(l)}
=
g_{t,i}^{(l)}
\phi_T'\left(d_{t,i}^{(l)}\right)
 z_{t,j}^{(l-1)}
$$

である。

ノイズ場eligibilityは

$$
\psi_{\sigma,t,i}^{(l)}
=
g_{t,i}^{(l)}
\phi_T'\left(d_{t,i}^{(l)}\right)
\left(
-\frac{d_{t,i}^{(l)}}
{\sigma_{t,i}^{(l)}}
\right)
$$

である。

両者は、最後の局所因子だけが異なる。

- $z_{\mathrm{prev}}$ を掛ければ重みeligibility
- $-d/\sigma$ を掛ければノイズ場eligibility

となる。

つまり、同一のforward creditを、重みとノイズ場が異なる局所座標で利用する。

### 10.3 ノイズ場の時間traceと更新

ノイズ場にも時間traceを持たせる。

$$
e_{\sigma,t}
=
\gamma\lambda_\sigma e_{\sigma,t-1}
+
\psi_{\sigma,t}
$$

更新は

$$
\Delta\sigma
=
\alpha_\sigma\Delta_t^R e_{\sigma,t}
$$

とする。

これにより、報酬は同じcredit原理によって、

- 現在のモード内部で何を計算するか
- どのサブネットワークを動員するか

の双方を学習できる。

## 11. 絶対ノイズ強度と相対的field pattern

ノイズ強度を自由に学習すると、重みnormの変更とノイズscaleの変更が同じ入出力変化を生む場合がある。特にcrossing幅をノイズscaleへ比例させる構成では、絶対的なscaleが重みnormと冗長になりやすい。

**※ 本節は交差半幅を連動させる体制（$h_k=c_h\sigma_k$、`docs/idea_core.md` §4.7 の体制 (a)）を前提にしている。** この体制では $(w_k,b_k,\sigma_k,h_k)\mapsto\alpha_k(w_k,b_k,\sigma_k,h_k)$（$\alpha_k>0$、ユニットごとに独立）が出力をサンプルパスごとにビット単位で不変に保つ厳密な局所ゲージ対称性となる。なお **§20 以降の実測はすべて $h=0.15$ 固定（体制 (b)）で走っており**、そこではゲージは存在しない。本節の冗長性の議論を §20 以降の実験にそのまま外挿してはならない（体制の切り替えについては §17.2-7 の※、§20.5 の初期ハイパーも参照）。

行動モード形成に重要なのは、絶対的なノイズ量よりも、ユニット間でどこへ相対的にノイズを配分するかである。

そこで、optionごとのノイズ強度を

$$
\sigma_{o,k}=\sigma_0 q_{o,k}
$$

とし、

$$
q_{o,k}\ge0,
\qquad
\sum_k q_{o,k}=B
$$

という固定noise budgetを設ける。

この制約には次の利点がある。

1. 全ユニットのノイズが同時に増減する退化を防ぐ。
2. 各fieldが限られた計算資源を異なる場所へ割り当てるようになる。
3. active-unit数、計算量、消費電力との対応を明確にできる。〔※ **誤り。撤回する。**〕
4. field間の違いを絶対scaleではなく相対的な動員patternとして解釈できる。

**※ 上の budget 定式はゲージ依存であり、そのままでは目的関数・評価指標に使えない**（`docs/idea_core.md` §4.3・§4.8）。理由は 2 つある。

- **(i) 制約が固定するのは大域スケール 1 次元だけである。** 体制 (a) のゲージ群は $(\mathbb R_{>0})^{\sum_l H_l}$ でユニットごとに独立なので、$\sum_k q_{o,k}=B$ という 1 本のスカラー制約はゲージを固定しない。$\{q_{o,k}\}$ の**配分そのもの**が、出力を一切変えずに（$\alpha_k$ を選んで重みノルムで補償するだけで）任意に組み替えられる。したがって「どこへノイズを配分したか」は、それ自体では場の実体ではない。
- **(ii) 利点 3 は正典が名指しで禁じた罠である。** $\sigma_k$ の絶対値・$\lVert w_k\rVert$ 単独・$\mathrm{mean}(\mathcal P)$ はいずれもゲージ依存量であり、「$\sigma$ が小さいユニットの電源を落とす」型の消費電力ゲーティングは、タスク損失を変えずにゲーティング指標だけを動かせてしまう。実測でも、ゲージ変換により出力は $\max|dy|=0$ のまま $\mathrm{mean}(\mathcal P)$ が $0.5016\to1.1033$ と倍近く動いた（`idea_duality.md` §12.6）。

**正しい書き直し（ゲージ不変な形）。** 本節の意図（「絶対量ではなく相対的な動員パターンが本体」）は、次の不変量で述べれば体制 (a)・(b) のいずれでも成立する。

$$
\nu_{o,k}=\mathbb E[z_k\mid P_o]\in[0,0.5],
\qquad
\sum_k \nu_{o,k}=B_\nu \quad(\text{動員予算}),
\qquad
\tilde s_{o,k}=\log\sigma_{o,k}-\frac1{K}\sum_{o'}\log\sigma_{o',k}
$$

- **動員予算は交差率 $\nu$ で張る。** $\nu_k$ は $(d,h,\sigma)$ の 0 次同次関数でゲージ不変であり、「そのユニットが計算に参加しているか」の唯一正しい尺度である。電力・active-unit 数・計算量に対応づけてよいのも $\nu$ の方であり、**PE を落としてよいのは $\nu_k=0$ のユニットだけ**である（$\sigma_k$ が小さいことは根拠にならない）。実装上は $\nu_k=0$ を厳密に作る手段が動員ダイヤル $\rho_k$ であり（§7.1 の※）、$\sum_k\nu_{o,k}$ は $\{\rho_{o,k}\}$ を通じて制御する。
- **field 間の違いは文脈間の相対場で測る。** $\tilde s_{o,k}$（各ユニットについて option 間で $\log\sigma$ を中心化したもの）はゲージ不変であり、単一 option しかない場合は恒等的に 0 になる ―― これは「単一文脈では場の自由度は 100% 冗長」という事実の正しい反映である。すなわち**場の実体は option 間の相対量にしかない**。
- 副次的に使える不変量として、実効ゲイン $\gamma_k=\lVert[w_k,b_k]\rVert/\sigma_k$ と比 $h_k/\sigma_k$ がある。

field分化を示すためにoverlap penaltyを直接入れると、分化そのものが正則化の自明な帰結になる。主結果では、

- 固定noise budget〔※ 上記のとおり $\nu$ 予算 $\sum_k\nu_{o,k}=B_\nu$ として実装すること〕
- field persistence
- option switching cost
- 環境報酬

だけを与え、結果としてfieldが分化するかを検証する方が強い。

## 12. 重みとノイズ場の時間スケール

重みとfieldを同じ速度で更新すると、方策本体とサブネットワーク選択が同時に動き、学習が不安定になりやすい。

自然な構成は、二つの時間スケールを分けることである。

### 12.1 高速に更新する量

各環境stepで更新する。

- 行動readout
- option内方策の重み
- value readout
- eligibility trace

### 12.2 低速に更新する量

option終了時またはepisode終了時に更新する。

- field prototype
- field中心
- field幅
- field生成パラメータ
- option切替特性

一般に、

$$
\alpha_P\ll\alpha_W
$$

とする。

ただし、field state $c_t$ 自体は行動中に変化してよい。低速にすべきなのは、fieldを生成するパラメータやprototypeである。

## 13. 統合アルゴリズム

時刻 $t$ における処理を、以下のように構成する。

1. 内部field state $c_t$ からノイズ場 $P_t$ を生成する。
2. $P_t$ の下でNNNを $T$ sample動作させる。
3. sample集合から行動分布を推定する。
4. そのうち一つの行動 $a_t$ を実環境へ適用する。
5. policy score $u_t$ を計算する。
6. covariance weight mirrorとcrossing slopeを用いて、score creditを各層へ再帰する。
7. 重みeligibility $\psi_{W,t}$ を生成する。
8. ノイズ場eligibility $\psi_{\sigma,t}$ を生成する。
9. それぞれを時間traceへ蓄積する。
10. 報酬 $r_t$ と次状態 $s_{t+1}$ を得る。
11. TD誤差 $\Delta_t^R$ を計算する。
12. $\Delta_t^R e_{W,t}$ により重みを更新する。
13. $\Delta_t^R e_{\sigma,t}$ によりノイズ場またはfield生成パラメータを更新する。
14. fieldの低速ダイナミクスに従って $c_{t+1}$ を生成する。

**※ 手順 13 は §21.6 の実測により限定される。** 「ノイズ場」の直接更新（per-unit $\sigma_k$ を $e_{\sigma,t}$ で動かす §10 の路線）は policy score では ill-posed であり棄却された（$\mathbb E[\partial\log\pi/\partial\sigma]\approx0$）。生き残るのは「field 生成パラメータ」側、すなわち**低次元の場座標**を場レベルの policy gradient で動かす路線であり（§19・§21.5 で実証）、これは `docs/idea_core.md` §4.5 の「$\mathcal P$ が動かせるのは各ユニットの動径 1 次元だけ」とも整合する。手順 13 は後者に読み替えること。

この構成では、環境報酬をforward sample間の共分散へ直接利用する必要はない。先にNNN内部でeligibilityを生成し、後から到来した報酬でそれを変調する。

## 14. 実験計画

### 14.1 第1段階：報酬変調型eligibilityの検証

CartPoleなどの二値行動環境を用いる。

比較条件として、

- backpropagationを用いる標準actor–critic
- 真の転置重みを使うNNN eligibility
- covariance weight mirrorを使うNNN eligibility
- $\lambda=0$ の瞬間eligibility
- $\lambda>0$ の時間trace
- pooled reward covariance
- policy-score eligibility

を設ける。

評価項目は累積報酬だけでなく、以下を含める。

- autogradで計算した $\nabla_W\log\pi$ とのcosine similarity
- weight mirrorの推定誤差
- trace長 $\lambda$ の影響
- sample数 $T$ の影響
- reward delayに対する頑健性
- forward sample分散と学習安定性

### 14.2 第2段階：ノイズ場による行動モード形成

Foraging、Threat avoidance、Shelteringが同一環境内で必要となる課題が適している。

ただし、行動モードの教師ラベルは与えない。入力として、

- food level
- threat proximity
- shelter proximity
- 過去の報酬
- 現在の内部field state

などを与える。

学習後に、

- field位置と行動様式の対応
- fieldの滞在時間
- behavioral transitionとfield transitionの一致
- field間のsubnetwork overlap
- 潜在的行動モードとfield stateのmutual information
- fieldを固定したときの行動変化
- 重みを固定してfieldだけを切り替えたときの行動変化

を解析する。

特に、

> fieldを固定すると一つの行動様式へ留まり、fieldを切り替えると同一の重み集合のまま別の行動様式へ遷移する

ことを示せれば、ノイズ場をoptionまたは行動モードとして解釈する根拠が強くなる。

### 14.3 第3段階：重みとfieldの共同学習

最後に、重みeligibilityとfield eligibilityを同一のforward creditから生成し、両者を報酬変調する。

比較条件として、

- 重みのみ学習
- fieldのみ学習
- 重みとfieldの共同学習
- field固定・重み学習
- 重み固定・field学習
- overlap penaltyあり／なし〔※ 定義を差し替えること。下記〕
- 固定noise budgetあり／なし〔※ $\nu$ 予算として実装すること。§11 の※〕

を設ける。

**※ overlap penalty をゲージ依存量で定義しないこと**（`docs/idea_core.md` §4.3・§4.8）。場の重なりの慣用的な指標 $\cos(\mathcal P_A,\mathcal P_B)$ は $\sigma$ ベクトルの絶対値に依存するゲージ依存量であり、これを目的関数に入れると、**タスク損失を一切悪化させずに正則化項だけを下げるゲージ方向**（各ユニットの $\alpha_k$ を選んで $\sigma$ を伸縮し、重みノルムで補償する）が存在するため、最適化として無意味になる。実測でも、出力がビット単位で不変なゲージ変換で overlap が $0.8305\to0.8720$ と動いた（`idea_duality.md` §12.6）。必要なら次のゲージ不変な形で定義し直す。

$$
\mathrm{Ovl}_\nu(A,B)=\sum_k \nu_{A,k}\,\nu_{B,k}
\qquad\text{または}\qquad
J(A,B)=\frac{|\mathcal A_A\cap\mathcal A_B|}{|\mathcal A_A\cup\mathcal A_B|},\quad
\mathcal A_o=\{k\mid\nu_{o,k}>0\}
$$

（後者は活性集合の Jaccard 係数で、§21.4 が `Jaccard(active)=0.41` として既に採用している測り方である。）

重要なのは、単に報酬が高いことだけでなく、重みとfieldが異なる役割を担っているかを示すことである。

## 15. 中心的な学術的主張

この構想の中心は、NNNを既存強化学習アルゴリズムのpolicy networkとして置き換えることではない。

中心的な主張は次のように整理できる。

> NNNのforward fluctuationは、行動をsampleすると同時に、その行動に関与したsynapseとnoise-field要素へeligibilityを残す。後から到来する報酬またはTD誤差は、それらの局所痕跡を変調する。noise fieldは低速な内部状態として、どのサブネットワークが行動生成と学習に参加するかを決定し、その持続と遷移が行動モードを形成する。

この構成では、

- 方策分布
- 探索
- 局所感度
- 層間credit assignment
- 時間方向credit assignment
- サブネットワーク動員

が、NNN内部の同一ノイズ機構に統合される。

## 16. 研究上の名称候補

内容を比較的正確に表す名称として、以下が考えられる。

- Reward-modulated field-gated covariance learning
- Noise-field-gated eligibility learning
- Reward-modulated covariance eligibility in NNNs
- Field-recruited reinforcement learning in NNNs

このうち、学習則を前面に出すなら **Reward-modulated field-gated covariance learning**、行動モード形成を前面に出すなら **Noise-field-gated eligibility learning** が適している。

---

## 17. 内容へのフィードバック（2026-07-20 追記）

このセクションは、既存実装（`nnn/credit.py`, `nnn/activation.py`, `nnn/noise_field.py`）と関連する3系統の検討（`draft_nce.md` の学習則、`docs/idea_neuromod.md` のノイズ場＝神経修飾場、`docs/idea_consolidation.md` の動員ダイヤルによる資源制御）に照らした査読的コメントである。構想の骨格そのものは支持できる。以下は「どこが実装済み資産で強く、どこに理論的な穴と設計上の緊張があるか」を切り分けるためのものである。

### 17.1 理論的に最も強い部分（構想の芯として残すべき）

1. **forward-covariance credit の再利用は理論リスクが低い（§3.2–3.3）。** 出力 score $u_t$ を top-level $\delta$ として層間へ再帰する操作は、`credit.py` の `cov_weight`（weight mirror）と `activation.py` の `CrossingSample.backward`（$\phi_T'$ の KDE 推定）で **すでに検証済みの Jacobian 推定そのもの** である。実際、教師あり損失の出力誤差を policy score $u_t=\Sigma_t^{-1}(a_t-\mu_t)$ に差し替えるだけで、$\psi_{W,t}=\delta\, z_{\mathrm{prev}}$ は forward 統計から推定した $(\partial\mu/\partial W)^\mathsf{T}u_t=\nabla_W\log\pi$（$\Sigma$ をパラメータ非依存とみなす近似のもとで）に一致する。ここは `draft_nce.md` の結果の直接転用であり、新規性は「score を何にするか」だけに局在する。RL 化の理論的な足場として堅い。

2. **$\psi_W$ と $\psi_\sigma$ の共通構造は論文のフックになる（§10.2）。** $g_t\,\phi_T'$ を共有し、局所座標だけが $z_{\mathrm{prev}}$（重み）か $-d/\sigma$（ノイズ場）かで分かれる。これは `credit.py` で `covariance_credit`（ユニット credit）と `cov_weight`（weight mirror）が同一の共分散構造を共有するのと同じ美学であり、「同一の forward credit を異なる局所座標で読む」という一文は主張として明快である。

3. **node perturbation との機構的差（§13 末尾・§14）。** 「環境報酬を forward sample 間の共分散へ直接使う必要はない」という設計は、報酬とノイズの相関を各ノードで直接取る node perturbation（後述）と対照的で、weight mirror で層別 credit を構成する点が異なる。〔改訂: 当初これを「分散優位」として前面に出す方針だったが、§20 の実測で優位は示せず撤回した。node perturbation は打ち負かす相手ではなく ablation として扱う（§20.14）。〕

### 17.2 詰めるべき論点（設計上の緊張と未定義部）

1. **policy score の自己参照バイアス（§3.1）。** 実行行動 $a_t=a_t^{(m^*)}$ は、$\mu_t,\Sigma_t$ を推定した同じ $T$ sample の一つである。empirical $\mu_t$ を使う限り $(a_t-\mu_t)$ は $1/T$ オーダで縮む縮小バイアスを持つ。**$m^*$ を除いた $T-1$ sample で $\mu,\Sigma$ を推定する hold-one-out** を既定にするのが安全。

2. **共分散のパラメータ依存を捨てる非対称性（§3.1 vs §10）。** $W$ については $\partial\Sigma/\partial\theta$ を無視するのに、$\sigma$ については別途 $\psi_\sigma$ を作っている。しかも $\psi_\sigma$ は「**平均応答 $\bar\phi$ の $\sigma$ 感度**」であって「**探索共分散 $\Sigma$ の $\sigma$ 感度**」ではない。つまり報酬が動かすのは recruitment（平均ゲート）であって探索幅そのものではない。この区別を明示しないと「報酬で探索温度を学習している」と誤読される。主張範囲を「場は動員する部分網を選ぶ」に留め、探索温度の学習は別問題として切り離すべき。

3. **credit 用ノイズと探索用ノイズの同一視は、売りであると同時に制約（§2.2）。** 同じ $\sigma$ が (a) weight mirror / crossing slope の推定精度（SR 最適を持つ）と (b) 行動探索の温度を同時に決める。両者の最適が一致する保証はない。noise budget（§11）は絶対 scale を固定するが、**推定精度と探索の二律背反** は残る。「計算に必要なノイズがそのまま学習信号になる」を主張として保つなら、この2つの最適が衝突しない $\sigma$ 領域の存在を実験で示す必要がある（§17.3 の「RL 版 SR 曲線」）。

4. **online での weight-mirror 分散（§3.2 の実務）。** `cov_weight` は入力方向 $N$ でプールして分散を抑える設計だが、online RL では各 step で state が1個（$N=1$）しかなく、$T$ sample のみで mirror を推定すると高分散になる。対策は既存コード資産で解ける。**mirror を step 間で EMA 保持し、`ManualOpt.update` が返す既知の重み減分で Kolen–Pollack 追跡する**（`credit.py` のコメントに既にこのフックがある）。重みは低速変化なので、mirror を毎 step ゼロから推定する必要はない。

5. **field 生成パラメータへの credit 経路が未定義（§9・§12.2）。** $\psi_\sigma$ は $\sigma$ を直接 credit するが、$\sigma$ は field 座標 $c$ の Gaussian-bump 写像（`noise_field.noise_pattern`）で決まり、$c$ は modulatory core $f_{\mathrm{mod}}(s,c)$ が生成する。**$\Delta_t^R$ が $f_{\mathrm{mod}}$ の重みへ届く経路（$\partial\sigma/\partial c\cdot\partial c/\partial\theta_{\mathrm{mod}}$）が書かれていない。** これが最大の未定義部。二案:
   - (a) $c$ を「上位行動」とみなし、field レベルで同じ policy-gradient 機構を適用する（option-critic の intra-option / termination gradient に相当）。
   - (b) $\partial\sigma/\partial c$ を forward 推定して chain する。
   構想としては (a) の方が「同一原理の階層適用」として一貫し、§18-G の option-critic との対比も明確になる。

6. **value 側 credit の統一（§6）。** value readout を MSE で更新と書くが、隠れ層 credit も **同じ weight-mirror 再帰（value error を top-level $\delta$）** で通すべき。そうしないと「すべて forward noise から」の主張が actor だけの話に縮む。actor score $u_t$ と critic error $V-\text{target}$ の2つを top-level $\delta$ として同じ再帰に流す、と明記するのが良い。

7. **$-d/\sigma$ 恒等式の成立条件（§10.1）。** $\partial\bar\phi/\partial\sigma=-(d/\sigma)\bar\phi'(d)$ が **厳密に成り立つのは、crossing 幅 $h$ をノイズ scale $\sigma$ に比例させた coupled-width regime**（`activation.py` の `HatApproxCrossingAnalytic` coupled mode、または `ParabolicCrossingAnalytic` の `radius`$=\sigma$）に限る。応答が $d/\sigma$ のみの関数になるためである。$h$ 固定なら近似にとどまる。$\psi_\sigma$ を厳密に使うなら「coupled-width 前提」を明記すべき。これは §11 の「crossing 幅をノイズ scale へ比例させる構成では絶対 scale が重み norm と冗長」という記述とも整合する（同じ regime を仮定している）。

8. **「gating を避ける」の正直な位置づけ（§9）。** modulatory core は結局 $s,c\to\tilde c$ の学習写像であり、soft-gating であること自体は否めない。mixture-of-experts との **真の差** は「core が出力するのは per-expert 重みではなく低次元 field 座標であり、field は出力を混合せずノイズ（recruitment / gain）を変調する」点にある。「gating network がない」と主張するより、「**gating が neuromodulatory field の形をとる（出力混合でなくノイズ変調で作用する）**」と正直に述べる方が、「単一物理量が選択と計算を兼ねる」という主張（`docs/idea_neuromod.md`）とも噛み合って強い。

### 17.3 実験計画への補足（§14 に足すべきもの）

- **RL 版 SR 曲線を第1段階に追加。** $\sigma$ を掃引し、(i) weight mirror の autograd 勾配に対する cosine similarity と (ii) 到達報酬が、同じ $\sigma$ 領域で最適化されるかを見る。§17.2-3 の二律背反の直接検証であり、「ノイズは鍵ではなく最適強度をもつ機能的資源である」という SR の内点最適曲線（`docs/idea_neuromod.md` §6）の RL 版になる。`--model sample` と `--model analytic` の対照もそのまま持ち込める。
- **node perturbation（pooled reward covariance）との variance-vs-network-size 比較を主軸に。** Fiete–Seung 型の $O(N)$ 分散劣化に対し、層別 credit がどれだけ改善するかが最も説得力のある定量結果になる。これを §14.1 の評価項目の筆頭にすべき。
- **「重み固定・field だけ切替で行動様式が変わる」（§14.2 の強い主張）は L1（アドレッシング）と同一実験。** RL 文脈では「学習後に field を固定 vs 切替」で示せる。`docs/idea_neuromod.md` §1.1 の L1 と相互引用でき、証拠を共有できる。

---

## 18. 既存研究との位置づけ（2026-07-20 追記）

この構想は独立した新機構ではなく、**既存の複数系譜の交点に、NNN の forward-noise 原理で橋を架ける** ものとして位置づけるのが正確かつ防御的である。以下、系譜ごとに「代表研究」と「NNN-RL の差分」を対にする。

### A. 三要素則・神経修飾可塑性
- 代表: Frémaux & Gerstner (2016) の reward-modulated Hebbian / three-factor learning、Gerstner et al. (2018) の eligibility trace レビュー、Williams (1992) REINFORCE。
- 差分: $\psi\cdot\Delta_t^R$（§3–4）は **まさに三要素則**（前シナプス活動 × 局所感度 × 大域変調）である。NNN の独自性は、大域 credit $g_t$ を backprop でも固定ランダム feedback でもなく、**forward covariance から層別に推定** する点にある。三要素則の「大域信号をどう各シナプスへ配るか」という未解決問題に対する、forward-only の具体的回答。

### B. 方策勾配・eligibility trace actor-critic
- 代表: Sutton & Barto の actor-critic with eligibility traces、TD($\lambda$)。
- 差分: §4 の $e_{W,t}=\gamma\lambda_W e_{W,t-1}+\psi_{W,t}$ と $\Delta W=\alpha\Delta_t^R e_{W,t}$ は、この定式の **$\nabla\log\pi$ を forward 統計に差し替えたもの**。時間方向 credit（trace）の枠組みは既存のまま流用でき、新規性は空間方向 credit の作り方に集約される。この切り分けを明示すると、査読者に「既知の骨格＋新規の一点」として読ませられる。

### C. node / weight perturbation（最近傍の神経 RL）
- 代表: Fiete & Seung (2006) node perturbation、Werfel, Xie & Seung (2005) の学習則の収束解析、Williams の REINFORCE 系。
- 差分: node perturbation は各ノードのノイズと大域報酬の相関で勾配を推定する。NNN は同じ forward ノイズを使いつつ weight mirror（`cov_weight`）で層別 credit を構成する点が機構的に異なる。**ただし「分散でこれに勝つ」という当初の主張は §20 の実測で成り立たなかったため撤回する（§20.14）。** CartPole 規模では両者は分散・学習曲線とも同等で、出力相関版 node perturbation は NNN 固有の安価な forward-gradient 推定としてむしろ有力だった。したがって node perturbation は**打ち負かす主対照ではなく、統合が余計な構造なしに機能することを示す ablation** と位置づけ直す。本構想の主張は性能でなく自然な統合（§20.14）にある。

### D. backprop なしの credit 伝播（forward-only 学習）
- 代表: Lillicrap et al. (2016) feedback alignment、Nøkland (2016) direct feedback alignment、Kolen & Pollack (1994) の weight 追跡、Hinton (2022) forward-forward。
- 差分: weight mirror は **固定ランダム feedback（FA）でも別学習の feedback（KP）でもなく、forward 共分散から推定された feedback 重み**。FA の「ランダムでも学習は進むが精度に上限」という限界に対し、NNN の mirror は真の $W$ の推定なので原理的に精度上限がない。KP 追跡は `credit.py` に既に実装フックがあり、online 化（§17.2-4）の自然な道具。

### E. 探索のためのノイズ注入（deep RL）
- 代表: Fortunato et al. (2017) NoisyNet、parameter-space noise for exploration。
- 差分: **直接の対比相手**。NoisyNet は探索のために重みへ学習可能なパラメトリックノイズを足すだけで、ノイズは探索専用。NNN のノイズは per-unit recruitment であり、(i) SR 最適（内点最適強度）を持ち、(ii) 探索と credit 推定を兼ね、(iii) 空間場として部分網を動員する。「ノイズを足す」以上の役割を持たせている点が差。

### F. 神経修飾と RL のメタ制御
- 代表: Doya (2002) metalearning and neuromodulation、Yu & Dayan の uncertainty と ACh/NE。
- 差分: §7–9 の noise field は、Doya の「神経修飾物質が学習・探索のメタパラメータを制御する」という枠組みの **具体的な計算実体** とみなせる。field 座標が探索と動員を制御する低速内部状態である、という構成はこの系譜に真っ直ぐ乗る。「ノイズ場＝神経修飾場」（`docs/idea_neuromod.md` §2–§5）と同じ位置づけを RL 側から補強できる。

### G. options / 階層強化学習
- 代表: Sutton, Precup & Singh (1999) options、Bacon, Harb & Precup (2017) option-critic、Vezhnevets et al. (2017) FeUdal、Eysenbach et al. (2018) DIAYN。
- 差分: 通常の option は **別パラメータの sub-policy $\pi_\omega$** を持つ。NNN-RL は option を、**共有重みを連続 noise field で addressing した点** として表現する（§7.3 の連続 field 座標）。したがって option 間は field 空間で連続に補間でき、未学習の中間 option が滑らかに現れる（§14.2 の主張）。これは離散 option の option-critic とは異なる「連続 option 埋め込み」であり、§17.2-5 で示した通り、field への credit を option-critic の intra-option / termination gradient として定式化すれば、この文献群と直接比較できる。

### H. context-dependent gating（継続学習）
- 代表: Masse et al. (2018 PNAS) context-dependent gating。
- 差分: Masse らはランダム割当のゲート＋別途のシナプス安定化という二機構だが、NNN は単一物理量 $(\sigma,h)$ が選択と計算の両方を担う。RL 文脈では「同一物理量が option 選択と方策計算を兼ねる」として同じ対比が使える。

### 研究プログラム内での位置づけ（相互引用の設計）

この RL 構想は、既存3系統（学習則・神経修飾場・資源制御）の **第4の断面であると同時に、
それらを統合する capstone** に当たる。

- `draft_nce.md`（forward 統計から credit を作る学習則）を **そのまま actor/critic の空間方向 credit に使う**（§3・§17.1-1）。
- **ノイズ場による addressing**（noise field が共有重み上の複数方策を addressing する、`docs/idea_neuromod.md`）を **option 機構として使う**（§7・§17.3）。
- **動員ダイヤルによる資源制御**（動員ダイヤル $\rho_k$ でノイズ場を資源として制御する、`docs/idea_consolidation.md`）を **動員（recruitment）の制御則として使う**（§7.1 の※・§23.7）。〔※ 修正。$\rho_k$ は $\sigma_k=\rho_k\sigma_0,\ h_k=h_0/\rho_k$ を与える**スカラーの制御ダイヤル**であって $(\sigma,h)$ の対ではない（`docs/idea_core.md` §1.2）。また §11 の noise budget は $h$ を動かさない $\sigma$ 単独の配分であり、$(\log\sigma,\log h)$ 平面で $(+1,-1)$ 方向へ動く $\rho$ とは別物である（同 §4.6）。両者を同一視していた当初の対応づけは成り立たない。〕
- そして **報酬 $\Delta_t^R$ が、この3つに共通する変調子** になる。学習則の credit も、場による動員も、資源配分も、すべて同一の reward-modulated eligibility 原理で更新される（§10.3・§15）。

したがって論文戦略上は、RL を単独で立てるより「NCE の学習則と神経修飾場側の addressing が報酬のもとで出会う地点」として、既存3系統へのポインタ付きで位置づけるのが最も強い。逆に言えば、RL 論文が成立するには §17.2 の未定義部（特に 5: field への credit 経路）を埋めることが前提になる。

### 参考文献（要 venue 再確認）

以下は代表文献。著者・年は確度が高いが、正確な巻号・掲載誌は投稿前に各自で確認すること。

- Williams (1992) "Simple statistical gradient-following algorithms" (REINFORCE), Machine Learning.
- Fiete & Seung (2006) node perturbation, Phys. Rev. Lett.；Werfel, Xie & Seung (2005), NIPS/Neural Computation.
- Sutton, Precup & Singh (1999) options, Artificial Intelligence；Bacon, Harb & Precup (2017) option-critic, AAAI.
- Frémaux & Gerstner (2016) reward-modulated three-factor rules, Front. Neural Circuits；Gerstner et al. (2018) eligibility traces, Front. Neural Circuits.
- Lillicrap et al. (2016) feedback alignment, Nat. Commun.；Nøkland (2016) DFA, NIPS；Kolen & Pollack (1994)；Hinton (2022) forward-forward.
- Fortunato et al. (2017) NoisyNet, ICLR 2018.
- Doya (2002) metalearning and neuromodulation, Neural Networks.
- Eysenbach et al. (2018) DIAYN；Vezhnevets et al. (2017) FeUdal Networks, ICML.
- Masse et al. (2018) context-dependent gating, PNAS.

---

## 19. Field 生成パラメータへの credit 経路（§17.2-5 の具体化・2026-07-20 追記）

§17.2-5 で「$\Delta_t^R$ が modulatory core $f_{\mathrm{mod}}$ の重み $\theta_{\mathrm{mod}}$ へ届く経路が未定義」と指摘した最大の穴を、ここで埋める。結論は、**field を連続潜在行動とみなす二層方策として定式化し、行動級と同じ forward-covariance credit を field 級にも走らせる** ことである。これにより「同一のノイズ機構」の主張が階層方向へも一貫する。

### 19.1 まず二つの regime を分離する

現在の草稿は §10 と §9 で暗黙に別の前提を置いており、これを分けると穴の所在が明確になる。

- **Regime A（free field、§7.2 の離散 prototype）**: 各 prototype のノイズ強度 $\sigma_o$（または per-unit $\sigma$）を **直接学習するパラメータ** とする。この場合、§10 の $\Delta\sigma=\alpha_\sigma\Delta_t^R e_{\sigma,t}$ がそのまま適用でき、$\theta_{\mathrm{mod}}$ は存在しないので credit 経路の問題も生じない。
- **Regime B（generated field、§7.3–9 の連続 $c$）**: $\sigma_{t,k}=\sigma_{\max}G(u_k;c_t,\tau)$ であり、$\sigma$ は $c_t$ から生成される **従属変数** で、学習対象は $\theta_{\mathrm{mod}}$（および $\tau$）である。credit 経路が問題になるのはこの regime だけ。

したがって §10 の $\psi_\sigma$ は、regime B では「直接学習する量の eligibility」ではなく、「$c\to\sigma$ の chain における中間量」として読み直す（§19.5）。この読み替えを明記すれば、§9 と §10 の見かけ上の不整合が解消する。

### 19.2 Field を連続潜在行動とする二層方策

§9.2 の因果順序（時刻 $t-1$ で生成した field を $t$ に適用、$c_{t+1}=f_{\mathrm{mod}}(s_t,c_t)$）により、これは同一 step 内の循環を持たない **階層 MDP** になる。すなわち field 選択 $c_t$ は時刻 $t-1$ の決定で、その帰結は時刻 $t$ 以降のすべての報酬である。

$$
\text{field 級（低速）:}\quad
c_t\sim\pi_{\mathrm{mod}}(\cdot\mid s_{t-1},c_{t-1})
=\mathcal N\!\left(\bar c_t,\ \Xi\right),
\qquad
\bar c_t=(1-\kappa)c_{t-1}+\kappa\,\tilde c_{t-1}
$$

$$
\text{行動級（高速）:}\quad
a_t\sim\pi(\cdot\mid s_t;W,P(c_t))
$$

ここで $\tilde c_{t-1}=f_{\mathrm{mod}}(s_{t-1},c_{t-1})$、$\Xi$ は §8 の field 探索揺らぎ $\xi_t$ の共分散である。joint likelihood が $\pi_{\mathrm{mod}}\cdot\pi$ と factorize するため、policy gradient は **二つの score の和** に分解する。行動級 score は §3–4 で既に構成した。以下は field 級 score の構成である。

### 19.3 Field 級 score も forward 統計から作る

行動級の出力 score $u_t=\Sigma_t^{-1}(a_t-\mu_t)$（§3.1）と **完全に同型** に、field 級の出力 score は

$$
u_t^{\mathrm{mod}}
=\Xi^{-1}\left(c_t-\bar c_t\right)
=\Xi^{-1}\xi_t
$$

となる。field 生成パラメータへの score は

$$
\nabla_{\theta_{\mathrm{mod}}}\log\pi_{\mathrm{mod}}
=\kappa\left(
\frac{\partial \tilde c_{t-1}}{\partial\theta_{\mathrm{mod}}}
\right)^{\!\mathsf T}
u_t^{\mathrm{mod}}
$$

である。ここで $\partial\tilde c_{t-1}/\partial\theta_{\mathrm{mod}}$ は、modulatory core を **tonic noise を持つ NNN 部分網**（§9.1）として構成すれば、行動級とまったく同じ weight-mirror 再帰（`cov_weight` + crossing slope）で forward 推定できる。つまり **同一の forward-covariance credit 機構が、二つの時間スケールで二回走る**。行動級では $u_t$、field 級では $u_t^{\mathrm{mod}}$ が、それぞれ top-level $\delta$ になる。これが §15 の「同一ノイズ機構への統合」を階層方向へ拡張する具体形である。

### 19.4 Field 級 eligibility と報酬変調

field は多数 step にまたがって作用するため、行動級より長い持続の trace を持たせる。

$$
\psi_t^{\mathrm{mod}}
=\kappa\left(
\frac{\partial \tilde c_{t-1}}{\partial\theta_{\mathrm{mod}}}
\right)^{\!\mathsf T}\Xi^{-1}\xi_t,
\qquad
e_t^{\mathrm{mod}}
=\gamma\lambda_{\mathrm{mod}}\,e_{t-1}^{\mathrm{mod}}
+\psi_t^{\mathrm{mod}}
$$

$$
\Delta\theta_{\mathrm{mod}}
=\alpha_{\mathrm{mod}}\,A_t^{\mathrm{mod}}\,e_t^{\mathrm{mod}}
$$

ここで $\lambda_{\mathrm{mod}}>\lambda_W$（option 継続長に合わせた長い持続）、$\alpha_{\mathrm{mod}}\ll\alpha_W$（§12 の二時間スケールと整合）とする。

$A_t^{\mathrm{mod}}$ は field 選択の advantage である。分散低減のためには baseline を field でも条件づけるべきで、§6 の共有 critic を **field 拡張 value $V(s_t,c_t)$** に拡張し（value readout の入力に $c_t$ を加える）、

$$
A_t^{\mathrm{mod}}=r_t+\gamma V(s_{t+1},c_{t+1})-V(s_t,c_t)
$$

とするのが自然。単純化するなら行動級と同じ $\Delta_t^R$ を流用してもよい（不偏だが高分散）。

### 19.5 Recruitment 経路の pathwise 補正（任意・低分散化）

$c_t$ は当該 step の $a_t$ の分布そのものも形づくる（$c_t\to\sigma_t\to$ 動員部分網 $\to a_t$）。realized $a_t$ に対する、この経路の pathwise score は

$$
\frac{\partial \log\pi(a_t\mid s_t,c_t)}{\partial c_t}
=\left(
\frac{\partial \sigma_t}{\partial c_t}
\right)^{\!\mathsf T}\psi_{\sigma,t}
$$

である。$\partial\sigma_t/\partial c_t$ は Gaussian-bump 写像の **解析 Jacobian**（`noise_field.gaussian_fill` の分離型ガウス積から閉形式で得られる、$\partial\sigma_k/\partial c_d=\sigma_{t,k}\,(u_{k,d}-c_d)/\tau_d^2$）で、$\psi_{\sigma,t}$ は §10 の noise-field eligibility である。

これは §19.4 の純 score-function 推定に対する **低分散な代替経路** である。二重計上を避けるため、実装は次のどちらかに統一する。

1. **純 REINFORCE**: field 効果はすべて $\xi_t$ と多段 return の相関（§19.4）で捉える。単純だが高分散。
2. **ハイブリッド（reparameterization + score-function）**: 当該 step の行動形成への **即時効果を pathwise 項で解析的に置換** し、downstream の帰結だけを score-function 項で捉える。

この選択は、field を「score-function で探索する確率行動」とみなすか、「pathwise で微分できる連続制御量」とみなすかの違いであり、NNN では両方が forward 統計から作れる点が利点である。

### 19.6 persistence $\kappa$ を soft termination とみなす

option-critic の termination 関数 $\beta$ に対応するのが、§8 の変化速度 $\kappa$ である。$\kappa$ を学習するなら、$\bar c_t$ の $\kappa$ 依存性から独自の eligibility を与えられる。ただし最初は **$\kappa$ 固定** が安全で、§8 の主張（persistence dynamics 自体が切替を定義し、外付け終了判定器を避ける）とも整合する。

### 19.7 §13 統合アルゴリズムへの追加ステップ

§13 の 14 step に、field 級 credit を次のように挿入する。

- **5.5**: field 級 score $u_t^{\mathrm{mod}}=\Xi^{-1}\xi_t$ を計算する。
- **8.5**: modulatory core の weight mirror で $\partial\tilde c/\partial\theta_{\mathrm{mod}}$ を推定し、$\psi_t^{\mathrm{mod}}$ を生成して $e_t^{\mathrm{mod}}$ に蓄積する。
- **13.5**: $\alpha_{\mathrm{mod}}A_t^{\mathrm{mod}}e_t^{\mathrm{mod}}$ により $\theta_{\mathrm{mod}}$ を低速に更新する。

### 19.8 option-critic との対応と差分（§18-G の精密化）

- **field 級 policy gradient** は、option-critic の intra-option policy gradient を **連続 option 埋め込み** へ一般化したものに当たる。離散 option 集合ではなく field 座標 $c$ 上の連続分布を学習する。
- **$\kappa$（persistence）** は termination 関数の soft 版。
- **決定的な差**: option-critic は両級を backprop で学習するが、NNN-RL は **両級とも forward-covariance 推定** で学習し、かつ option を別パラメータの sub-policy でなく **共有重み上の noise-field addressing** として実現する（§18-G、`docs/idea_neuromod.md` §1.1 の L1）。

これで「forward fluctuation が行動と field の双方の credit を、同一原理で二時間スケールに配る」という構図が閉じ、§17.2 の最大の未定義部が解消する。残る実装上の要点は、field 拡張 critic $V(s,c)$ の導入（§6 の拡張）と、modulatory core を weight mirror が推定可能な NNN 部分網として構成することの二点である。

---

## 20. 第1段階の実験プロトコル（実装仕様・2026-07-20 追記）

このセクションは、次の実装作業がそのまま着手できる粒度の仕様である。対象は §14.1 の第1段階、すなわち **CartPole 規模で covariance eligibility ＋ TD 変調が online で学習し、node perturbation に対し credit 分散で勝つか** の検証。§17 の critical-path 評価（online mirror の質と node-perturbation 優位が荷重を支える二点）を、そのまま go/no-go に落とす。

**設計不変条件（最重要）**: この実験は、外部 RL アルゴリズム（PPO・SAC 等）の policy network として NNN を差し込むもの **ではない**（§1）。credit・eligibility・探索・行動 sample のすべてを **NNN の forward fluctuation path の内部で生成する** ことが本質であり、そこが崩れると検証の意味が失われる。したがって `examples/nnn_sb3_ppo_intrinsic_demo.py`（SB3/PPO で NNN を方策に使う既存デモ）は本プロトコルの参照対象ではない。backprop actor–critic は**上界の対照**として並べるだけで、手法そのものは常に forward 推定に閉じる。

### 20.0 検証する主張と go/no-go

- **C1（mirror の online 成立）**: 各 env step で state 1 個（$N=1$）× $T$ 内部 sample しかない条件で、forward-covariance credit（`cov_jac` 再帰）が autograd の $\nabla_W\log\pi$ を十分な精度で再現する。
- **C2（学習と分散優位）**: 完成した学習則が CartPole-v1 を学習し、node perturbation より credit 分散が低く、その差がネットワーク幅 $H$ とともに拡大する。

判定ゲート:
- **G1（先行ゲート）**: C1 の per-step cosine が閾値を超える（暫定値は §20.6、FNCL 回帰 PoC の cosine を基準に較正）。超えなければ幅・$T$・mirror EMA を調整。それでも駄目なら「**online mirror は制御に不十分**」という否定的結論そのものが第1段階の知見であり、field/option の半分（§7–19）へ進む前に構想の前提を見直す材料になる。
- **G2**: C2 の学習到達と分散優位を満たす。

### 20.1 再利用する資産（重要 ―― 新規実装を最小化する）

`data_nce/fncl/` に credit エンジンが既にあり、**stage 1 は「教師あり出力誤差を policy score に差し替え、時間 trace ＋ TD を足し、online 化する」ことに帰着する**。流用する具体物:

- `fncl.network.Capture`: forward フックで per-sample の $d^{(l)}, z^{(l)}$（[N,T,H]）と readout の per-sample 出力 $y_{\text{samples}}$（[N,T,1]）を記録。
- `fncl.network.kde_slope(crossing_layer, d)`: 転置重み不使用の分布フリー局所傾き $dz/dd$（crossing 自身の backward = antithetic 有限差分）。`phi_prime` は解析版。
- `fncl.train.cov_weight(d_next, z_prev, pool)`: weight mirror $\widehat W=\mathrm{Cov}(d_{\text{next}},z)/\mathrm{Var}(z)$。
- `fncl.train.train_cov` の `cov_jac`/`cov_jac_full` の**再帰そのもの**: $\delta^{(l)}=(dz/dd)^{(l)}\odot(\widehat W^{(l+1)\mathsf T}\delta^{(l+1)})$、EMA weight mirror、および Kolen–Pollack PREDICT（既知の重み減分だけ mirror をずらす）。これは §17.2-4 で「online 化の道具」と述べたものの実体。
- `fncl.train.ManualOpt`: 手動勾配の in-place SGD/Adam、適用した減分を返す（KP 追跡用）。
- `fncl.perturb.Perturber`, `gate_masks`, `rng_snapshot/restore`: node perturbation baseline（CRN/antithetic 摂動）。

### 20.2 環境と方策

- **環境**: `gymnasium` CartPole-v1（obs 4 次元、行動 2）。依存は `gymnasium[classic-control]`（リポジトリで既に利用可能）。
- **方策**: §5 の Bernoulli。共有 NNN body `structure=[4, 64, 64]`（`SimpleNNNBase` を土台に、`Capture` で中間を記録する forward）、hidden 最終層から2つの線形 readout ―― action logit（$\to 1$）と value（$\to 1$）。
- **per-step の流れ**（$N=1$, $T$ sample）:
  1. hidden 最終 $z^{(m)}\in[1,T,H]$。logit readout の per-sample 値 $o^{(m)}=W_o z^{(m)}\in[1,T,1]$。
  2. 発火確率 $p=\sigma\!\big(\tfrac{1}{T}\sum_m o^{(m)}\big)$、行動 $a\sim\mathrm{Bernoulli}(p)$（実環境へ送るのはこの1個）。
  3. **出力 credit（top-level $\delta$）は logit 上の誤差 $(a-p)$**。これは §5 の $u=(a-p)/(p(1-p))$ に $\partial p/\partial o=p(1-p)$ を掛けたもので、logit 上では $p(1-p)$ が相殺して $(a-p)$ になる。連続行動へ拡張する場合のみ §3.1 の $\Sigma^{-1}(a-\mu)$ に差し替える。

### 20.3 空間 credit の3方式 + gold reference

すべて同じ top-level $(a-p)$ から出発し、**それを hidden へどう配るか** だけを変える統制比較にする。

- **ours（`cov_jac`）**: $\widehat W_o=\texttt{cov\_weight}(o, z^{(L-1)})$; $g^{(L-1)}=\widehat W_o^{\mathsf T}(a-p)$; $\delta^{(l)}=g^{(l)}\odot\texttt{kde\_slope}^{(l)}$; $g^{(l-1)}=\widehat W^{(l)\mathsf T}\delta^{(l)}$; $\psi_W^{(l)}=\delta^{(l)}\otimes z^{(l-1)}$（T 平均）。real $W^\mathsf T$ を一切読まない。
- **true-transpose（oracle）**: 同じ再帰で $\widehat W$ を real weight に置換。autograd と一致し、mirror 誤差ゼロの上界。
- **node perturbation（baseline）**: `Perturber` で各 unit に摂動、`covariance_credit(z^{(l)}, L, "pooled")` で unit credit を return 相関から直接推定、$\psi_W=g\otimes z_{\text{prev}}$。mirror も再帰もなし（§18-C の最近傍対照）。
- **gold $\nabla_W\log\pi$**: autograd で $\log\pi(a\mid s)=a\log p+(1-a)\log(1-p)$ を $W$ 微分（crossing の KDE backward ＋ real $W^\mathsf T$）。M1 の基準。

### 20.4 mirror の online 維持（C1 の要）

各 step の `cov_weight` 推定を EMA（$\beta\approx0.99$）で平滑化し、加えて `ManualOpt.update` が返す既知の重み減分を mirror にも適用する（`train_cov` の KP PREDICT を online ループへ移植）。$N=1$ の高分散を時間方向で均すのが狙い。EMA/KP の有無は M1 でアブレーションする。

### 20.5 時間 trace と TD 更新

$$
e_{W,t}^{(l)}=\gamma\lambda_W\,e_{W,t-1}^{(l)}+\psi_{W,t}^{(l)},
\qquad
\Delta_t^R=r_t+\gamma V(s_{t+1})-V(s_t),
\qquad
\Delta W^{(l)}=\alpha_W\,\Delta_t^R\,e_{W,t}^{(l)}
$$

更新は `ManualOpt`。**critic は最初は minimal**（hidden mean の線形 readout を TD 二乗誤差で更新、隠れ credit は backprop か素通し）にして、actor-credit の比較を critic 品質で交絡させない。§17.2-6 の full 版（value も `cov_jac`）は G2 通過後の追試に回す。

初期ハイパー: $\gamma=0.99$, $\lambda_W\in\{0,0.5,0.9\}$, $T\in\{16,32,64,128\}$, $h=0.15$, $\text{std}=0.6$（`SimpleNNNBase` 既定近傍）, $\alpha_W$ は log スケール探索。

**※ 体制の切り替えに注意（`docs/idea_core.md` §4.7）。** §11・§17.2-7 は交差半幅を連動させる体制 (a)（$h_k=c_h\sigma_k$、ゲージが存在し $\sigma$ は動径 1 次元のみ）を前提に書かれているが、**§20 以降の実測はここで固定した $h=0.15$ の体制 (b)**（$h$ は大域ハイパーパラメータ、ゲージ不在、$\sigma$ は本物の自由度）である。したがって §11 の「絶対 scale は重みノルムと冗長」という主張は §20 以降の実験系には及ばない。代わりに体制 (b) では、$\sigma$ 勾配推定量 $\langle\hat g\,\phi_T'\,(-d/\sigma)\rangle$ が $h$ 固定でも Euler 恒等式を恒等的に満たすという構造バイアスが付く（§10.1 の※）。以降の節（§21・§23・§25）はすべて体制 (b) と読むこと。

### 20.6 測定指標

- **M1（G1 用、学習不要）**: 毎 step、ours の $\psi_W$ 方向と gold $\nabla_W\log\pi$ の cosine similarity。幅 $H\in\{16,32,64,128,256\}$・深さ・$T$ を掃引。EMA/KP あり/なし、`pool` あり/なしも。**暫定 G1 閾値**: $H=64$ online で per-step median cosine $\gtrsim 0.6$（FNCL 回帰 PoC の cosine 実測に合わせて確定する。cosine が不完全でも学習が回るなら M2 で救済されうるので、最終判断は M2 と併せる）。
- **M2（G2 用）**:
  - (a) 学習曲線: return vs env steps、seed 8–16 の mean±std。ours / node-pert / backprop actor-critic / true-transpose / $\lambda=0$ vs $\lambda>0$。
  - (b) **分散指標（中心的数字）**: 固定 $(s,W)$ で内部ノイズを多数回 draw し、各手法の更新方向 $\hat g$ の正規化分散 $\mathrm{Var}=\mathbb E\|\hat g-\mathbb E\hat g\|^2/\|\mathbb E\hat g\|^2$ を推定。$H$ 掃引で **ours と node-pert の差が $H$ で拡大するか**（Fiete–Seung の $O(N)$ 劣化に対する層別 credit の優位、§18-C）。
  - (c) reward delay 頑健性: 報酬を $k$ step 遅延させ、$\lambda_W$ 依存性を見る。

### 20.7 比較条件（§14.1 をスイッチに割付）

| §14.1 の条件 | 実装スイッチ |
|---|---|
| 標準 actor–critic (backprop) | `agent=backprop` |
| 真の転置重み NNN eligibility | `agent=true_transpose` |
| covariance weight mirror NNN eligibility | `agent=cov_jac`（ours） |
| $\lambda=0$ の瞬間 eligibility | `lambda_w=0` |
| $\lambda>0$ の時間 trace | `lambda_w>0` |
| pooled reward covariance | `agent=node_pert` |
| policy-score eligibility | ours の top-level を $(a-p)$ にする既定 |

### 20.8 SR sweep（dual-use tension の検証、§17.2-3）

ノイズ強度 $\sigma$ を掃引し、M1 の cosine（mirror 推定精度）と M2 の到達 return を**同一図に重ねる**。両者の最適 $\sigma$ 領域が重なるかで、探索ノイズと推定ノイズの二律背反の有無を直接見る。`--model sample`（機構）と `--model analytic`（平均場）の対照は、SR が機構（sample）水準にのみ現れるという対照（`docs/idea_neuromod.md` §6）の RL 版。これは第1段階の最も概念的に重要な補助実験。

### 20.9 実装配置

**すべて `tmp/` 配下に置く**（共通部分も含む）。共通部分（package 化した再利用モジュール）は plain な名前で `tmp/rl/` にまとめ、それ以外の各検証コード（runner・実験スクリプト）は `tmp/` 直下に `rl_*` の命名で置く。credit エンジンは `data_nce/fncl/` を import して薄く包む（fncl 側は変更しない）。

共通部分 `tmp/rl/`:

- `env.py`（CartPole ラッパ、reward delay オプション）
- `policy.py`（`Capture` 流用の記録 forward ＋ logit/value readout、Bernoulli sampling、$(a-p)$ 生成）
- `credit.py`（fncl の `cov_weight`/`kde_slope`/`cov_jac` 再帰を online 用に包む ＋ node-pert ＋ true-transpose ＋ autograd gold）
- `mirror.py`（EMA ＋ KP PREDICT の online mirror 状態）
- `trace.py`（eligibility trace）
- `agents.py`（`cov_jac`/`node_pert`/`backprop`/`true_transpose` の共通インターフェース）
- `metrics.py`（cosine, normalized variance）
- `train.py`（online ループ）, `viz.py`, `constants.py`

検証コード（`tmp/` 直下、`rl_*` 命名）:

- `rl_stepA_cosine.py`（Step A: mirror 品質 M1、学習なし、`--H --T --sweep {width,none}`）
- `rl_cartpole_train.py`（Step B: 完全ループ M2、`--agent --lambda_w --T --H --seed`）
- `rl_variance.py`（Step B: 分散指標 M2-(b)、$H$ 掃引）
- `rl_sr_sweep.py`（Step C: SR sweep、`--model {sample,analytic}`）

依存追加なし（`gymnasium[classic-control]`, torch 2.6, numpy は既存）。`tmp/rl/` を import できるよう、検証スクリプトは `tmp/` を sys.path に加えるか `tmp` からの相対 import で解決する（fncl の `network.py` が PROJECT_ROOT を sys.path に足しているのと同じ方式）。

### 20.10 段階と受入（次作業の分割）

- **Step A（mirror 品質のみ、学習なし）**: `policy.py` ＋ gold ＋ M1。ランダム重み・軽い事前学習の両方で cosine を測り G1 判定。ここが本当の関門。
- **Step B（完全ループ）**: trace ＋ TD ＋ node-pert baseline ＋ M2。学習到達と分散優位で G2 判定。
- **Step C**: SR sweep（§20.8）。

各 step の受入は「cosine 閾値」「CartPole return（例: 195/200 を安定到達）」「ours/node-pert の分散比 < 1 かつ $H$ 単調」で定義する。

### 20.11 per-step 疑似コード（Step B の中核）

```
s = env.reset()
init eligibility e_W[l]=0, mirror EMA \hat W[l], \hat W_o
for t in steps:
    d[l], z[l], o_samples = policy.forward_capture(s)      # Capture, N=1,T sample
    p = sigmoid(o_samples.mean(over T)); a ~ Bernoulli(p)
    V_s = value_head(z[-1].mean(over T))
    # --- spatial credit from the policy score (no transposed W) ---
    \hat W_o = ema(\hat W_o, cov_weight(o_samples, z[-1]))
    g = \hat W_o^T (a - p)
    for l from L-1 downto 1:
        delta[l] = g * kde_slope(crossing[l], d[l]).mean(over T)
        psi_W[l] = outer(delta[l], z[l-1].mean(over T))
        \hat W[l] = ema(\hat W[l], cov_weight(d[l], z[l-1]))
        g = \hat W[l]^T delta[l]
    # --- temporal trace ---
    e_W[l] = gamma*lambda_w*e_W[l] + psi_W[l]
    # --- step env, TD, modulate ---
    s2, r, done = env.step(a); V_s2 = value_head(...)
    dR = r + gamma*V_s2*(not done) - V_s
    for l: dW = ManualOpt.step(W[l], -dR*e_W[l]); \hat W[l] -= dW   # KP PREDICT
    value_head.td_update(dR)
    s = s2; if done: s = env.reset(); reset e_W
```

この仕様で、Step A の G1 判定（online mirror が成立するか）が最初のマイルストーンになる。ここを通れば学習則の半分が実証され、通らなければ「online mirror の質」という §17 で最重要と評価したリスクが顕在化したことになり、いずれにせよ構想にとって決定的な情報が得られる。

### 20.12 Step A 実測結果（2026-07-20、G1 = PASS）

実装 `tmp/rl/`（共通）と `tmp/rl_stepA_cosine.py`（runner）。CartPole ランダム rollout の 128 状態を **online（N=1、1状態ずつ、EMA なしの single-shot mirror）** で評価。各値は per-step cosine の median。

| H \\ T | 16 | 64 | 256 |
|---|---|---|---|
| **16** | 0.888 | 0.975 | 0.994 |
| **64** | 0.809 | 0.965 | 0.993 |
| **256** | 0.597 | 0.924 | 0.985 |

（`covjac~gold`。対照: `true_transpose~gold` は T=16/64/256 で 0.92/0.98/0.995 と幅 H に**不依存**で、残差は mirror でなく KDE slope 推定の T 依存分。すなわち再帰の実装は正しい。）

読み取れること:

1. **G1 は明確に通過**。既定 $T=64$ で cov_jac cosine は H=16–256 にわたり 0.92–0.98 で、暫定閾値 0.6 を大きく上回る。最悪の corner（$T=16, H=256$）だけが 0.60 に落ちる。
2. **online single-shot mirror の質は $T$ で素直に改善し、幅 $H$ を上げると要求 $T$ が増える**（§17.2-4 で予告した online mirror 分散の定量化）。$H\le64$ は $T=64$ で十分、$H=256$ は $T\ge64$ を要する。$T$ は環境と無関係な内部 sample 数なので安価に増やせ、さらに Step B の EMA/KP でこの要求は緩む（今回は EMA なしの下界）。
3. **median cosine では cov_jac と node_pert は分離しない**（`node~gold` は cov_jac とほぼ同値、僅かに下）。予告どおり、node perturbation に対する優位は median 方向一致ではなく **分散**にあり、Step B の M2-(b)（固定 $(s,W)$ で内部ノイズを多数 draw した更新方向の正規化分散の $H$ スケーリング）で判定する。

結論: 学習則の半分の前提（online で forward mirror が policy gradient 方向を復元する）は CartPole 規模で成立。次は Step B（完全ループ + node-pert baseline + 学習曲線 + 分散スケーリング）で G2 を判定する。

### 20.13 Step B 実測結果（2026-07-20）

実装: `tmp/rl/{mirror,agents,train,policy,credit}.py`（mirror EMA+KP、eligibility trace、TD、二時間スケール）+ 検証 `tmp/rl_cartpole_train.py`（学習曲線）、`tmp/rl_variance.py`（分散）、`tmp/rl_cartpole_demo.py`（デモ）。credit・探索・行動 sample はすべて NNN forward path 内部で生成し、cov_jac は転置重み backward を一切使わない。

**(1) 学習は成立し CartPole を解く（G2 の学習側 = PASS）。** cov_jac（SGD, $\alpha_a=0.02, \alpha_c=0.05, \lambda=0.9, T=64, H=64$、観測正規化、last-hidden 線形 TD critic）は online で学習し、**greedy 評価で return 500（CartPole-v1 満点）に到達**。デモ `tmp/out/rl_cartpole_demo.gif` は同一 run のチェックポイントで failure→success を示す（step 0: return 11、5000: 117、7500: 461、12500: 500）。外部 RL アルゴリズムも backprop も使わず、forward fluctuation の credit だけで制御が成立することの最初の実証。

実装上の要点（今後の再現のため）:
- **Adam は eligibility trace と相性が悪く不安定**（振動・崩壊）。**SGD が安定**。
- **観測正規化が必須**（CartPole の生特徴はスケール差が大きく、固定ノイズ交差を飽和させる）。
- critic は minimal な線形 TD（last-hidden ensemble mean を特徴）で actor-credit の比較を交絡させない。critic の TD($\lambda$) は高 LR で崩壊したため既定は TD(0)。
- online mirror は EMA($\beta=0.99$)+Kolen–Pollack PREDICT。

**(2) node perturbation に対する分散優位（§18-C）は、この設定では示せなかった（重要な否定的結果）。** $H=64$、深さ 2–8 で cov_jac と node_pert の正規化分散比 node/cov は **0.71–0.99**、すなわち cov_jac は分散で勝たない。むしろ深さが増すと mirror 推定誤差が再帰で乗算的に累積し、flat な出力相関の方がやや低分散になる。原因は baseline の取り方にある:

- ここで実装した node_pert は「各ユニットを**出力 logit** $o^{(m)}$（T sample で変動）に直接相関」させる版で、これは出力の感度情報を持つ**強い** baseline。cov_jac の最終隠れ層 credit（mirror `cov_weight(o,z)`）とは実質同一の回帰で、上流層だけが再帰 vs 直接相関で異なる。浅い/中程度の深さでは差が出ず、深いと再帰の誤差累積で cov_jac が不利。
- §18-C が本来想定していた $O(N)$ 劣化を持つのは「**報酬**相関」版 node perturbation（credit$_i=\Delta^R\xi_i$、出力感度を使わない）であり、これは別の**弱い** baseline。その比較は per-step の score 分散ではなく、full-loop の sample 効率で示すべき。

ここで node_pert は「同じ trace/TD 骨格に credit 源だけ差し替えた」**ablation** であって、打ち負かす対象ではない（§18-C の競争的な読みは撤回する。後述の主張較正を参照）。

**(3) 学習曲線比較（cov_jac / node_pert / backprop、各 2 seed、40k step、`tmp/out/rl_m2a_curves.png`）**: 三者とも return ~20→60–80 へ上昇し、**互いに重なって区別できない**（training 中の探索込み return。greedy 評価では (1) の通り 500 到達）。すなわち forward mirror credit は backprop に劣らず学習でき、同時に flat な node_pert とも同等で、CartPole/深さ 2 の規模では手法間に有意差が出ない。(2) の分散結果と整合する。

### 20.14 主張の較正（堅い版・2026-07-20）

性能優位は本構想の主眼ではない（ヒューリスティックを多く含む既存手法に性能で勝つことは目的にしない）。証拠が支える主張は次に限定する。

> **NNN の forward fluctuation だけから、外部の方策分布・探索スケジュール・転置重み backward を一切導入せずに、行動 sample・探索・局所感度・層間 credit・eligibility を単一の機構として構成でき、報酬による trace 変調と合わせて CartPole を学習できる（greedy return 500）。この credit は autograd の $\nabla_W\log\pi$ を cosine ~0.95 で復元し、揃えた online actor-critic では backprop と区別できない。**

言い過ぎになる表現とその理由:

- **「backprop 同等（equivalent）」は言えない**。seed 2・単一タスク・浅い（深さ2）・素朴 backprop 対照で、示せたのは「区別できない／劣らない」まで。統計的同等性（信頼区間の重なり）は測っていない。
- **「node perturbation に対する優位」は言えない**。分散でも学習曲線でも同等で、flat baseline も同じバーをクリアする。ゆえに「転置重みが要らない」ことは cov_jac 固有の手柄にできない。

要するに本研究の芯は **性能ではなく自然な統合**（§1・§15）である。すなわち、RL の構成要素（方策・探索・感度・credit・eligibility）が NNN の同一ノイズ機構から立ち上がること自体が主張であり、CartPole 学習はその十分性の最小実証と位置づける。node_pert・backprop は優劣を競う相手ではなく、この統合が余計な構造なしに機能することを示す ablation として並べる。

### 20.15 次の方向（「NNN を自然に RL へ繋ぐ」視点で）

性能競争（§18-C の報酬相関版比較や深さ・幅スケーリング）は主眼から外し、**自然統合をより純化・検証する**方向を優先する。

1. **Step C: SR sweep = 自然統合の核の直接検証（最優先、§20.8・§17.2-3）**。単一のノイズ強度 $\sigma$ が、(a) crossing の計算成立と mirror 推定精度、(b) 探索、(c) 到達 return を**同時に**最適化する領域があるかを見る。重なれば「NNN が計算に必要とするノイズがそのまま探索と学習を担う」という自然統合の主張が最も強く立つ。衝突すれば、役割分離を**ノイズ場の空間配分**（§7–12）に委ねるという、これも NNN 固有の自然な解へ導かれる。どちらでも本質的な知見。
2. **単一 NNN での actor+critic+credit の forward-native 統一（§6・§17.2-6）**。現状の外付け線形 TD critic を廃し、value 誤差も同じ weight mirror 再帰の top-level $\delta$ として通す。「一つのノイズが方策・価値・探索・credit を全部担う」を実装レベルで閉じ、外部 scaffolding を減らすほど自然統合の主張が締まる。**〔重要な中継地点マイルストーン。課題として登録済み（Step C の後に着手）。忘れないこと。〕**
3. **探索と option を同一ノイズが担うノイズ場（§7–19、最も NNN 固有の frontier）**。計算に必要なノイズ＝探索ノイズ＝サブネットワーク動員ノイズ、という三位一体を行動モード形成（Foraging/Avoidance/Sheltering）で示す。ノイズ場が共有重み上の多重化方策を addressing するという主張（`docs/idea_neuromod.md`）と相互に支え合う。
4. **（radical・任意）パラメトリック score を捨てる**。Gaussian/Bernoulli を仮定した $u=\Sigma^{-1}(a-\mu)$ すら外部由来。報酬が forward-noise covariance credit を直接変調する reward-modulated covariance eligibility（§16 の名）へ寄せれば、log-prob の外部定義なしの最も NNN-native な学習則になる。理論的妥当性（それでも policy gradient か）が問い。

### 20.16 Step C: SR sweep 実測（2026-07-20、`tmp/rl_sr_sweep.py`）

自然統合の核（§17.2-3）を、ノイズ強度 $\sigma$ の掃引で直接検証する。covariance credit は sample 機構でしか存在しない（analytic 平均場には T sample がなく mirror を作れない）ため、この掃引は本質的に sample 機構についてのものである。

**(a)+(b) static 掃引（学習なし、`tmp/out/rl_sr_static.png`）**: 固定重みで $\sigma$ を 0.05→2.0 に振ると、**mirror cosine（計算/credit 品質）と logit spread（探索信号の大きさ）がほぼ平行に単調増加**する（cosine 0.91→0.99、spread 0.12→0.28）。すなわち計算成立と探索は **同一の $\sigma$ 依存を共有し、方向が一致**する（両者とも σ が大きいほど良くなり、対立しない）。これは自然統合の (a)–(b) 軸における肯定的な証拠。低 σ での急な credit 崩壊（SR 障壁）はこの範囲・ランダム初期化では顕在化せず（σ=0.05 でも cosine 0.91）、σ が h=0.15 を下回っても pre-activation の広がりが十分な crossing を生む。

**(c) train 掃引（各 σ を固定して学習、1 seed・30k step、`tmp/out/rl_sr_train.png`）**:

| σ | greedy return | mirror cosine（学習後重み） | action entropy |
|---|---|---|---|
| 0.10 | 55 | **0.953** | 0.65 |
| 0.30 | 22 | 0.755 | 0.35 |
| 0.60 | 426 | 0.588 | 0.41 |
| 0.90 | 150 | 0.490 | 0.35 |
| 1.30 | **500** | 0.444 | 0.56 |
| 2.00 | 72 | 0.522 | 0.46 |

読み取れること（1 seed のため非単調は割り引くが、頑健な構造は3つ）:

1. **return は内点に良好域を持つ**（σ≈0.6–1.3 で 426–500、両端 σ=0.1/2.0 は 55/72 と低い）。制御についての SR 的な内点最適が確認できる。
2. **学習後重みでの mirror cosine は σ とともに単調に低下**（0.95→0.44）。**これは static 掃引（cosine が σ とともに上昇）と逆**。random 初期化では「ノイズが多いほど crossing が増え mirror が良い」が、各 σ で学習すると重みがその σ に適応（高 σ では決定性を得るため重みが育ち pre-activation が飽和域へ）し、mirror 推定はむしろ悪化する。SR の動作点は重み regime で移動する。
3. **最良制御の σ（0.6–1.3）は最良 credit の σ（0.1）と一致しない**。すなわち計算忠実度と制御は、学習後重み軸で **対立する**。

**自然統合の核についての判定（正直な版）**:

- **強い主張「単一 σ が計算・探索・制御を同時に最大化」は成り立たない**。static では計算(a)と探索(b)は高 σ で揃うが、学習後は計算忠実度が低 σ を、制御が中〜高 σ を好み、両者は逆を向く。§17.2-3 の二律背反は「計算 vs 探索」ではなく「**計算忠実度 vs 制御**」という形で顕在化した。
- **弱い（実用的な）主張「全役割が十分に機能する共通 σ 領域が存在」は成り立つ**。σ≈0.6–1.3 は制御が優秀で、credit は劣化するが機能する。
- **最重要の副次発見: RL は低忠実度の forward credit に頑健**。σ=1.3 は cosine 0.44（≒ backprop 勾配と半分程度しか揃わない）でも return 500 に到達する。正確な勾配は不要で、「概ね正しい向き」の forward-noise credit で十分学習できる。これは対立を **soft** にし、かつ「厳密な勾配計算を要さない自然な credit で RL が回る」という点で自然統合の主張をむしろ補強する。

**含意（次への接続）**: 計算忠実度は低 σ、制御は中〜高 σ を最大化する以上、両者を同時に最大化したいなら **ノイズを一様でなく空間配分する**（一部ユニットは低 σ で綺麗な credit、他は高 σ で探索/表現）のが自然な解になる。これは §7–12 の **ノイズ場**が「二律背反の解」として要請されることを、実測から動機づける。加えて multi-seed 化で非単調（σ=0.3 の落ち込み等）を均す確認が要る。

### 20.17 Task #1: critic 統一（単一 NNN で policy+value+探索+credit を forward-native に閉じる）

実装 `tmp/rl/unified.py`（`train_unified`）+ 検証 `tmp/rl_cartpole_unified.py`。Step B の critic は detached 特徴上の**外付け線形 TD 回帰**だったが、ここでは value readout を actor と同じ共有 NNN body の上に置き、**その隠れ層 credit も actor の policy score と同一の forward weight-mirror 再帰で流す**。差は top-level 信号（actor は $(a-p)$、value は $1$）と readout（actor head mirror か value head mirror か）だけ。外部に残る scaffolding はなく、単一の forward fluctuation が action sample・探索・局所感度・**両 head の層間 credit**・eligibility を供給し、報酬が trace を変調する。実装は `credit._recurse_body`（top-level を受け取る汎用再帰）を actor/value で共有し、`MirrorState` が body・actor head・value head の3 mirror を EMA+KP で保持する。

- **機構の正しさ**: value credit は actor と同一の `_recurse_body`（Step A で cosine ~0.95 検証済み）に value head mirror（同一の `cov_weight`）と top-level 1 を与えるだけなので、構成上 actor 同様に成立する。
- **共有 body の値-credit は「妨げ」でなく「助け」**: value credit が共有 body へ届く強さ `value_body_coef` を 0→1 で振ると、**coef を下げるほど学習が悪化**（coef=0 は body が actor のみで駆動 → ほぼ学習せず eps 多数、coef=1.0 が最安定）。すなわち value 目的の勾配が共有表現を壊すのではなく、有用な特徴形成を助ける。これは「一つのノイズが方策・価値を同時に形づくる」統合像を支持する。
- **学習曲線（unified vs 外付け critic、各 2 seed・40k、`tmp/out/rl_unified_curves.png`）**: unified は学習する（return 15→~50、単一 episode 最大 226–241 step ＝ 実際に balance する方策を獲得）が、**外付け critic 版より明確に弱く平坦**（外付けは持続 ~100–150、peak 318/413；unified は持続 ~30–50、peak 226/241）。共有 body を actor と critic の両目的で同時に駆動する forward-native 構成は、外付け線形 critic より不安定・低性能で、これは深層 RL で既知の共有表現の難しさと整合する。

**判定（Task #1）**: **機構としての統合は閉じた** ―― 外部 scaffolding なしに、value も転置重みなしの同一 forward mirror で credit を受ける単一 NNN が学習する。これが実装レベルの達成物。**ただし性能は外付け critic に劣る**（正直な記録）。value credit が共有 body を助ける（coef を下げると悪化）ことは統合像を支持するが、actor と critic を単一表現に載せた forward-native AC の安定化には追加の工夫（critic warmup、勾配スケール、あるいは §7–12 のノイズ場で計算-制御の役割を空間分離）が要る。性能追求は本構想の主眼でない（§20.14）ため、ここでは統合機構の成立と正直な性能記録をもって Task #1 を完了とする。

---

## 21. ノイズ場 RL（自然統合の本丸・2026-07-20 着手）

SR sweep（§20.16）と critic 統一（§20.17）という独立した二実験が、同一の結論に収束した ―― **一様ノイズでは計算（credit 忠実度・低 σ 選好）と制御（return・中〜高 σ 選好）が対立するが、ノイズを空間的に配分すれば両立しうる**。これにより §7–12 のノイズ場は「仮定された frontier」から「二つの実測が要請する必然的な次段」へ格上げされた。

この本丸を最小構成から段階的に検証する。実装 `tmp/rl/field.py`（per-unit 場 prototype）、policy は per-unit σ 場を受ける（`policy.field`）。

### 21.1 Sub-A: 非一様な固定場は一様 σ の対立を緩めるか（前提検証）

`tmp/rl_field_prototypes.py`。固定 per-unit 場（uniform_lo/mid/hi、spatial split、graded）で CartPole を学習し、**greedy return（制御）と mirror cosine（credit 忠実度）を同時に測る**。狙いは、一様場が描く return–cosine の対立フロンティアに対し、spatial 場が「高 return かつ高 cosine」の右上へ抜けられるか。

**結果（各 2 seed・30k、`tmp/out/rl_field_subA.png`）= 否定的**:

| 場 | return | cosine |
|---|---|---|
| uniform_lo (σ=0.3) | 88 | 0.750 |
| uniform_mid (σ=0.6) | 222 | 0.604 |
| uniform_hi (σ=1.3) | 255 | 0.446 |
| **split** (0.3/1.3 半々) | 89 | 0.460 |
| **graded** (0.3→1.3 ramp) | 255 | 0.474 |

一様場は予想どおり右下がりの対立フロンティアを描く（低 σ=高 cosine/低 return、高 σ=低 cosine/高 return）。**spatial 場はこの対立を抜けない**: `split` は劣位に支配される（return も cosine も低い＝両方の悪いとこ取り）、`graded` は uniform_hi とほぼ同じ点（フロンティア上、cosine が僅かに高いだけ）。どちらも「高 return かつ高 cosine」の右上に届かない。**SR-resolution を空間配分で実現するという仮説は、この最小の固定場では支持されなかった。**

**なぜ効かないか**: 単一タスクで全ユニットが同じ readout に入り credit を受ける構成では、一部ユニットを低 σ にしても「綺麗な credit を担う計算サブ群」にはならず、単に寄与の小さいユニットになるだけ。機能的な役割分離がないため、per-unit の σ 配分は対立を分解しない。対立は「大域ノイズ量 vs 二つの大域目的」の間にあり、均質な単一タスク readout 上の空間配分では解けない。

**含意（Sub-B の前提が崩れる → 場の価値の置き直し）**: 「報酬が対立を抜ける場を選ぶ」という Sub-B の動機は成立しない（抜ける場が無い）。ただしこれはノイズ場方向を否定しない。ノイズ場本来の価値は「**共有重み上で異なる場が異なる行動を実体化する**」（§7.2 / §14.2 / 神経修飾場側の L1）であって、SR 対立の解消ではない。そしてそれを示すには **複数の行動が必要な環境**が要る ―― CartPole は単一行動なので原理的に場の価値を示せない。したがって場/option の検証は CartPole を離れ、最小の multi-mode 環境へ移すのが筋。

### 21.2 Sub-B: 報酬による prototype 選択（§7.2 の option 機構、Task #2）

**Sub-A の否定を受けて再設計**: 「報酬が SR 対立を抜ける場を選ぶ」という当初の動機は崩れた（CartPole に抜ける場が無い）。ノイズ場本来の問い ―― **共有重みのまま場を切り替えると行動様式が変わるか（§14.2 / 神経修飾場の L1 の RL 版）** ―― は、複数行動を要する環境でしか示せない。よって CartPole を離れ、最小の multi-mode 環境で「場が行動をアドレスする」ことを RL で示す。

**環境 `MultiModeReach`**（`tmp/rl/envs_multimode.py`）: 1 次元・2 ターゲット（±1）。各エピソードのレジーム（どちらを目指すか）は**観測に含めない**。したがって行動を選べるのは**ノイズ場だけ**で、場の役割が観測と冗長にならない。場は recruitment field（`field.recruit`：半分のユニットを σ、残り半分を 0＝分離サブネット動員、§7.1）で、P_0 と P_1 が共有重み上の別サブネットを担う。**※ この σ-only ゲートで $\nu_k=0$ が厳密に成立するのは、前活性が $T$ 方向に定数である第 1 隠れ層に限られる**（§7.1 の※、`docs/idea_core.md` §3.4）。層 $l\ge2$ では上流サンプルゆらぎが残るため $\sigma_k=0$ でも交差が漏れる（§21.4 で顕在化し、単一隠れ層に落として回避した。任意の層で厳密に沈黙させるには ρ/h ゲートが要る — §23.7）。学習は **episodic REINFORCE ＋ per-timestep baseline ＋ advantage 白色化**、actor credit は forward weight-mirror（転置重みなし）。

**結果（`tmp/rl_multimode.py`、`tmp/out/rl_multimode.png`）= 成立**:

- 学習は収束（return −71→−5.9、ターゲット到達）。
- **決定的テスト（場を固定、レジームは隠れたまま、3000 ep）**: 場 P_0（target −1）→ 終点 x = **−1.01 ± 0.09**、場 P_1（target +1）→ 終点 x = **+1.03 ± 0.08**。**終点は場だけで決まり、隠れたレジームに依らない。** すなわち **同一重みのまま、ノイズ場を切り替えると行動（目指すターゲット）が切り替わる** ―― 神経修飾場側の L1 addressing を RL で実現した最小実証（§14.2）。軌道図 `rl_multimode.png` は、同一初期条件から P_0 固定で全軌道が −1 へ、P_1 固定で +1 へ収束することを示す。

**実装上の要点**: 当初の TD critic は全負の dense 報酬でベースラインが機能せず学習が崩壊した（missing-baseline 問題）。critic を捨て REINFORCE ＋ per-timestep baseline にして解決。credit 側（forward mirror）はそのまま。

**限定と次**: `recruit(quiet=0)` は左右で概ね **disjoint なパラメータ**を使うため、厳密には「多重化」でなく「分割」寄り（L1 addressing は成立、L2 の overlapping multiplexing は未検証）。次段は (i) **報酬による自律的な場選択**（規制信号 → 場、§9 の modulatory core）、(ii) **重なりを持つ場**での多重化（神経修飾場の L2 の RL 版）、(iii) 場そのものを報酬で学習（σ eligibility、§10）。

### 21.3 自律的な場選択（§7.2・§9 の option 機構、Task #2 = 完了）

Sub-B は場を「与えれば」行動をアドレスすることを示した。ここでは場を**報酬で自律的に選ばせる**。選択器（softmax 選好 `theta[context, field]`）が文脈から場 prototype をエピソード毎に選び、行動本体は選ばれた場の下で x のみを見て動く。**prototype には意味を事前付与せず**（§7.2）、選択器と本体を**同時学習**する。実装 `tmp/rl/multimode_select.py` + `tmp/rl_multimode_select.py`。

**結果（5000 ep, `tmp/out/rl_multimode_select.png`）= 成立**:

- **選択器が自律的に分化**: 学習後 π(field|context) は完全な対角（ctx0→P0=1.00、ctx1→P1=1.00）。報酬だけで「文脈→場」の一貫した routing を獲得。
- **合成行動が正しい**: context 0（target −1）→ 選択 P0 → 終点 −0.96、context 1（target +1）→ 選択 P1 → 終点 +1.03。両文脈とも正しいターゲットへ。
- **2 文脈は異なる場へ**（縮退なし）。prototype の意味は与えていないので、この対応は**報酬による自己組織化**の結果である。

すなわち、報酬が「**文脈 → ノイズ場 → 行動**」の三段を自己組織化し、**ノイズ場が option として自律的に機能する**ことを RL で実証した（§7.2 の狙いの最小達成）。

**実装上の要点**: 選択器と本体の同時学習は後期に不安定化しうる。本体を Adam にすると ep3000 以降で片方の行動が崩壊（return −6→−22）。**本体を SGD にすると危険域を越えて安定**（選択器はいずれも正しく分化する）。CartPole での Adam-不安定と同型で、forward-mirror REINFORCE は SGD が堅い。

**到達点**: ノイズ場方向は、(§21.1) 単一タスクでは SR 対立を抜けない一方、(§21.2) 複数行動環境では場が行動をアドレスし、(§21.3) 報酬が場を option として自律選択できる、というところまで最小実証された。次段は 神経修飾場側の L2（overlapping 場での多重化・共有ユニット損傷）と、場そのものを報酬で形成する σ eligibility（§10、§19 の field credit 経路）。

### 21.4 重なり場での多重化 vs 分割（神経修飾場の L2 の RL 版、Task #3 = 完了）

§21.2/21.3 は **disjoint** な recruitment 場だった（2 行動が別ユニット群＝分割寄り）。ここでは **重なりを持つ場**（`field.overlapping_pair`、recruit_frac=0.7、P_0 と P_1 が中央 26 ユニットを共有、Jaccard(active)=0.41）で 2 行動を学習し、ユニット群を損傷（actor readout 列をゼロ化）して、行動が分割されているか多重化されているかを直接調べる。実装 `tmp/rl_multimode_lesion.py`。

**設計上の落とし穴と修正（NNN の重要な性質）**: σ だけを絞る recruitment（σ=0→不活性）が厳密に効くのは、**入力が T 方向に一定な層だけ**である（しきい値を連動させる ρ/h ゲートなら任意の層で厳密になる — §23.7。本実験は σ-only の時点の記録）。2 隠れ層だと、最終層の σ=0 ユニットも上流の揺らぎから発火するため、名目上の「共有／片側」分類が実使用と一致せず損傷テストが交絡する（実際、最初の 2 層版は名目 shared 損傷が無影響という辻褄の合わない結果になった）。**単一隠れ層**にすると場が readout 直前のユニットを直接ゲートし、σ=0 ユニットが真に dead になる（検証済み: P_1 下で p0_only ユニットの mean|z|=0）。以下は単一隠れ層のクリーンな結果。

**結果（3000 ep, `tmp/out/rl_multimode_lesion.png`）= 多重化を確認**:

重なり場でも 2 行動を完全に学習（baseline task error P0=0.00, P1=0.00）。損傷による task error の増分:

| 損傷群 | 行動0 の劣化 | 行動1 の劣化 |
|---|---|---|
| **shared**（共有 26） | **+0.94** | **+1.00** |
| P0-only（19） | +0.88 | +0.02 |
| P1-only（19） | +0.01 | +1.17 |
| random（26、対照） | +0.12 | +0.16 |

- **共有ユニットを損傷すると両行動が同時に崩壊**（+0.94 と +1.00）＝**共有サブネットが両行動を担う多重化の署名**。
- 片側専用ユニットはその行動だけを担う（P0-only→行動0 のみ、P1-only→行動1 のみ）。
- random 対照は小さい＝「特定の 26 ユニット」が効くのであって「任意の 26」ではない。

もし行動が**分割**されていれば、どのユニット群を壊しても高々一方しか劣化しないはず。共有群が両方を同時に劣化させることは、**区画分割仮説への直接の反証**であり、神経修飾場側の L2（分割でなく多重化）を RL で実証したことになる。§21.1 の限定（disjoint 寄り）への回答にもなっている。

**ノイズ場 RL のまとめ（§21）**: (21.1) 単一タスクでは per-unit σ 配分は SR 対立を抜けない、(21.2) 複数行動では与えた場が行動をアドレスする（L1）、(21.3) 報酬が場を option として自律選択する、(21.4) 重なり場で 2 行動が共有サブネットに多重化される（L2）。神経修飾場側の L1・L2 が RL 側で最小実証され、かつ報酬による自律選択まで到達した。残る本丸は場そのものを報酬で形成する σ eligibility（§10 / §19 の field credit 経路）。

### 21.5 場を報酬で形成：連続場中心の学習と補間（§7.3・§19、Task #4 = 完了）

これまでは場（prototype）を与えるか離散選択していた。ここでは **prototype を与えず、連続の場中心 $c\in[0,1]$（Gaussian recruitment bump、`field.bump`）を報酬で形成する**。場レベルの policy が文脈ごとの中心 $\mu_c[\text{ctx}]$ を持ち、各エピソードで $c=\mu_c+\xi$ を sample、bump(c) の下で本体が動く。報酬が場中心（場レベルの policy gradient、§19.3 の $u_{\mathrm{mod}}=\Xi^{-1}\xi$ に相当）と本体（forward-mirror REINFORCE）を同時に動かす。実装 `tmp/rl/multimode_field.py` + `tmp/rl_multimode_field.py`。単一隠れ層（§21.4 のクリーンなゲート）、本体 SGD。

**結果（5000 ep, `tmp/out/rl_multimode_field.png`）= 成立**:

- **対称初期から報酬が対称性を破り、2 つの連続場中心を自己組織化**（両中心 0.5 スタート → $\mu_c=[0.97, 0.18]$）。prototype に意味を与えていないので、この分離は報酬による自己組織化。
- 各中心で正しい行動：ctx0（target −1、中心 0.97）→ 終点 −1.05、ctx1（target +1、中心 0.18）→ +1.01。
- **連続 option の決定的テスト（補間）= 成立**: 学習後に場中心 $c$ を 0→1 に掃引すると、終点が **滑らかな sigmoid 状に +1 から −1 へ遷移**する（$c{=}0.55$ 付近で終点 ≈ 0 という**未学習の中間行動**が現れる）。すなわち場は離散スイッチでなく**連続 option 座標**であり、場空間で近い位置は部分的に重なる部分網を動員して近い行動を生む（§7.3 / §14.2）。

**正直な範囲**: これは §19 の「**場を連続潜在行動とみなし場レベルの REINFORCE で中心を動かす**」最小版（提案した手堅い入口）である。§10 の純粋な **per-unit σ eligibility**（$\psi_\sigma=g\,\phi_T'(-d/\sigma)$ を forward 統計から作り σ を直接 credit する）は、より NNN-native な次段として残る。ここで動かしたのはスカラーの場中心であって per-unit の σ を forward-covariance credit で更新してはいない。

**§21 の最終まとめ**: ノイズ場方向は最小実証を一通り達成した ―― (21.1) 単一タスクでは σ 空間配分は SR 対立を抜けない（否定）、(21.2) L1 addressing、(21.3) 報酬による場の自律選択、(21.4) L2 多重化（共有ユニット損傷）、(21.5) 場を報酬で連続形成＋補間で未学習の中間行動。神経修飾場側の L1・L2 に加え、報酬による自律選択・連続場形成・option 補間まで RL 側で示せた。最も NNN-native な残件は per-unit σ eligibility（§10）と、§19 の field credit 経路を weight mirror で forward 推定する版。

### 21.6 per-unit σ eligibility の実装と検証（§10）= 明確化的な否定

§10 の per-unit ノイズ場 eligibility $\psi_\sigma$ を実装し（`credit.sigma_grad` = 教科書の $-d/\sigma$ 形、`credit.sigma_grad_forward` = crossing 自身のノイズから $\partial z/\partial\sigma$ を局所推定する一般形。後者は `kde_slope` と同じく転置重み不使用）、autograd の $\partial\log\pi/\partial\sigma$ に対して cosine 検証した（`tmp/rl_sigma_credit.py`）。

**検証結果 = policy-score の σ 勾配は ill-posed**:

- 単一 pass の $\partial\log\pi/\partial\sigma$ は norm ~60 と大きいが、**独立な 2 pass 間の cosine ≈ −0.04**。すなわち per-pass の σ 勾配は**ノイズ支配**で、安定な per-unit 構造を持たない（重み勾配が Step A で cosine ~0.95 と安定だったのと対照的）。
- **200 pass 平均で $\partial\log\pi/\partial\sigma$ の norm は ~0**（単一 pass の約 $10^5$ 分の 1）。つまり **$\mathbb E[\partial\log\pi/\partial\sigma]\approx 0$**。
- $-d/\sigma$ 形も forward 推定形も、この（ノイズないしゼロの）gold に対し cosine ≈ 0。

**解釈（構想にとって重要）**: fixed recruitment の下では、**per-unit のノイズ量 σ は行動尤度に系統的な効果を持たない**（σ は分散＝探索/変動を変えるだけで、平均方策 $\bar\phi$ をこの動作点では動かさない）。ゆえに policy-score から σ を credit する §10 の枠組みは、動作点近傍では系統信号がなく成立しない。これは §17.2-2 の指摘（$\psi_\sigma$ は探索共分散でなく平均応答の感度）を実測で強めたもの。

**これは §21.5 の選択を裏づける**: 場の有用な自由度は「per-unit の σ 量」ではなく「**どのユニットを動員するか（recruitment）**」であり、それを動かすのは場中心のような**低次元座標**である（§19）。場中心を動かすと動員される部分網が系統的に変わり（→ REINFORCE で学習可能、§21.5）、一方 fixed recruitment で per-unit σ を掃いても平均方策は変わらない（→ 勾配 ≈ 0）。したがって **場は §10 の per-unit σ でなく §19 の低次元 recruitment 座標として報酬学習するのが正しい**、という設計判断が実証された。

**残る可能性**: σ の効果は recruitment 境界（σ=0↔σ>0）でのみ系統的に現れうる（境界ユニットの σ を上げると動員が増える）。すなわち per-unit σ credit は bulk では ~0 でも境界で非零の可能性がある。ただし主要な場の学習は低次元座標で足りることが §21.5 で示されており、per-unit σ を forward-covariance で直接学習する路線は、少なくとも policy-score 経由では優先度が低い。**per-unit の内部ノイズ $\sigma_k$ については**、探索/変動への効果（探索温度の学習）は policy-score でなく return 分散を通じた別チャネルを要する（今後）。

**※ 限定の明確化（§25.3(c) と整合させるための注記）。** 本節が ill-posed と結論したのは、**per-unit の内部ノイズ $\sigma_k$ に対する尤度スコア**である。これに対し、**行動レベルの分散 $\sigma_a$ に対するスコア** $\partial\log\pi/\partial\sigma_a=((a-\mu)^2-\sigma_a^2)/\sigma_a^3$ は well-posed である（§25.3(c)）。両者が食い違わない理由は、$\sigma_a$ が方策分布のパラメータとして尤度に**直接**現れるのに対し、per-unit の $\sigma_k$ は動作点近傍では平均方策 $\bar\phi$ を動かさず尤度に映らないからである。したがって「探索温度は policy-score では学習できない」という言明は **per-unit $\sigma_k$ に限った主張**であって、行動レベルの温度には及ばない。この 2 段（スコアを $\sigma_a$ が観測可能な行動レベルで取り、per-unit でなく低次元座標へ写す）が §25.3 の設計の核である。

---

## 22. 結論：到達点と最も NNN-native な RL

**（注記）** 本節は §20–21（最小環境での検証）時点の結論である。その後の実タスク展開（§23 補遺：swing-up の完全解決・ρ/h ゲート・完全 NNN critic・PPO/SAC 統合）を含む最終的な到達点は §0 の一覧と §23.12 を参照。本節の「限定と今後」(a)(b) は §23 で部分的に前進し（実タスク＝swing-up での検証・ρ/h ゲートによる厳密分離）、(c)(d) は未解決のまま残っている。

### 22.1 到達点の総括（二本柱）

本プログラムは、構想の二本柱をそれぞれ最小実証した。

- **学習則の自然統合**（§20）: online で forward mirror が policy gradient 方向を復元し（Step A, cosine ~0.95）、forward path 内の credit だけで CartPole を学習し（Step B）、単一 NNN・単一ノイズで policy+value+探索+credit を担う（Task #1）。ただし単一 $\sigma$ は計算忠実度と制御を同時最大化せず（SR 対立, §20.16）、RL は低忠実度 credit に頑健、という限定が付く。
- **ノイズ場による行動モード**（§21）: 与えた場が行動をアドレスし（L1）、報酬が場を option として自律選択し（§21.3）、重なり場では 2 行動が共有サブネットに多重化され（L2, §21.4）、連続場座標の補間で未学習の中間行動が現れる（§21.5）。per-unit σ でなく低次元 recruitment 座標が正しい学習対象である（§21.6）。

### 22.2 最も NNN-native な RL：ノイズ場を option 変数とする階層 RL

これまで試した方式のうち、**NNN からRLを考える必然性に最も富み、NNN の定義的機構に最も密に統合されている**のは、**ノイズ場（recruitment 場）を行動モード＝option 変数として用いる方式**である。とりわけその最密結合形は、**連続場座標を報酬で学習する（§19 / §21.5）** ことと、**重なり場での多重化（§21.4）** の組合せである。

**なぜこれが「必然性に最も富む」か。** 判定の基準は「RL の構造が NNN に *押し付けられて* いるか、NNN から *示唆されて* いるか」である。

- **forward-covariance credit（§3, §20）は RL→NNN の方向**である。出発点は標準 RL（policy gradient / actor-critic）で、「その勾配を NNN の forward ノイズで実装できるか」を問う。RL の構造は既存のまま、NNN は backprop の代替を提供する。実際これは backprop と同等以上でも以下でもなく（§20.14）、しかも $\sigma$ の二役に二律背反を抱える（§20.16）。有用な *配管* だが、NNN 固有の RL を生まない。
- **ノイズ場 option（§7, §21）は NNN→RL の方向**である。NNN の定義的機構は「ノイズ強度がどのサブネットワークを機能させるかを決める（確率共鳴／recruitment）」ことにある。ここから出発すると、自然に次を問うことになる：**「ノイズ場を制御可能な内部状態にしたら何が起きるか」**。答えは、場が下位行動の選択子になる ―― これはそのまま **options / 行動モードの階層 RL 構造**である。標準 RL では option を別パラメータの sub-policy として *後付け* する（option-critic 等）が、NNN では **同一重み集合の上に多重化された複数方策を、ノイズ場が addressing する**（§21.2, §21.4）。すなわち option 構造を発明する必要がなく、NNN のノイズ機構に *既に埋め込まれている*。

**※ この 2 つの「方向」は着想の由来の区別であって、$\theta$ と $\mathcal P$ が別々の学習機構で動く 2 つの系だという意味ではない**（§1 の※、`docs/idea_core.md` §4.5）。両者は同一の前向き共分散 credit $\hat g$ を、外積の相手として $z_{\mathrm{prev}}$ に当てるか $-d/\sigma$ に当てるかで読み分けているにすぎず、$\mathcal P$ が動かせるのは各ユニットの動径 1 次元だけで $\theta$ の作用の部分空間である。

**この方式が密結合である具体的な理由**（実証と対応づけて）：

1. **機構が NNN 固有**。recruitment（ノイズがサブネットワークを機能ゲートする）は標準ネットには存在しない。σ=0 のユニットは、入力が T 方向に一定な層では σ 単独で、一般の層では ρ/h ゲート（$\sigma=\rho\sigma_0,\ h=h_0/\rho$；§23.7）で、真に不活性化する（§21.4・§23.7 で検証）。この物理量がそのまま option 選択子になる。
2. **多重化 vs 分割を実証**（§21.4, L2）。重なり場で 2 行動を学習し、**共有ユニットを損傷すると両行動が同時に崩壊**、片側専用ユニットは片側のみ崩壊。これは「複数方策が共有サブネットに多重化されている」ことの直接証拠であり、神経修飾場側の L2 を RL で示したもの。標準の option 分割では出得ない署名である。
3. **場は連続の内部状態**（§21.5）。報酬が場中心を動かし（場レベルの policy gradient, §19.3）、対称初期から 2 つの連続場座標を自己組織化する。学習後に座標を補間すると終点が滑らかに遷移し（$c\approx0.55$ で中間行動）、場は離散スイッチでなく **連続 option 埋め込み** である。連続的な行動モード間の滑らかな遷移という、NNN の graded recruitment に固有の性質が RL の option 補間として現れる。
4. **報酬による自律選択**（§21.3）。文脈から場を選ぶ選択子が、意味を事前付与していない prototype に対し、報酬だけで一貫した「文脈→場→行動」対応を自己組織化する（選択器 π(field|context) が完全な対角に収束）。option の獲得が報酬から創発する。
5. **同じノイズが探索も担う**。行動 sample のばらつきは NNN 内部ノイズそのもので、探索を外付けしない。場は「どのサブネットワークで探索するか」まで規定する。
6. **§21.6 が答えを鋭くする**。per-unit の σ 量は policy-score に系統効果を持たず（$\mathbb E[\partial\log\pi/\partial\sigma]\approx0$）学習対象として ill-posed。有用な自由度は「どのユニットを動員するか＝recruitment」であり、それを動かすのは **低次元の場座標**である。つまり NNN-native な RL 変数は raw な per-unit ノイズ強度ではなく、**recruitment 場の低次元座標**だと実測が特定した。これは option 変数としての場という描像を、さらに具体化・限定する。

**神経科学的な含意**。この方式は、「ノイズ場＝神経修飾場」という描像（`docs/idea_neuromod.md` §2–§5）の RL 版に当たる。神経修飾物質が回路の実効的参加（ゲイン／興奮性）を変えて行動モードを切り替える、という計算原理が、ここでは「報酬で学習・選択される低次元ノイズ場が、共有重み上の多重化方策を option として addressing する」という形で実装される。option 変数が抽象的な離散 index ではなく **物理量（ノイズ場）** である点が、NNN 起点の RL としての特異性である。

**位置づけの要約**。forward-credit（§20）は、場も重みも forward-native に学習可能にする **不可欠な配管**であり、実際 §21 の場学習もこの credit の上に乗っている。しかし **NNN でなければ生まれない RL 構造**、すなわち「thinking RL from NNN の必然性」が最も濃いのは、**ノイズ場を連続 option 座標として報酬学習し、共有重み上の多重化方策を addressing する** 方式である。

**限定と今後**。(a) 実証はいずれも最小環境（CartPole／1D 2-target）であり、規模・行動数を上げた検証（§14.2 の Foraging/Avoidance/Sheltering）が要る。(b) L2 は重なり場だが、学習された多重化が recruitment 境界にどこまで依存するかの解析が残る。(c) 場の唯一の未踏 DOF は「探索温度としての $\sigma$」で、**per-unit の内部ノイズ $\sigma_k$ については** policy-score が使えず **return 分散を通じた別チャネル**を要する（§21.6）。〔※ 限定を追記。行動レベルの分散 $\sigma_a$ に対するスコアは well-posed であり、§25.3(c) がこの経路を採る。〕(d) §19 の場-credit 経路を weight mirror で完全に forward 推定する版（場中心すら autograd を使わない）は、自然統合を実装レベルで最後まで閉じる課題として残る。

---

## 23. 補遺：CartPole 振り上げ安定化への挑戦（2026-07-20 着手 → 07-23 完結：full balance・ρ/h ゲート・完全 NNN 化・PPO/SAC 統合まで）

**（本章の読み方）** 実施順・失敗込みの実験記録である。§23（無印）は初日の部分的成功の記録で、その限界は後続節で順に解消された：full balance は §23.1（外付け critic）→ §23.9（完全 NNN）、スキル保護は §23.4（漏れで部分的）→ §23.7（ρ/h ゲートで解決）、学習の安定性は §23.10（PPO）、off-policy 化は §23.11（SAC）、総括と一区切りは §23.12。

バランス（§20）や multi-mode option（§21）より難しい **swing-up**（ポールを下垂れから振り上げて頂点で安定化）に、同じ NNN RL（forward-mirror credit、転置重みなし）で挑戦した。実装 `tmp/rl/envs_swingup.py`（標準 cart-pole 力学、下垂れ開始、cos(θ)+エネルギー整形報酬、カートは壁で full horizon、obs=[x/xthr, ẋ, cosθ, sinθ, θ̇]、bang-bang 2 行動 ±F）、`tmp/rl/swingup.py`（episodic REINFORCE）、`tmp/rl_cartpole_swingup_demo.py`（matplotlib cart-pole renderer + gif）。

**到達点 = 部分的成功**（デモ `tmp/out/rl_cartpole_swingup_demo.gif`）:

- episodic REINFORCE（cov_jac forward-mirror credit）＋ per-timestep baseline ＋ ランダム初期角カリキュラム ＋ エネルギー報酬 ＋ **壁張り付きペナルティ**で、下垂れから **本物のポンピング**（カートを中央付近で振動させる）を獲得し、ポールを頂点（cos_max = 1.0）まで振り上げる。
- eval-from-bottom の mean cos が checkpoint で **−0.23 → +0.28** と改善し、**獲得過程がアニメ化できた**（ハング → ポンピング学習 → 振り上げ）。
- **ただし頂点で安定化しない**（frac_upright ≈ 0.2；上げても落ちて再度振り上げる）。full swing-up-and-balance には未達。

**難所と診断（正直な記録）**: 素の cos 報酬ではカートが即境界へ（→壁化で full horizon）。次に方策が **定数行動に潰れる**（常に片方へ押し、壁反動で受動的に上がるだけ；greedy eval が checkpoint 間で凍結）。one-step actor-critic・カリキュラム・エネルギー報酬でも潰れ、**壁張り付きペナルティで局所最適を壊して初めて本物のポンピングが出た**。安定化未達は **bang-bang（no-op なし）の限界**（ハンド制御でも頂点滞在 14%）。

**full balance に要すると思われる次段**: 3 行動（no-op 追加。カート制御と頂点静止に有効。credit 機構は多出力 readout に一般化済み）または連続力制御。swing-up は標準深層 RL でも連続制御＋入念な調整を要する難題であり、本ツール（bang-bang・小規模網・forward-mirror REINFORCE）の適正スコープはバランスと option 実証で、swing-up は「振り上げの獲得」までが正直な到達点である。

### 23.1 full swing-up + balance の達成（方向1：連続力 NNN actor ＋ 外付け GAE critic・2026-07-21）

§23 の部分成功（bang-bang・forward-mirror REINFORCE）を、**統合を一点だけ割り切って**押し切った。NNN 完全統合の内製 critic を捨て、**actor は NNN のまま（cov_jac の forward-mirror credit、転置重みなし）、critic は外付けの MLP を backprop で学習**する A2C 構成にしたところ、**full swing-up + balance を達成した**。実装 `tmp/rl/policy_cont.py`（連続力方策）、`tmp/rl/critic.py`（外付け ValueMLP）、`tmp/rl/a2c_swingup.py`（GAE）。

**構成（方向1）**:

- **連続力 NNN 方策**（§3.1 の自然形）：readout 平均 $\mu$ を NNN が出し、行動 $a\sim\mathcal N(\mu,\sigma_e^2)$、力 = $F_{\max}\cdot a$。スコア $u=(a-\mu)/\sigma_e^2$ を **cov_jac** が body へ伝播（NNN の貢献）。〔※ 記号を修正。ここでの $\sigma_e$ は**外付けの探索ノイズ**であり、NNN の内部注入ノイズ $\sigma_k$ とは別物である（`docs/idea_core.md` §1.2）。以下、本補遺の探索温度はすべて $\sigma_e$ と綴る。〕
- **外付け critic**：標準 MLP を backprop で GAE リターンに回帰。内製 critic の破綻を排除（これが唯一の統合上の割り切り）。
- **GAE(γ=0.99, λ=0.95)** ＋ advantage 正規化 ＋ 下垂れ寄りカリキュラム。
- **探索ノイズ $\sigma_e$ は固定＋アニール**（0.4→0.1）。readout のサンプル分散に $\sigma_e$ を結ぶと、分散の縮小とともにスコア $(a-\mu)/\sigma_e^2$ が発散して inaction へ崩壊するため、探索だけ外付けの固定 Gaussian にした。〔※ **根拠づけを訂正。** 当初これを「§21.6 の σ ill-posed と整合する設計判断」と書いたが、根拠の取り違えである。§21.6 が ill-posed としたのは per-unit の**内部**ノイズ $\sigma_k$ に対する尤度 credit であって、行動レベルの分散 $\sigma_a$ のスコアは §25.3(c) のとおり well-posed である。$\sigma_e$ を外付けにした実際の理由は上記の**スコア発散＝分散フロアの会計の欠如**という実装上の問題であり、§25.3(a) の周辺分散 $\mathrm{var}_t=\sigma_a^2+\sigma_\mu^2$ による分散フロアで解消しうる。すなわちこれは原理的な帰結ではなく暫定の割り切りである（§25.1 も同旨）。〕

**結果（`tmp/out/swingup_a2c.pt`、eval-from-bottom, 500 step）**:

| update | mean cos | frac_up | **last100_up** |
|---|---|---|---|
| 25 | −0.76 | 0.00 | 0.00 |
| 250 | +0.13 | 0.20 | 0.21 |
| 300 | +0.35 | 0.25 | 0.39 |
| 325 | +0.30 | 0.45 | **0.80** |
| **350** | **+0.80** | **0.87** | **1.00** |
| 400 | +0.70 | 0.77 | **1.00** |

**last100_up = 1.00（評価末尾 100 step すべて頂点保持）＝ full balance 達成**。獲得過程は「下垂れ → 部分的な振り（upd150–250）→ キャッチ（upd325, last100_up 0.80）→ 振り上げ＋安定保持（upd350+）」と綺麗に段階化し、アニメ化した（`tmp/out/rl_cartpole_swingup_demo.gif`）。

**含意（重要）**: これは「NNN の cov_jac が backprop 相当の policy credit として実タスク（swing-up + balance）を解ける」ことの実証であり、cov_jac を追求した動機そのものの検証になる。連続行動は §3.1 のとおり NNN に最自然で「妥協」ではなく、割り切ったのは critic の外付けのみ。次段（方向3）は、この土台の上に **pump/balance の2レジームをノイズ場 option で切り替える** multimodal actor 化で、NNN 固有の解法へ昇華する。

### 23.2 方向3a：ノイズ場 option で swing-up + balance を解く（2026-07-21）

方向1（一様場・単一方策）で解けた swing-up + balance を、**ノイズ場 option の multimodal actor** で解き直した。swing-up は本質的に **pump（頂点から遠い）と balance（頂点近く）の2レジーム**を持つので、これを**共有重み上の2モードとして場でアドレスする**（§7.2 / §21 の実タスク版）。

**構成（方向3a）**: 方向1のパイプライン（連続力 cov_jac actor ＋ 外付け GAE critic）に、per-step の**場ゲート**を追加（`tmp/rl/a2c_swingup.py` の `_set_field`）。2つの場 prototype $P_\text{pump}, P_\text{balance}$（soft recruitment、off 側 0.3 で活性＝容量維持）を、文脈ゲート $g=\sigma(6\cos\theta)$ で連続ブレンドする。下（$\cos\theta<0$）では pump 場、頂点近く（$\cos\theta>0$）では balance 場が動員される。場が変わっても**重みは共有**。

**結果（`tmp/out/swingup_field.pt`、eval-from-bottom, 500 step）= 成立**:

| update | mean cos | frac_up | **last100_up** |
|---|---|---|---|
| 250 | +0.22 | 0.31 | 0.37 |
| **275** | **+0.64** | **0.70** | **1.00** |
| 300 | +0.52 | 0.61 | **1.00** |
| 325 | +0.78 | 0.83 | 0.97 |
| 400 | +0.29 | 0.46 | 0.92 |

**upd275/300 で last100_up = 1.00 ＝ full balance 達成**。方向1よりやや不安定（ピーク後に揺れる）だが、**同じ swing-up + balance を、pump/balance の2モードを場がアドレスする multimodal actor で解けた**。デモ `tmp/out/rl_cartpole_swingup_option_demo.gif` は、場のモード（pump/balance）をラベル表示し、下では pump 場・頂点近くでは balance 場に切り替わりながら振り上げ→安定保持する様子を示す。

**含意**: §21 で最小環境（reach）で示した「場が行動をアドレスする option 機構」が、**実タスク（swing-up + balance）で機能する**ことの実証。しかも actor 勾配は cov_jac のまま（転置重みなし）。ここまでで NNN の RL は、(i) forward-mirror credit が backprop 相当の policy credit として難タスクを解き（§23.1）、(ii) その actor をノイズ場 option の multimodal 構成にしても解ける（§23.2）ことが示された。

**次段**: (3b) 場ゲートを固定文脈でなく**学習**（§21.3/§9 の modulatory core）にして、報酬が pump/balance モードを自律組織化することを示す。(4) 外付け critic も NNN cov_jac（GAE リターン回帰）に置換し、**全体を単一 NNN**として閉じる。

### 23.3 方向3b：場ゲートを学習する（modulatory core）― 解けるが自律分化はしない

3a の固定文脈ゲートを、状態から pump/balance ゲート $g$ を出す**小さな NNN modulatory core（cov_jac）**に置換し、$g$ を潜在行動（§19）として同じ GAE advantage で学習した。core も NNN なので **actor は完全 NNN**（force body ＋ gate core）、外付けは critic のみ。実装 `tmp/rl/gate_swingup.py`。

**結果（`tmp/out/swingup_gate.pt`）= 2つの側面**:

1. **タスクは解ける**：last100_up = 1.00（upd250 以降）、mean cos は +0.91 まで上昇。学習ゲート＋完全 NNN actor で swing-up + balance を達成。
2. **しかしゲートは pump/balance に自律分化しない**：学習後の $g$ は pump 領域（$\cos\theta<-0.5$）と balance 領域（$\cos\theta>0.5$）でほぼ同値（例 upd450: 0.29 vs 0.28、upd400: 0.82 vs 0.79）。ゲートは**文脈非依存のほぼ定数**に収束し、単一の場ブレンドで body が全体を解いている。

**なぜ分化しないか（重要な洞察）**: swing-up は方向1が示したとおり**単一方策（一様場）で解ける**ため、場を pump/balance で切り替える**必要がない**。報酬にモード分化への圧力がないので、ゲートは適当な定数に収束する。これは §21.1（単一行動タスクでは場が分化しない）と整合し、§21.3（reach は隠れレジームで2行動が必要 → ゲートが分化した）と対照的である。すなわち **ノイズ場 option の自律的なモード組織化は、タスクが「単一の場では実現できない複数の行動」を genuinely 要求するときにのみ創発する**。

**含意**: swing-up で自律的な pump/balance 分化を示すには、モードを**必要**にする必要がある。例えば §21.3 の隠れレジーム法を適用し、**body には角度（$\cos\theta,\sin\theta$）を見せず**、gate core だけが全観測を見て場を設定する。すると body は「今 pump か balance か」を場経由でしか知れず、ゲートは分化を強制される。ただし body を角度盲にすると難度が上がる（要検証）。

**到達点の整理**: 3a（固定文脈ゲート）で「場が2モードをアドレスする actor が実タスクを解く」ことは示せた（§23.2）。3b は「学習ゲート＋完全 NNN actor でも解ける」ことに加え、「場の自律分化はタスク依存（不要なら起きない）」という §21 と一貫する知見を与えた。

### 23.4 スキル再利用の試み（balance を先に学習 → swing-up で再利用）― σ-only recruitment の漏れ（§23.7 で解決）

ノイズ場 option の**本来価値＝共有重み上でのスキルの合成・保護**（§7.1 / `docs/idea_consolidation.md`）を実タスクで示すべく、二段の継続学習を試みた（ユーザ提案）。Phase 1: 場を $P_\text{balance}$（サブネットワーク A=units[0:64]）に固定し **balance を A に事前学習**。Phase 2: **A を凍結**し、場ゲート（下=$P_\text{pump}$（B=[64:128])、頂点近く=$P_\text{balance}$（A））で swing-up を下垂れから学習、**pump を B に追加**しつつ **balance は凍結 A を再利用**する。実装は `tmp/rl/a2c_swingup.py`（`freeze_mask`・`fixed_field`・`norm_obj`・`energy_reward`）＋ `tmp/_consolidate_p1.py`/`_p2.py`。

**結果 = 部分的成立、しかしクリーンでない**:

- **Phase 1 成功**：A が balance を学習（現実的な handoff 開始 θ∈[0.1,0.3] から **last100_up = 1.00**）。純 cos 報酬・2 層で収束（64 ユニットの A では aggressive な θ̇=±1 摂動には弱いが、pump が届ける穏やかな状態は保持できる）。
- **Phase 2**：**A の重みは凍結（不変を検証）**、pump を B に学習。swing-up は **部分的**（last100_up ≈ 0.38）。
- **再利用はクリーンでない（漏れ recruitment）**：Phase 2 の方策の頂点 balance は last100_up ≈ 0.34 で、**A 単体の 1.00 から劣化**。損傷実験も分離しない（A/B いずれの ablation も両行動を劣化）。直接の機構は §21.4 の知見どおり **2 層では σ-only recruitment が漏れる**こと（balance フェーズ〔$P_\text{balance}$〕でも layer-1 の B ユニットが上流から発火し、pump 用に学習した readout-B 列がそれを読む）。ただし劣化の**主因**は再実験（§23.7）で精密化された：σ-only の Phase 1 では **balance スキル自身が漏れ B ユニット（訓練済み readout-B 経由）の上に建っており**、Phase 2 がそれを上書きしたことが大きい。当時の凍結マスクが A 不変量として不完全だった（fcs1 の A **行**の重み・bias と出力 bias が自由）ことも寄与候補である。

**含意（解決済み — §23.7）**: この限界は**ノイズ場 recruitment という機構の限界ではなく、σ-only ゲートの限界**である。本節の漏れは `docs/idea_consolidation.md` §12.7.2 が「$\sigma=0$ リーク」として発見・解決した現象と**同一**であり（$\sigma$ は交差を駆動する3成分〔上流サンプルゆらぎ・バイアス・自前注入ノイズ〕のうち自前ノイズしか縮められない、同 §4.5）、同 §4.6 の**動員ダイヤル $\rho$**（$\sigma=\rho\sigma_0,\ h=h_0/\rho$；off は $\sigma=0$ ＋ $h$ 番兵）を recruitment 場に使えば、上流ゆらぎが有界である限り**任意の層で $z\equiv0$ が厳密に成立**し、KDE スロープと credit も同時に厳密 0 になるため、汚染経路（pump 学習済み readout-B が漏れ発火を読む）と列の再成長が**構造的に**消える。この修正は **sample モデルと cov_jac（weight mirror）をそのまま保つ**。層をまたぐスキルの分離・保護が ρ/h ゲートの下でクリーンに成立することは §23.7 で実証済み。

**補足（statistic/analytic 系との関係）**: 漏れは per-sample の揺らぎ $[N,T,H]$ を層をまたいで伝播する `SimpleNNNSample` に固有であり、`SimpleNNNStatistic` / `SimpleNNNAnalytic`（層入力が前層の決定論的期待活性）では $\sigma=0$ が深さに依らず厳密に dead になるため、そもそも生じない（`activation.py`: 「radius = 0 makes both the output and the derivative exactly zero」）。ただしこれらへ乗り換えると `cov_jac` の weight mirror（`cov_weight`、$T$ sample の共分散が必要）を失い、credit を解析的な局所微分 $\phi_T'$（`phi_prime`）経由で構成し直す必要がある。したがって漏れ対策としては ρ/h ゲート（sample モデルのまま塞ぐ）が正解であり、statistic/analytic 化は credit 設計そのものを変える場合の選択肢に留まる。

**到達点**: 方向3 の swing-up は、(3a) 固定文脈ゲートの場 option で full balance（§23.2）、(3b) 学習ゲート＋完全 NNN actor で full balance（§23.3、ただし自律分化はしない）まで示せた。スキル保護は本節（σ-only ゲート＋不完全な凍結）では部分的（0.34）に終わったが、**ρ/h ゲート＋完全不変量凍結の再実験（§23.7）で、保持 1.000（構成的）・drift 0.00e+00・swing-up 1.000（olap アーム）のクリーンな保護として解決**した。

### 23.5 critic の NNN 化：単一 NNN（backprop ゼロ）で解く ― 学習するが critic が律速（§23.9 で解消）

当初計画の締めくくりとして、外付け MLP critic を **NNN critic（cov_jac、GAE リターンへの回帰）**に置換し、**actor・critic とも forward-mirror、転置重み backward をどこにも使わない単一 NNN システム**にした（ユーザ要望「全体を1つのネットワークとして見る」）。実装 `tmp/rl/critic.py`（`NNNCritic`）＋ `tmp/rl/a2c_nnncritic.py`。value 誤差 $(V-\text{GAE リターン})$ を top-level score として actor と同じ cov_jac 再帰で回帰し、リターンは running 標準化。

**結果（`tmp/out/swingup_nnnac.pt`、eval-from-bottom）= 学習するが full balance 未達**:

- 完全 NNN actor-critic は swing-up を**学習する**（mean cos −0.81 → +0.2〜0.29、last100_up はピーク ≈ 0.44）。**backprop を一切使わずポールを頂点まで振り上げ、時々保持する**。
- しかし **full balance（last100_up = 1.0）には届かない**（方向1の外付け MLP critic は 1.0 に到達）。critic が律速。

**診断（正直な限界）**: cov_jac の value 回帰は、backprop MLP critic より **advantage の質が低い**。理由は (i) forward-mirror 推定が backprop より高分散、(ii) バッチ 1 パス更新（MLP は 8 epoch backprop）で critic の学習が遅い、(iii) bootstrap する GAE リターン（critic 自身に依存）を高分散 critic で回帰するため誤差が乗る。これは本取り組みを通じた一貫観察 ―― **actor 側の policy credit は cov_jac が backprop 相当に機能する（§23.1 で full balance 達成）が、critic（価値回帰）は cov_jac だと質が落ち、full balance を支えきれない** ―― を追認する。

**含意（主張の較正）**: 「NNN cov_jac が backprop 相当」という主張は **actor（policy gradient）については実タスクで検証された**（§23.1）。一方、**完全 backprop フリー（critic も NNN）にすると、価値関数回帰の質が律速となり full balance は未達で partial に留まる**。したがって現時点の堅い到達点は「**cov_jac actor ＋（信頼できる）critic で full balance を解く**」であり、critic まで NNN 化した単一システムは「解きつつあるが critic 強化が課題」である。

**critic 強化の道**: (i) critic の cov_jac 更新を複数 epoch（各 epoch で再 forward、低速）にする、(ii) `cov_jac_full`（readout 誤差も forward 統計から）や pooled mirror で分散を下げる、(iii) critic を actor と body 共有（§6/§20.17）にして表現を強化、(iv) 場ゲートを併用する場合の漏れ対策は ρ/h ゲート（§23.7）で足りる（statistic/analytic 系は credit 設計を変える場合の選択肢）。→ **実施結果は §23.8–23.9：critic 律速は解消され、backprop ゼロ・全特徴学習のまま full balance に到達**（本質は EMA mirror + KP の RL 移植〔§23.9〕。§23.8 の凍結特徴は不要と判明、(iii) の body 共有は負の結果）。

### 23.6 次段の優先課題（作業再開時のメモ）

本セッションはここで一区切り（主目標＝NNN cov_jac で swing-up + full balance を解く、は §23.1/3a/3b で達成）。再開時の優先順位は以下。

**最優先 (A): NNN critic を強化し、完全 backprop フリー版（§23.5）を full balance へ届かせる。→ 解決済み（最終形は §23.9）：mirror の EMA 化（fix 1）だけで、critic の特徴を学習したまま last100_up = 1.000。§23.8 の凍結特徴は経由地（読み替えあり）。以下は当時の候補リスト（1 は不要と判明、2 が EMA mirror として本質、3 は負の結果 — §23.8）。**
現状、actor（cov_jac）は backprop 相当だが、critic まで NNN 化すると価値回帰の質が律速で last100_up がピーク 0.44 に留まる。原因に直接効く候補（推奨順）:
1. critic の cov_jac 更新を **複数 epoch**（各 epoch で再 forward。低速だが MLP の 8-epoch backprop に相当する学習量を与える）。
2. **`cov_jac_full`**（readout 誤差も forward 統計から）や **pooled mirror**（`cov_weight(..., pool=True)`）で mirror 分散を下げる。
3. critic を actor と **body 共有**（§6 / §20.17 の統合 critic）にして表現を強化。
4. 漏れ対策は **ρ/h ゲート**（§23.4 更新・§23.7）を第一候補とする。statistic/analytic 系への乗り換え（credit は解析 $\phi_T'$ 経由）は cov_weight mirror を失うため優先度を下げ、最終手段に回す。
目標: last100_up = 1.0 を **backprop ゼロ**で達成し、「単一 NNN で難タスクを解く」を完成させる。

**次点 (B): 素朴な A2C でなく、同系のより進んだ RL アルゴリズムを NNN（cov_jac）へ統合する。→ 両方実施済み：PPO は §23.10（成功・後半全 ckpt 1.000）、SAC は §23.11（feasible・PPO 比 4 倍遅）、比較検討は §23.12。以下は当時の計画の記録。**
現状は on-policy A2C＋GAE の素朴構成。cov_jac が $\nabla_W\log\pi$ を forward-only で与えるので、より進んだ policy-gradient 系は自然に載る:
1. **PPO（最有力・直接の発展）**: clipped surrogate ＋ 複数 epoch の再利用。重要度比 $\pi_\text{new}/\pi_\text{old}$ のクリッピングで安定化・sample 効率向上。既存の GAE ＋ 連続 Gaussian-from-samples 方策（§3.1）にそのまま乗る。cov_jac は ratio の対数勾配 $\nabla\log\pi$ を供給。
2. **SAC（より野心的・off-policy）**: max-entropy ＋ replay ＋ twin critics。**entropy 項は NNN の内部ノイズ（探索）と概念的に直結**し、NNN 固有の魅力がある。〔※ **訂正。** 現行実装で方策エントロピーを決めているのは NNN の内部ノイズ $\sigma_k$ ではなく、外付けの探索ノイズ $\sigma_e$ である（§23.1 の※）。$\sigma_e$ が固定である限り entropy 項は定数となり、SAC の $\alpha$ 自動調整も無意味になる（実際そうなった — §23.11）。この「直結」が成り立つのは、行動ノイズを NNN の物理ノイズに置き換えたとき、すなわち §25.3 の内部化を経た後である（§25.3(c) 末尾が同じ接続を指摘している）。〕ただし off-policy と cov_jac の整合（mirror は方策非依存だが score は importance 補正が要る）と replay 下の mirror 推定は要検討。
目的: swing-up+balance を超え、より難しい連続制御へ NNN RL を広げ、「NNN RL が実用的に competitive」を強める。cov_jac actor はこれらの上位アルゴリズムの policy-gradient 部品として差し替え可能な位置にある。

**実装メモ**: 完全 NNN 版は `tmp/rl/a2c_nnncritic.py`（`critic.NNNCritic`）。critic 強化は critic 更新ループの epoch 化から。PPO 化は `train_a2c` に old-policy log-prob 保存＋ratio クリップを足すのが最小変更。

### 23.7 ρ/h ゲートによるスキル保護の再トライ（2026-07-22 実施）＝ 解決：漏れは機構の限界ではなく σ-only ゲートの限界だった

§23.4 の二段カリキュラム（balance を A=units[0:64] に事前学習 → A を凍結し pump を B=[64:128] に追加）を、§23.4 更新の優先順位 (1) に従い **ρ/h ゲート**で再実行した。実装は `tmp/rl/field.py`（`H_DEAD`・`recruit_rho`）、`tmp/rl/a2c_swingup.py`（`_apply_rho`：per-unit に $\sigma=\rho\sigma_0$ と $h=h_0/\rho$ を同時設定、$\rho\to0$ は h 番兵で厳密沈黙。`rho_mode` として後方互換で追加）、ドライバ `tmp/rl_consolidate_rho.py`、結果は `tmp/out/rl_consolidate_rho/`。文脈ゲートは §23.2 と同じ $g=\sigma(6\cos\theta)$ のブレンドを **ρ に対して**行う（σ は線形従動、$h=h_0/\rho$ は非線形従動；端では不感帯に入り厳密 off）。

**Phase 1（balance の獲得、ρ 場）。** $P_\text{bal}$ = {A: ρ=1, B: ρ=0}。B は**両層とも max mean $z$ = 0.0000 の厳密沈黙**（σ-only 対照では L1 が 0.34–0.48 で漏れ発火 = §23.4 の漏れの最小再現、V1）。3 seeds × 400 updates（6 episodes/update）で **3/3 seeds が handoff 開始（θ∈±[0.1,0.3]）から last100_up = 1.000** に到達。ただし方策勾配は途中崩壊と回復を繰り返すため、50 updates ごとの ckpt 評価と最良選択が必須だった（2/3 seeds は途中 ckpt が最良）。厳密ゲート下の balance は漏れ B の隠れ容量を使えないぶん学習は重いが、**獲得されたスキルは A に完全に局在する**。

**Phase 2（pump の追加、3 アーム）。** 共通：phase-1 最良 body（seed 2）から開始、**A の完全不変量を凍結**（fcs0 の A 行・bias、fcs1 の A **行**全列・A bias、readout の A 列、出力 bias。§23.4 の凍結は fcs1 の A **列**のみで、A 行の入力側と出力 bias が自由だった — 不完全）、外付け GAE critic・energy 報酬・400 updates（ρ hard は 800 も実施）。

| arm | prototypes | V1 漏れ（$P_\text{bal}$ 下の B、L1） | V2 保持 last100_up | V3 drift | V4 swing-up last100_up |
|---|---|---|---|---|---|
| σ-only（§23.4 対照） | `recruit` σ 場 | 0.48（漏れ） | 1.000 | 0.00e+00 | **1.000**（upd 50 で到達） |
| ρ hard | pump={B:1,A:0} / bal={A:1,B:0} | **0.0000（厳密）** | **1.000（構成的）** | 0.00e+00 | 0.71 ピーク（400/800 upd とも振動・未達） |
| **ρ olap（推奨）** | **pump={A:1,B:1} / bal={A:1,B:0}** | **0.0000（厳密）** | **1.000（構成的）** | 0.00e+00 | **1.000**（upd 50 で到達） |

**知見：**

1. **ρ/h ゲートは主張どおり機能する（V1–V3）。** ρ 系アームでは頂点近くで B が厳密沈黙し（ゲートが不感帯へ入る）、不変量が凍結済みなので、合成方策の balance は phase-1 スキルと**構成的に同一** — 保持 1.000 は測定値である以前に構造の帰結である。§23.4 が要求した「readout gating か非漏れ機構」は ρ/h ゲートで充足され、**sample モデル・cov_jac のまま**解決した。statistic/analytic 化は不要。
2. **想定外：σ-only 対照は §23.4 の劣化（0.34）を再現しなかった。** 原因は phase 1 の質にある。本実験の phase 1 は ρ ゲートで獲得したため balance が最初から A に完全局在し、pump が readout-B を上書きしても壊れるものがない。対して §23.4 の phase 1 は σ-only（漏れあり）で学習しており、**balance スキル自身が漏れ B-l1 活動（訓練された readout-B 経由）に部分依存**していた — それを phase 2 が上書きしたのが 0.34 の主因と再解釈できる（不完全な凍結マスクも寄与候補）。教訓：**漏れの実害は「後続学習による汚染」以上に「スキルが保護外の漏れユニットの上に建つこと」であり、獲得時のゲートの厳密性が決定的**。なお σ-only の保持 1.000 は頂点近くの B 寄与が共訓練された経験的な値であり、構造的保証ではない。
3. **ρ hard の代償：共有ゼロの pump は B 単独（64 ユニット）では安定に解けない**（best 0.71、800 updates でも 0.15↔0.71 を振動）。σ-only が即座に解けた（upd 50）のは、漏れが底で「凍結 A の特徴と出力」を**意図せぬ共有語彙**として供給していたためである。漏れは保護を壊す欠陥であると同時に、偶然の soft sharing として獲得を助けてもいた — §21.4 の多重化、および consolidation の hard/soft トレードオフ（`idea_consolidation.md` §12.9.6）の RL 再現。
4. **ρ olap が両立を与える（主結果）。** pump モードで凍結 A を**意図的に**全動員（read-only 語彙 = consolidation 案2 readout-share の RL 版）し、balance モードでのみ B を厳密ゲートすると、swing-up は σ-only と同速（upd 50 で 1.000）、保持は構成的 1.000 のまま。**「保護が要る場所では hard、共有が有益な場所では soft」をモードごとに選ぶ自由度が、単一ダイヤル ρ の場の設計だけで表現できる。** §23.4 への最終回答：漏れ recruitment の限界は機構の限界ではなく σ-only ゲートの限界であり、ρ への一般化は保護をクリーンにするだけでなく、漏れが偶然与えていた共有をも設計変数に変える。
5. 限界の記録：phase 2 は単一 seed・単一タスク対。phase 1 の方策勾配の不安定性（best-ckpt 選択で吸収）は ρ と独立の A2C 側の課題。ρ の中間値（恒常的な部分動員）は本実験ではブレンド過渡にのみ現れ、その効用は未検証（consolidation §12.9.15 の sin 族では限定的だった）。

**到達点の更新**：§23.4 の「スキル保護は部分的」は撤回され、**層をまたぐモジュラーなスキルの分離・保護・合成は、ρ/h ゲート（+ 完全不変量の凍結）の下でクリーンに成立する**。§23.6 (A)-4 の漏れ対策はこれで完了とし、statistic/analytic 化は critic 強化の文脈でのみ検討すればよい。

**デモ**：olap アームの獲得過程（upd 0 = pump 未学習でぶら下がり → upd 50 で full swing-up + balance、場モード〔PUMP: B + 凍結 A 語彙／BALANCE: A のみ・B 厳密沈黙〕を色分けラベル表示）をアニメ化した。実装 `tmp/rl_consolidate_rho_demo.py`、出力 `tmp/out/rl_consolidate_rho_demo.gif`。

### 23.8 critic 律速の解消（2026-07-22 実施）＝ backprop ゼロの単一システムで full swing-up + balance 達成

§23.5 の残課題（critic を NNN 化すると価値回帰の質が律速で last100_up ≈ 0.44 止まり）に対し、原因を4つの非対称性に分解した上で、修正候補を段階的に検証した。実装は `tmp/rl/critic.py`（`FrozenNNNFeatures`）、`tmp/rl/credit.py`（`MirrorEMA`）、`tmp/rl/a2c_sharedv.py`、ドライバ `tmp/rl_sharedv_swingup.py`、結果は `tmp/out/swingup_sharedv*.pt`・`tmp/out/sharedv_*.log`。

**原因の分解（§23.5 の診断の精密化）。** (1) actor は勾配の**方向**だけ合えばよい（cosine ~0.95、誤差は更新平均で洗われる）が、critic の出力は**値**として GAE bootstrap に直接消費され、誤差が自己増幅する。(2) NNN critic は完璧に学習できても T サンプル評価ノイズを advantage に直入させる。(3) RL 側の `cov_jac_grad` は**毎ステップ1状態の T サンプルから mirror を単発再推定**しており、教師あり側で実証された永続 EMA ＋ Kolen–Pollack 追跡（`CovJacTrainer`）と乖離していた。(4) 価値関数の頂点近傍の鋭い構造に対する表現解像度（cov_jac は $|w|$ を育てにくい — consolidation §12.9.12 と同根）。

**修正の梯子（負の結果2つを含む）：**

- **A（共有 body ＋ 線形 value head）＝ 負の結果。** value を actor の最終隠れ層平均活動 $\bar z$ への線形ヘッド（update ごとに ridge で厳密再フィット、mirror 不要）とし、value 誤差は body に流さない。序盤は value R² +0.88 と適合するが、**actor の更新が特徴写像を動かし続けるため適合が維持できず**（R² → +0.01）、方策も退化（best 0.18）。教訓：**critic に必要なのは「actor が学んだ表現」ではなく「定常で十分に豊かな基底」**である。value は行動選択と異なる特徴（状態の良さの大きさ情報）を要し、policy 特徴の線形プローブでは張れない。Task #1 の「body 共有 critic」路線（§20.17、§23.6-A3）への明確な警告になる。
- **A+B ＝ 同じく失敗**（best 0.21）。B は A の設計欠陥を救済しない。
- **A′（凍結ランダム NNN 特徴バンク ＋ ridge ヘッド）。** critic を「**未訓練で凍結**した単層 SimpleNNNBase（H=256、マルチスケール行ゲイン {1,2,4} — 案C §12.9.12 の流儀）の平均交差活動 + 生 obs」への線形ヘッドにする。特徴が定常なので厳密フィットがそのまま有効で、**value R² は全訓練を通じ +0.93〜+0.99 を維持**。ただし単独では best 0.29 — critic は直っても actor 側の律速 (3) が残る。
- **B（永続 EMA mirror ＋ Kolen–Pollack を RL に移植）。** `MirrorEMA`（β=0.1、定常分散 ≈ 単発の 1/19；actor 更新の適用量を mirror に加算して追跡）で actor credit の mirror を置換。

**結果（seed 0、400 updates、§23.5 と同条件）：**

| 構成 | value R² | swing-up last100_up（best） |
|---|---|---|
| §23.5 separate NNN critic（cov_jac body） | — | 0.44 |
| A：共有 body ＋ 線形ヘッド | 0.88 → **0.01 に崩壊** | 0.18（失敗） |
| A＋B | 0.25 | 0.21（失敗） |
| A′：凍結特徴 ＋ ridge ヘッド | 0.94–0.98 | 0.29 |
| **A′＋B** | **0.93–0.99** | **1.000**（upd 300・375） |

**知見：**

1. **backprop ゼロの単一システムで full balance に到達した。** actor = cov_jac（EMA mirror、転置重みなし）、critic = 凍結 NNN 特徴 + ridge ヘッド（backprop なし・mirror なし）。§23.6 最優先課題 (A) は達成である。
2. **二つの律速が直列だった。** critic の質（A′ で解消：R² 0.98）と actor の mirror 分散（B で解消）は独立の律速で、**両方直して初めて 1.000 に届く**（A′ 単独 0.29、B は A 系では無力）。§23.5 の「critic が律速」という診断は正しかったが不完全で、critic を直すと次は per-step 単発 mirror が律速になる。
3. **critic に要るのは学習された表現ではなく定常な基底。** 凍結ランダム基底（reservoir/ELM 的）で R² ~0.98 が出る一方、taskに最適化され続ける policy 特徴では 0.01 に崩れる。consolidation の「初期化は学習が再生産しない資源」（§12.9.16）・マルチスケール基底（案C）と響き合う結果で、「NNN critic」の正しい形は**凍結 NNN 基底 + 適応読み出し**である。
4. **正直な限界。** (i) 単一 seed。(ii) 学習は振動的で（1.000 の後 0.11 へ落ちる upd もある）、§23.7 phase 1 と同様に best-ckpt 選択が必要 — A2C 側の安定化（PPO 化、§23.6-B）が次の課題。(iii) ridge ヘッドは閉形式解であり局所学習則ではない（forward-only・weight transport なしではあるが、オンライン局所化するなら RLS/LMS への置換が要る）。(iv) B の β・A′ の H/スケールは未チューニングの初期値である。

**追記（重要な但し書き）**：本節の A′（凍結特徴）は「NNN の特徴も forward 学習される」という構想の核と相容れない、という指摘（ユーザ）を受けて、**特徴を学習したまま**の最小修正を §23.9 で再検証した。結果、**凍結は不要**であり、律速は mirror 分散だけだったことが確定した。本節の恒久的な寄与は knowledge 2（二つの律速が直列）と knowledge 3 の前半（critic の特徴は速い学習ヘッドより遅く動く必要がある）に縮小され、「凍結ランダム基底」はその十分条件の一つ（かつ NNN 的に不自然なもの）にすぎなかった、と読み替えるべきである。**§23.9 の採用に伴い、本節の実装（`FrozenNNNFeatures`・`a2c_sharedv.py`・`rl_sharedv_swingup.py`）はコードベースから削除した**（結果ログ `tmp/out/sharedv_*.log`・`swingup_sharedv*.pt` は記録として残置）。本節は負の結果（A：body 共有の崩壊）と読み替えの記録として残す。

### 23.9 §23.5 の最小修正：mirror の EMA 化だけで、学習される critic のまま full balance（2026-07-22 実施）

§23.8 A′ の凍結特徴に対する妥当な批判 —「forward 処理だけで誤差逆伝播に相当する学習ができ**特徴も学習される**のが NNN の特徴であり、凍結は解決になっていない」— を受け、**§23.5 の完全 NNN actor-critic（critic の特徴も cov_jac で学習）に対し、fix 1（`MirrorEMA` + Kolen–Pollack を actor と critic の両方へ）だけを適用**した。他は §23.5 と同一（critic は 2 隠れ層 H=64 の `NNNCritic`、単発 mirror を EMA β=0.1 の永続 mirror に置換したのみ）。実装は `tmp/rl/a2c_nnncritic.py` の `mirror_beta` オプション（既定 None = §23.5 と同一経路）、ドライバ `tmp/rl_nnncritic_ema.py`、結果は `tmp/out/swingup_nnnac_ema.pt`・`tmp/out/nnnac_ema.log`。

**結果（seed 0、400 updates、§23.5 と同条件）：last100_up = 1.000（upd 275 と 375 の2回）— §23.5 の 0.44 を突破し、凍結特徴なしで backprop ゼロの full balance を達成。**

| 構成 | critic 特徴 | value R² | best last100_up |
|---|---|---|---|
| §23.5（単発 mirror） | 学習 | —（低品質） | 0.44 |
| §23.8 A′＋B（凍結特徴＋EMA mirror） | **凍結** | 0.93–0.99 で安定 | 1.000 |
| **§23.9 fix 1 のみ（EMA mirror ×2）** | **学習** | 中央値 0.17、終盤 0.76–0.86（振動大） | **1.000**（upd 275・375） |

**知見：**

1. **§23.5 の律速の正体は mirror 分散だった（これで確定）。** critic の設計（学習される 2 層 NNN + cov_jac 回帰）自体は full balance を支えられる。欠けていたのは、教師あり側 `CovJacTrainer` が最初から持っていた「永続 EMA mirror + Kolen–Pollack 追跡」であり、RL 移植の際にこの設計が落ちていたこと（per-step 単発推定）が 0.44 の原因である。修正は機構の追加ではなく**実証済み設計への回帰**であり、「forward-only で特徴も学習される」という NNN の主張は critic を含めて無傷のまま成立する。
2. **§23.8 の読み替え。** A′（凍結特徴）が効いたのは「定常性」が本質だったからではなく、凍結が mirror 誤差を消す最も乱暴な方法（推定不要化）だったからである。A（actor body 共有）の失敗が示した「critic の特徴は学習ヘッドより速く動いてはならない」という条件は残るが、critic 専用 body を自分の value 誤差で学習する構成（§23.5/§23.9）はもともとこれを満たしていた — 動きが速すぎたのは actor に相乗りした場合だけである。
3. **代償は value fit の安定性。** 学習される critic の R² は凍結基底（0.93–0.99）より遥かに荒れる（−0.8〜+0.86 を振動、中央値 0.17）。それでも方策学習には足りる — actor に必要なのは advantage の**方向**の相関であり、fit の絶対品質ではない（§23.8 原因分解 (1) の裏面）。R² の安定化が必要なら §23.9 実施前に挙げた残りの候補（マルチスケール初期化・二時間スケール readout・λ→1 目標・EMA target）が積み増しの選択肢になるが、full balance の達成には不要だった。
4. **限界の記録。** 単一 seed。評価は依然振動的（1.000 の後に 0.19–0.34 へ落ちる ckpt もある）で best-ckpt 選択に依存 — これは §23.8 と共通の A2C 側の課題（次の一手は PPO 化、§23.6-B）。β=0.1 は未チューニング。

**§23.5–23.9 の総括**：「NNN cov_jac は backprop 相当か」への答えは、actor（§23.1）に続き critic でも肯定に変わった。**完全 backprop フリー・全特徴学習・転置重みなしの単一 NNN システムが CartPole swing-up + full balance を解く**。必要だったのは新機構ではなく、supervised で実証済みの mirror 設計（EMA + KP）を RL ループへ正しく持ち込むことだけである。

**採用（2026-07-22）**：本結果を受け、**EMA mirror（β=0.1）＋ KP 追跡を a2c 系トレーナのデフォルトにした**（`a2c_nnncritic.train_a2c_nnn` は actor・critic の両 mirror、`a2c_swingup.train_a2c` は actor mirror。いずれも `mirror_beta=None` で旧単発挙動に戻せる — §23.1–23.7 の過去実験の再現用）。凍結特徴系のコードは削除（§23.8 追記）。残課題だった振動（best-ckpt 依存）は **PPO 化（§23.10）で解消**した。

### 23.10 PPO 化：完全 NNN actor-critic の安定化 — 後半 checkpoint 全て 1.000（2026-07-22 実施）

§23.9 の残課題（評価が振動し best-ckpt 選択に依存）に対し、§23.6-B1 の計画どおり **PPO（clipped surrogate + エポック再利用）を完全 NNN actor-critic（§23.9 構成：cov_jac actor + 学習される NNN critic、両 EMA mirror）へ統合**した。実装は `tmp/rl/ppo.py`、ドライバ `tmp/rl_ppo_swingup.py`、結果は `tmp/out/swingup_ppo_s0.pt`・`tmp/out/ppo_s0*.log`。

**NNN 側の追加部品はほぼ不要である。** 方策は固定 σ_e の Gaussian（§23.1）なので log π が閉形式で、重要度比 $r=\pi_\text{new}/\pi_\text{old}$ は保存した $(a,\mu_\text{old})$ と再 forward の $\mu_\text{new}$ から計算できる。cov_jac の score $(a-\mu)/\sigma^2$ は $\partial\log\pi/\partial\mu$ そのものだから、**クリップ勾配は「clip 判定 × r × A × 既存 credit」**に帰着し、credit 機構は無変更で載る。critic は凍結バッチ目標（標準化 GAE リターン）への複数エポック回帰となり、§23.6-A1 の「複数エポック」も同時に実現される。

**しかし素朴な移植は2つの NNN 固有の問題で失敗した（v1–v2、負の結果の記録）：**

1. **v1 = ノイズ clip 凍結。** NNN の $\mu$ は T=64 サンプルのアンサンブル平均で推定ノイズ $\sigma_\mu\approx\mathrm{std}_T/\sqrt T$ を持つ。log 比はこのノイズだけで揺れ、σ_e が 0.1 までアニールされると**方策が動いていなくても** $|\log r|>\epsilon$ が約半分のサンプルで成立し（clip_frac ≈ 0.5）、勾配が死んで方策が凍結した（upd 75 以降 eval 完全固定）。決定論的ネットの PPO には存在しない、「**方策の平均自体が確率的**」であることの帰結である。修正：(i) 実行行動は実際には周辺分布 $\mathcal N(\bar\mu,\ \sigma_e^2+\sigma_\mu^2)$ から出ているので、score と比の分散を **var_t = σ_e² + σ_μ(t)²** にする、(ii) clip しきい値を per-sample の**ノイズ不感帯** $\epsilon_t=\epsilon+2\sigma_{\log r}(t)$ にする（比の偏差が自身のノイズ床を超えたときだけ発火）、(iii) 勾配スケールの r を $[1-\epsilon_t,1+\epsilon_t]$ にクランプ。
2. **v2 = 定数行動アトラクタへの崩壊。** ノイズ対応で凍結は解消し（clip_frac 0.02–0.22）、**upd 50 で full balance に到達（A2C の約5倍のサンプル効率）**したが、直後に §23 で既知の定数行動局所解へ落ちて回復しなかった。原因は2つ：(a) エポック再利用が同一バッチへの勾配を無制御に累積し trust region を踏み越える、(b) 保存していた行動が**クランプ後**の値で、μ が飽和域に近づくと score/比が実行サンプルと系統的にずれる（実装バグ）。修正：(a) **KL 早期停止**（$\mathrm{KL}\approx\overline{(\mu_\text{new}-\mu_\text{old})^2/2\mathrm{var}_t}>0.02$ でそのバッチの残エポックを打ち切る）、(b) クランプ前の行動を score から復元（$a=\mu+\mathrm{score}\cdot\sigma_e^2$）。
3. **v3 = ほぼ成功、終端で崩壊。** upd 75–250 の8連続 ckpt で 1.000 を維持（late-half mean 0.718）したが、σ_e が 0.1 へアニールし切る終盤に KL が暴騰（0.38–0.79）して崩壊。分散の縮小が実効ステップを肥大させる終端問題で、**探索 σ_e の下限を 0.2 に留める**（v4）ことで解消した。

**結果（v4 = 全修正、seed 0、300 updates）：**

| 版 | best last100_up | late-half mean | 備考 |
|---|---|---|---|
| A2C（§23.9） | 1.000 | ~0.4 | 振動、best-ckpt 依存 |
| PPO v1 | 0.28 | 0.15 | ノイズ clip 凍結 |
| PPO v2 | 1.000（upd 50） | 0.15 | 定数行動へ崩壊 |
| PPO v3 | 1.000 | 0.72 | upd 75–250 維持、終端崩壊 |
| **PPO v4** | **1.000** | **1.000** | **upd 75–300 の10連続 ckpt すべて 1.000、崩壊なし** |

**知見：**

1. **振動問題は解消した。** v4 は upd 75 以降のすべての checkpoint が full balance で、best-ckpt 選択は不要（最終 ckpt がそのまま 1.000）。訓練時リターンも +0.77〜0.84 で持続し、value R² は 0.89–0.92 で安定（critic の複数エポック回帰の効果。A2C の中央値 0.17 と対照的）。**backprop ゼロ・全特徴学習・転置重みなしの単一 NNN システムが、swing-up + full balance を安定に学習・保持する**に至った。
2. **サンプル効率も向上した。** full balance 到達は upd 75（v2 では 50）で、A2C の upd 275 より 3–5 倍速い。エポック再利用の恩恵である。
3. **移植の本質的な仕事は「μ の確率性の会計」だった。** クリップも KL も分散も、すべて「方策の平均がアンサンブル平均であり推定ノイズを持つ」ことへの補正を要した（周辺分散・ノイズ不感帯・KL のノイズ床）。この一連の補正は、NNN を任意の ratio ベース RL アルゴリズム（TRPO/SAC 等）へ載せる際に再利用できる一般的な処方である。
4. **限界の記録。** 単一 seed（多 seed 化が次課題）。σ_e 下限 0.2 は「終端の精密化を捨てて安定を取る」選択で、greedy eval には影響しないが、探索を絞り切る運用が要る場面では KL 目標や lr を var_t に連動させる精密化が要る。kl_target 0.02 は σ_μ 由来のノイズ床を含んだ測定値に対して効いており（実測 kl 0.06–0.12 でも健全）、床を差し引いた真の KL での制御は未実装。

### 23.11 SAC への NNN 統合の検証（2026-07-23 実施）＝ 実現可能：完全 NNN の off-policy SAC が swing-up + full balance を学習

§23.10 の処方 (i)–(v) が **off-policy・replay・twin-Q・max-entropy** という SAC の体制まで cov_jac を運べるかを検証した（多 seed 化に優先。ユーザ指定）。実装は `tmp/rl/sac.py`（設計と負の結果の理由をコメントで内包）、ドライバ `tmp/rl_sac_external_swingup.py`、結果は `tmp/out/sac_s0_v*.log`・`tmp/out/swingup_sac_s0.pt`。

**設計（likelihood-ratio SAC）。** (1) **Twin Q-critic** は入力 $[s,a]$ の NNN（`QNNN`、2 隠れ層 H=64）で、TD 目標 $y=r+\gamma(1-d)\,(\min(Q_1',Q_2')(s',a')-\alpha\log\pi(a'|s'))$（$a'$ は現在方策から新規サンプル、$Q'$ は polyak target）への cov_jac 回帰。各 Q が専用 EMA mirror + KP を持つ。(2) **actor** は SAC 標準の reparameterization 勾配が使えない（Q を貫く backprop が要る）ため、同じ目的関数の **score-function 形** $\nabla J=\mathbb E_{a\sim\pi}[\nabla\log\pi\cdot(\min Q-\alpha\log\pi-b)]$ を採り、$\nabla\log\pi$ は cov_jac がそのまま供給する。(3) cov_jac の再帰が top score に**線形**であることを利用し、per-sample 重みを score に前乗せしてミニバッチ全体を1回の $[B,T,H]$ forward で処理する（バッチ化。従来の per-step ループ比で大幅に高速：0.4 秒/episode）。

**§23.6-B2 の懸念「replay 下の mirror 推定」は解消と確認。** mirror は「現在の重み」を「現在の forward の T ゆらぎ」から測る量であり、replay データの古さは mirror に入らない（データ分布は測定点を選ぶだけ）。EMA + KP は変更なしで機能し、全 run を通じ Q の回帰は安定だった。

**5 回の設計反復（負の結果 4 つはすべて off-policy 固有で、診断済み）：**

1. **v1 = TD 発散（Qmin → −10¹⁴）。** A2C/PPO で有効だった running 目標標準化は、bootstrap 目標が「脱標準化した critic 自身」を通るため、スケールと目標が正帰還で増幅し合う。on-policy の回帰目標（GAE リターン＝実報酬に有界）では起きない、bootstrap 固有の破綻。修正：適応標準化を廃止し、**固定 reward scale + TD 目標の原理的クランプ**（$|y|\le r_{\max}/(1-\gamma)$、標準 SAC の流儀）。
2. **v2 = actor が学習しない。** 1状態1行動＋バッチ平均 baseline では、重み $w=Q(s,a)-\overline Q_{\rm batch}$ が行動の優劣でなく**状態価値の差**に支配される（GAE が自動でやっていた状態値除去が無い）。修正：**状態ごとに K 本の行動をサンプルし状態内平均を baseline に**（leave-one-out）。cov_jac の線形性から K 本分の score は1 forward の内部で合成でき、「状態内の行動サンプルと Q の共分散」という形になる。
3. **v3/v4 = fresh サンプル方式の SNR 限界。** ハングは脱出するが壁アトラクタで停滞（best 0.21–0.38）。antithetic 対（KDE スロープと同じ発想）や Q の 2 パス平均でも不足 — μ 近傍での Q の行動感度が評価ノイズ＋回帰残差に埋もれる。
4. **v5 = replay 行動への係留 + 処方 (ii) ＝ 決め手。** actor の学習信号を**実行済み replay 行動**（実際の遷移の帰結を Q が学習している）に変え、soft advantage $A=Q(s,a_{\rm replay})-\overline{Q(s,a_k)}_K$（K サンプルは baseline 専用）。off-policy 補正には収集時の $(\mu_{\rm old},\sigma_{\mu,\rm old})$ を replay に保存し、**§23.10 の noise-deadband 付き重要度比**をそのまま適用。ここで処方 (i)(ii)(iv)(v) がすべて SAC 内で稼働する（(iii) は fresh ミニバッチゆえ不要）。

**結果（v5、seed 0、1000 episodes ≈ 40 万 env steps）：**

| epi | mean cos | frac_up | last100_up |
|---|---|---|---|
| 500 | +0.70 | 0.07 | 0.000 |
| 750 | +0.82 | 0.24 | 0.000 |
| 925 | +0.84 | 0.48 | 0.000 |
| 950 | +0.55 | 0.61 | 0.723 |
| **1000** | **+0.83** | **0.91** | **1.000** |

mean cos は単調に上昇し（振り上げの着実な改善）、epi 950 でキャッチが出現、**epi 1000 で full balance（last100_up = 1.000）に到達**した。訓練リターンも +0.57 と SAC 系で初めて正の定常に達している。

**知見：**

1. **SAC への NNN 統合は実現可能である。** backprop・転置重み・reparameterization をどこにも使わず、off-policy・replay・twin-Q・エントロピー正則化の SAC が swing-up + full balance を学習した。§23.10 の処方は on-policy 固有ではなく、**「μ が確率的であることの会計」として ratio ベース RL 全般に持ち運べる**ことが確認された（とくに (ii) の noise-deadband 比が off-policy 補正の要）。
2. **off-policy 化の真の困難は mirror でなく「目標と信号の統計設計」だった。** 事前の懸念（mirror × replay）は空振りで、実際に壊れたのは (a) bootstrap × 適応標準化（v1）、(b) baseline の状態値汚染（v2）、(c) fresh サンプル評価の SNR（v3/v4）— いずれも NNN 固有というより「score-function 勾配で SAC を組む」こと自体の困難で、NNN の寄与（μ ノイズ）はそこに (i)(ii) の補正を要求する形で乗る。
3. **学習は成立するが PPO より遅く、収束の証明は未了。** PPO v4 が ~9 万 env steps で安定 1.000 に達したのに対し、SAC v5 は ~40 万 steps で最終 checkpoint が 1.000（キャッチ獲得は終盤）。late-half mean は 0.086 に留まり、**1.000 の持続性は未確認**（さらなる訓練での固化、または PPO のような安定化装置の追加が要る）。「NNN で SAC が可能か」は肯定、「SAC が PPO より有利か」はこのタスクでは否定的、が現時点の正直な評価である。
4. **限界の記録。** 単一 seed・単一タスク。α は固定（σ_e 固定のため entropy の自動調整は無意味 — 探索温度の学習は §22.2(c) のまま）。actor_mode="fresh"（v3/v4 の名残）はコード上残してあるが、標準は "replay" とする。

### 23.12 PPO と SAC の比較検討、および本章の総括（2026-07-23・一区切り）

#### 23.12.1 実測比較

同一タスク（swing-up + balance）・同一の完全 NNN 構成（cov_jac actor、学習される NNN critic/Q、EMA mirror + KP、backprop ゼロ）での比較：

| | PPO v4（§23.10） | SAC v5（§23.11） |
|---|---|---|
| full balance 初到達までの env steps | **~9 万**（upd 75） | ~40 万（epi 1000、最終 ckpt） |
| 総 env steps | 36 万 | 40 万 |
| 到達後の持続性 | **以後の全 ckpt で 1.000** | 未確認（到達直後に終了、late-half 0.086） |
| 学習曲線の形 | 立ち上がり後に一気に固定 | mean cos が単調にゆっくり上昇、キャッチは終盤に出現 |
| 実装反復（負の結果の数） | 3（凍結・定数崩壊・終端崩壊） | 4（TD 発散・baseline 汚染・SNR 不足・係留前） |

**このタスク・この実装では、サンプル効率を含む全面で PPO が勝った**（到達速度で約 4 倍）。

#### 23.12.2 なぜ一般的期待（SAC の方が高効率）が逆転したか

本質的理由は一つ：**SAC のサンプル効率の源泉である reparameterization 勾配（$\partial Q/\partial a$ を backprop で方策まで貫通させる低分散勾配 × replay 再利用 × 高い update/step 比）が、backprop フリー制約の下では使えない**。score-function 勾配への置換は分散を桁で悪化させ、v3/v4 の SNR 問題（Q の行動感度が評価ノイズに埋もれる）を直接引き起こした。SAC の看板の効率は、まさに NNN 制約が禁じる部品の上に載っていた。

副次的要因：(a) **credit の伝播経路** — PPO の GAE は軌跡に沿って報酬情報を1バッチで horizon 全体に運ぶが、SAC の actor は Q 経由でしか学べず、Q は 1-step TD で情報を1段ずつしか運ばない（SAC の単調で遅い学習曲線はこれと整合）。(b) **entropy 自動調整の不在** — σ_e 固定（§22.2(c)）のため、SAC のもう一つの強みである適応的探索が効かない。(c) **タスク相性** — エネルギー整形報酬つきの本タスクは on-policy 方策勾配に有利な部類で、SAC の優位が出やすい疎報酬・困難探索の設定ではない。

**公平性の留保**：単一 seed、SAC 側の調整は浅く（5 回の反復は設計修正でありチューニングではない）、SAC は update/step 比を上げてサンプル効率を計算時間で買うダイヤル（現状 rounds=32/episode は標準 SAC の 1 update/step 換算よりかなり控えめ）を残している。n-step 目標や σ 学習も未投入。この比較は確定的な優劣ではなく、**「backprop フリー制約下での構造的な相性」の測定**として読むべきである。

#### 23.12.3 結論：NNN-RL のアルゴリズム選択指針

「SAC が一般に PPO より効率的」という経験則は reparameterization を前提とした話であり、**backprop フリー制約はその前提を外す**。したがって NNN では、軌跡ベースの on-policy credit（GAE）＋比の安定化（clip/KL）という **PPO 系の機構がほぼ無傷で移植でき、構造的に相性が良い** — これが本検証の主結論である。SAC 系を選ぶ理由が生じるのは、疎報酬・困難探索で replay が本質的に効く場合か、探索温度の学習（σ の NNN 内部化）が実現した場合であり、その際も §23.11 v5 の「replay 行動係留 + noise-deadband 比」が出発点になる。SAC 側で差を詰める残り札は (i) update/step の増強、(ii) n-step 目標、(iii) σ 学習、だが、これらは可能性の検証を超えた最適化の領域として本章では踏み込まない。

#### 23.12.4 本章（§23 補遺）の総括 — RL プログラムの到達点

cov_jac は (a) on-policy policy gradient（§23.1）、(b) 価値回帰 critic（§23.9）、(c) PPO の clipped surrogate（§23.10）、(d) off-policy SAC の soft-Q actor-critic（§23.11）のすべてで、backprop・転置重みなしに実タスクを解く credit を供給した。必要だった追加はどれも新機構ではなく、**supervised で実証済みの mirror 設計（EMA + KP）の正しい移植**と、**「アンサンブル平均 μ の確率性」を各アルゴリズムの統計（分散・比・clip・KL・baseline）に正しく会計すること**だけである。あわせて、ノイズ場 option は ρ/h ゲート（§23.7）により深い層でも厳密なスキル分離・保護・合成を与える。

**本項目（swing-up を題材とした NNN-RL のアルゴリズム統合）はここで一区切りとする。** 残課題は次の4点に整理して引き継ぐ：(1) 多 seed 化（§23.9–23.11 はいずれも seed 0 のみ）と SAC の 1.000 持続性確認、(2) より難しい連続制御・§14.2 の行動タスクへの展開（「NNN RL が実用的に competitive」の検証）、(3) 探索温度の NNN 内部化（σ_e の撤廃・ρ 場による制御 — §21.6/§22.2(c) の未解決点）、(4) §19 の場 credit 経路の完全 forward 化（§22.2(d)）。

### 23.13 追試：ストッパ（壁）を使わない swing-up（2026-07-31〜08-01 実施）＝ 接触ゼロの振り上げ＋数秒保持まで成立、無期限保持は σ_e フロアが律速

**動機と発見**。PPO v4 の canonical 方策（§23.10）を greedy 評価で精査したところ、**1 エピソードあたり 64–88 step もストッパ（|x| = x_thr の壁）に接触**しており、頂点保持を壁にもたれて実現していた（max|x| = 4.00 張り付き）。v4 の「full balance」は壁アシストを含む達成だった、という重要な但し書きである。そこで「**最終方策が壁を一切使わない** swing-up + balance」を同じ PPO v4 機構（完全 NNN・backprop ゼロ）で達成できるかを検証した。実装は `tmp/rl/envs_swingup.py` の拡張（`wall_mode` end/stop、`x_barrier`、`alive_bonus`、`top_center`）、`tmp/rl/ppo.py` の拡張（同パラメータ・`fill_batch`・warm start・`lr_var_scale`・開始角カリキュラム）、ドライバ `tmp/rl_ppo_external_swingup.py`。**判定は常に厳格評価**（壁接触で即終了の env、下垂れから greedy、horizon 500、3 env seeds）で行った。

**設計反復の記録（9 構成、負の結果を含む）**：

1. **壁接触＝即終了（素朴）→ 壁死の学習**。整形報酬はぶら下がりで約 −2/step のため、**壁に当たって早期終了する方が生き続けるより高収益**になり、方策が壁への自殺を学習（ep_len が 34 に固着）。**生存ボーナス +2.2/step** で報酬順序を回復（終了＝将来報酬の放棄）。負報酬主体の整形と終了型ペナルティの組合せに固有の落とし穴。
2. **fresh 300 updates（終了壁＋生存ボーナス）**：壁接触 0 を維持して振り上げ・キャッチを獲得、**best last100_up = 0.647**。診断：キャッチが常に x≈+3（右端寄り）で起き、保持中に約 0.005/step で外側へドリフト → バリア領域（|x|>2.8）で崩れる。連続保持は最長 181 step。
3. **fresh 500 updates**（アニール伸長）：0.623。更新数では改善しない。
4. **warm start ＋ 頂点開始カリキュラム**（phase 2）：劣化（0.327、一部 ckpt で壁死復活）。
5. **warm start ＋ 中心化整形 top_center**（phase 2b、lr 半減・kl_target 0.01）：0.370、中盤崩壊。
6. **stop 壁＋重接触ペナルティ 2.5/step**（壁は物理的に存在、壁使用を罰する）：訓練中の壁接触は 0 に落ちるが、**greedy が壁域に平気で入り厳格評価で全滅**（survived 0/3）。stop 壁の訓練では「接触＝致命的」が学ばれず、平均的回避と厳格な不接触の間に汎化ギャップが残る。
7. **fresh ＋ top_center ＋ 頂点開始**（plan D、2 seeds、400 updates）：0.450 / 0.520。中心化は機能（保持中の max|x| が 3.8→2.0–2.9 に低下）したが、**保持はトラック中央でも 1–3 秒で壊れる** — ドリフトを直しても保持精度自体が律速と判明。
8. **推論時 T 増加（64→256）**：不変。μ のアンサンブル推定ノイズは律速でない（σ_μ 半減でも保持統計が同じ）。
9. **phase 3：σ_e 0.2→0.08 ＋ lr∝var_t**（実効ステップ一定化で §23.10 の終端崩壊を回避する狙い、2 基点）：**いずれも中盤で不安定化**（0.32）。本タスクでは **warm start からの継続 PPO が 4/4 で系統的に劣化**した（σ 再拡大・目的変更・単調 σ 減少のいずれでも）。

**最終結果（構成 2 の最終 ckpt、greedy 12 エピソード = torch 4 × env 3 seeds）**：**壁接触 0 回/12 エピソード**（全エピソード完走）、100 step（2 秒）以上の連続保持 9/12、**最長連続保持 228 step（4.6 秒）**、末尾 100 step 保持率 平均 0.52。エピソードは greedy でも T アンサンブルにより確率的で、末尾保持はどのキャッチが最後になるかの抽選になる：**完全成功エピソード（壁接触 0・末尾 100 step 完全保持・max|x| 3.07）は約 1/12 の頻度で出現**し（env seed 1 の 12 試行で 1 回、tail 中央値 0.52）、ラスター動画はその成功エピソードを記録したものである。

**結論と知見**：

- **成立**：backprop ゼロの単一 NNN（PPO v4）は、**ストッパに一切触れずに** swing-up・キャッチ・数秒スケールの頂点保持・再振り上げを行う方策を学習できる。壁アシスト（v4）に対し、これは正味の獲得である。
- **未達**：無期限の頂点保持（全エピソードで末尾 100 step 保持）。律速は **σ_e ≥ 0.2 の探索フロア下では微細なバランスゲインが学習されない**こと — §23.10 v4 が「終端の精密化を捨てて安定を取る」と明記した既知のトレードオフが、壁という受け皿を失って初めて顕在化した。σ_e を絞る素朴な解は score の発散（§23.10 v3）と warm-start 不安定性（本節 9）に阻まれる。
- **general 知見**：(a) 終了型ペナルティは報酬の符号構造と組でないと自殺アトラクタを作る、(b) stop 壁の訓練は厳格な不接触に汎化しない、(c) 保持の律速は推定ノイズでなく学習されたゲインの精度、(d) 本システムの warm-start 継続 PPO は不安定（新規 run が常に優った）。
- **次の一手（本命）**：これは §22.2(c)「探索温度の NNN 内部化」の実タスクからの再要請である。探索 σ_e と計算ノイズが分離され、頂点近傍でのみ探索温度を絞る **ρ 場による空間的・文脈的な温度制御**ができれば、微細ゲインの学習と大域探索が両立するはずで、§21.6 の「return 分散チャネル」設計の具体的なテストベッドになる。**〔追記（2026-08-02）〕この方向は §25 で実施され、本節の未達成目標は解決した**：σ_e を撤廃した内部ノイズ方策（readout ノイズ場 σ_out=0.35 一定）が、壁接触 0 のまま全 12 評価エピソードで末尾 100 step 保持を達成（§25.6）。ただし勝因は本節が仮説した「頂点の低温化」ではなく「一定の内部温度下でのロバスト化訓練」だった（同節知見 2）。

**成果物**：学習曲線 `tmp/out/ppo_nowall_curves_300upd.png`、獲得過程アニメ `tmp/out/rl_ppo_nowall_demo_300upd.gif`（カートポール＋訓練報酬カーソル＋エピソード内 cos 軌跡）、**actor（2層×128）/ critic（2層×64）の活動ラスター動画** `tmp/out/rl_ppo_nowall_raster_300upd.gif`・静止版 `ppo_nowall_raster_300upd.png`（ピーク時刻順ソート、挙動トレース同期。pumping 中の掃引的逐次活動と保持中の定常バンド、保持中の V 上昇が可視）。全 run の checkpoint・統計は `tmp/out/swingup_ppo_nowall_*.pt`・`ppo_nowall_*.log`。

---

## 24. 付録：試行手法一覧（説明と結果・2026-07-31 整理）

本レポートで実際に試した手法（対照・ablation・負の結果を含む）を、実施順に4群へ分けて一覧化する。判定の凡例：**成立** = 狙いを実測で達成、**否定** = 仮説が実測で棄却（それ自体が知見）、**部分的** = 動作するが目標未達、**対照** = 比較・ablation 用。詳細は各節を参照。

### 24.1 学習則の自然統合（forward-noise credit、CartPole バランス、§20）

| 手法（節） | 手法の説明 | 結果 | 判定 |
|---|---|---|---|
| **cov_jac forward-mirror credit・Step A**（§20.12） | policy score（logit 上の $a-p$）を top-level δ とし、forward 共分散から推定した weight mirror（`cov_weight`）と KDE crossing slope で層間へ再帰。転置重み backward を使わない。online（N=1、EMA なし単発 mirror）で autograd の $\nabla_W\log\pi$ との cosine を測定 | T=64 で cosine 0.92–0.98（H=16–256）。G1 通過。mirror 品質は T で改善し、H を上げると要求 T が増える | 成立 |
| **Step B 完全ループ**（§20.13） | cov_jac credit ＋ eligibility trace（γλ 減衰）＋ TD 誤差変調 ＋ EMA/KP online mirror ＋ 線形 TD critic（外付け・minimal）。SGD・観測正規化 | CartPole-v1 を greedy return 500（満点）まで学習。外部 RL アルゴリズム・backprop なしの最初の実証。Adam は trace と相性が悪く不安定、SGD が安定 | 成立 |
| **true-transpose oracle**（§20.12） | 同じ再帰で mirror を真の転置重みに置換した上界対照 | cosine 0.92–0.995（T 依存のみ、H 不依存）。残差は KDE slope の T 依存分＝再帰の実装は正しい | 対照 |
| **node perturbation（出力相関版）**（§20.13） | 各 unit 摂動を出力 logit に直接相関させる flat credit（mirror・再帰なし）。同じ trace/TD 骨格に credit 源のみ差し替え | 分散比 node/cov = 0.71–0.99 で cov_jac は分散で勝たず、学習曲線も重なる。§18-C の「分散優位」の主張は撤回（→ ablation と位置づけ直し） | 対照（優位は否定） |
| **backprop actor-critic**（§20.13） | 標準の backprop による上界対照 | 学習曲線は cov_jac・node_pert と重なり区別できない（2 seed・CartPole 規模） | 対照 |
| **SR sweep（Step C）**（§20.16） | ノイズ強度 σ を 0.05–2.0 に掃引し、mirror cosine（計算忠実度）・探索・到達 return の最適領域の重なりを検証 | 学習後重みでは最良制御の σ（0.6–1.3）と最良 credit の σ（0.1）が一致せず、「単一 σ が全役割を同時最大化」は不成立。ただし副次発見として **RL は低忠実度 credit に頑健**（cosine 0.44 でも return 500） | 否定（弱い主張は成立） |
| **Task #1: critic 統一（共有 body）**（§20.17） | 単一 NNN body に action/value 両 readout を載せ、value 誤差も同一の forward mirror 再帰で body へ流す（外部 scaffolding ゼロ） | 機構としては閉じ、value credit は共有 body を助ける（coef を下げると悪化）。ただし性能は外付け線形 critic に明確に劣る（peak 226–241 vs 318–413） | 部分的 |

### 24.2 ノイズ場を行動モード（option）とする検証（1D MultiModeReach、§21）

| 手法（節） | 手法の説明 | 結果 | 判定 |
|---|---|---|---|
| **固定 spatial 場で SR 対立の打破**（§21.1） | per-unit の固定場（uniform lo/mid/hi、split、graded）で CartPole を学習し、return–cosine 対立フロンティアを空間配分で抜けられるか検証 | split は劣位支配、graded は uniform_hi と同点。単一タスク・均質 readout では σ の空間配分は対立を解かない | 否定 |
| **disjoint recruitment 場による L1 addressing**（§21.2） | 隠れレジーム（±1 どちらのターゲットか観測に含めない）の 1D reach で、2つの disjoint 場 prototype を切替。episodic REINFORCE ＋ per-timestep baseline、credit は forward mirror | 場 P_0 固定で全軌道が −1.01±0.09、P_1 固定で +1.03±0.08 へ。**終点は場だけで決まる**＝同一重みのまま場が行動をアドレス（神経修飾場の L1 の RL 版） | 成立 |
| **報酬による場の自律選択**（§21.3） | softmax 選択器 `theta[context, field]` が文脈から場をエピソード毎に選択、本体と同時学習。prototype に意味は事前付与しない | 選択器が完全対角（ctx0→P0=1.00、ctx1→P1=1.00）に自己組織化し、両文脈とも正しいターゲットへ。本体は SGD なら安定（Adam は後期崩壊） | 成立 |
| **重なり場での多重化（損傷実験）**（§21.4） | 重なる2場（共有 26 unit、Jaccard 0.41）で 2 行動を学習後、unit 群ごとに readout 列をゼロ化して分割か多重化かを判定。単一隠れ層（σ-only ゲートが厳密に効く条件） | 共有群の損傷で**両行動が同時崩壊**（+0.94/+1.00）、専用群は片側のみ、random 対照は小＝多重化の署名（神経修飾場の L2 の RL 版）。副産物：2 層では σ-only recruitment が漏れる（→ §23.7 の ρ/h で解決） | 成立 |
| **連続場中心の報酬学習と補間**（§21.5） | prototype を与えず、連続の場中心 c∈[0,1]（Gaussian bump）を場レベル REINFORCE（$u_{\mathrm{mod}}=\Xi^{-1}\xi$、§19.3）で学習 | 対称初期（両中心 0.5）から報酬が対称性を破り μ_c=[0.97, 0.18] へ分化。学習後に c を掃引すると終点が滑らかに遷移し、**c≈0.55 で未学習の中間行動**＝連続 option 座標 | 成立 |
| **per-unit σ eligibility**（§10/§21.6） | $\psi_\sigma=g\,\phi_T'\,(-d/\sigma)$（教科書形）と forward 推定形で per-unit σ を policy-score から直接 credit | 独立 2 pass 間 cosine ≈ −0.04（ノイズ支配）、200 pass 平均の norm ~0：$\mathbb E[\partial\log\pi/\partial\sigma]\approx0$ で **ill-posed**。場は低次元 recruitment 座標として学習するのが正しいことを裏づけ | 否定（明確化） |

### 24.3 実タスク展開：CartPole swing-up + balance（§23–23.9）

| 手法（節） | 手法の説明 | 結果 | 判定 |
|---|---|---|---|
| **bang-bang episodic REINFORCE**（§23 無印） | 2 行動（±F）・cov_jac credit・per-timestep baseline・カリキュラム・エネルギー報酬・壁張り付きペナルティ | 本物のポンピングを獲得し頂点まで振り上げ（cos_max=1.0）るが、安定化せず（frac_upright≈0.2）。bang-bang（no-op なし）の構造的限界 | 部分的 |
| **連続力 NNN actor ＋ 外付け MLP critic（A2C+GAE）**（§23.1） | actor は連続 Gaussian 方策（score $(a-\mu)/\sigma^2$ を cov_jac が伝播）、critic のみ backprop MLP に割り切り。探索 σ_e は固定＋アニール | **last100_up = 1.00（full balance）達成**（upd350+）。cov_jac が backprop 相当の policy credit として実タスクを解くことの実証 | 成立 |
| **ノイズ場 option の multimodal actor（固定文脈ゲート）**（§23.2） | pump/balance の2場 prototype を文脈ゲート $g=\sigma(6\cos\theta)$ で連続ブレンド。重みは共有、credit は cov_jac のまま | upd275/300 で last100_up = 1.00。§21 の option 機構が実タスクで機能。方向1よりやや不安定 | 成立 |
| **学習ゲート（modulatory core）**（§23.3） | 固定ゲートを小さな NNN core（cov_jac）に置換し、ゲートを潜在行動として同じ advantage で学習（actor 完全 NNN） | タスクは解ける（last100_up=1.00）が、**ゲートは pump/balance に自律分化しない**（ほぼ定数に収束）。分化はタスクが複数モードを genuinely 要求するときのみ創発（§21.3 と整合） | 成立（分化は否定） |
| **スキル再利用・σ-only 凍結**（§23.4） | balance を場 A に事前学習 → A 凍結、pump を場 B に追加学習する二段継続学習（σ-only recruitment） | swing-up 部分的（0.38）、balance 保持が 1.00→0.34 に劣化。原因は σ-only ゲートの漏れ（＋不完全な凍結マスク）で、機構の限界ではない（→ §23.7 で解決） | 部分的（→解決済み） |
| **critic の NNN 化（単発 mirror）**（§23.5） | 外付け MLP critic を NNN critic（cov_jac で GAE リターン回帰）に置換し、システム全体を backprop ゼロに | 学習はする（振り上げ・時々保持）が last100_up ピーク 0.44 で full balance 未達。critic の価値回帰の質が律速 | 部分的 |
| **ρ/h ゲートによるスキル保護**（§23.7） | recruitment を σ-only から動員ダイヤル ρ（$\sigma=\rho\sigma_0, h=h_0/\rho$、off は h 番兵で厳密沈黙）に変更し §23.4 を再実行。olap 場（pump 時に凍結 A を read-only 語彙として全動員）も比較 | off 側 max mean z = 0.0000（厳密沈黙）、**保持 1.000（構成的）・drift 0.00e+00**。olap アームは swing-up 1.000 と保持 1.000 を両立。§23.4 の漏れは σ-only ゲートの限界だったと確定 | 成立 |
| **共有 body ＋ 線形 value head（A）**（§23.8） | value を actor 最終隠れ層の平均活動への ridge ヘッドにする（mirror 不要・value 誤差は body に流さない） | actor の更新が特徴を動かし続け value 適合が崩壊（R² 0.88→0.01）、方策も退化（best 0.18）。教訓：critic に要るのは「actor の表現」でなく「定常で十分豊かな基底」 | 否定 |
| **凍結ランダム NNN 特徴 ＋ ridge ヘッド（A′）**（§23.8） | critic を未訓練凍結の NNN 基底（H=256、マルチスケール行ゲイン）＋線形ヘッドに | value R² 0.93–0.99 で安定、EMA mirror（B）と併用で last100_up 1.000。ただし後に「凍結は不要」と判明（§23.9）し、mirror 誤差を消す最も乱暴な方法だったと読み替え。コードは削除済み | 成立（→読み替え） |
| **EMA mirror ＋ Kolen–Pollack の RL 移植（B / fix 1）**（§23.8–23.9） | per-step 単発の mirror 再推定を、supervised 側で実証済みの永続 EMA mirror（β=0.1）＋ KP 追跡に置換（actor・critic 両方） | **fix 1 のみで、特徴を学習する critic のまま last100_up = 1.000**。§23.5 の律速の正体は mirror 分散と確定。以後 a2c 系のデフォルトに採用 | 成立（決定打） |
| **ストッパ不使用 swing-up（PPO v4）**（§23.13） | 壁接触＝即終了＋生存ボーナス＋バリア報酬の env で PPO v4 を fresh 学習し、「壁を一切使わない」方策を狙う。9 構成の設計反復（壁死対策・stop 壁・warm start・中心化・σ 精密化・T 増加） | **壁接触 0 で振り上げ・キャッチ・最長 4.6 秒保持**（12/12 エピソード無接触、完全成功エピソードは ~1/12）。外付け σ_e の枠内では無期限保持が未達（σ_e≥0.2 フロアのゲイン精度が律速）。副産物：v4 の balance は壁アシストだったと判明、壁死アトラクタ・stop 壁の汎化ギャップ・warm-start 継続 PPO の系統的不安定（4/4）を記録。**→ §25.6 の内部ノイズ化で完全解決** | 部分的（→§25 で解決） |
| **探索温度の NNN 内部化（§25 Stage 1）**（§25.5–25.6） | 外付け σ_e を撤廃し、実行行動＝NNN の内部 readout サンプルそのもの。温度は readout ユニットのノイズ場 σ_out が制御（body σ 倍率は飽和して効かない）。絶対温度フロア（相対フロアだけでは §23.1 型崩壊が再現）＋ §23.10 の周辺分散会計。gated（cosθ で hot/cold）と const（一定）を比較 | **const（σ_out 0.35 一定）が壁なし swing-up を完全解決：12/12 エピソードで末尾 100 step 保持・壁接触 0・最長保持 461 step**（外付けベースラインは tail 平均 0.52・完全成功 0/12）。gated は 0.530 で敗北＝勝因は頂点低温化でなく一定内部温度下のロバスト化訓練。温度が測定可能な物理量になったこと自体が診断力を生んだ | 成立（完全達成） |

### 24.4 RL アルゴリズム統合：PPO・SAC（§23.10–23.12）

| 手法（節） | 手法の説明 | 結果 | 判定 |
|---|---|---|---|
| **PPO v1：素朴移植**（§23.10） | clipped surrogate ＋ エポック再利用を §23.9 構成にそのまま載せる | アンサンブル平均 μ の推定ノイズ σ_μ だけで比が揺れ、clip_frac≈0.5 で勾配が死に方策凍結 | 否定（負の結果） |
| **PPO v2：ノイズ会計**（§23.10） | (i) 周辺分散 var_t=σ_e²+σ_μ² で score/比を計算、(ii) clip 閾値にノイズ不感帯 $\epsilon_t=\epsilon+2\sigma_{\log r}$、(iii) 勾配の r をクランプ | 凍結解消・upd50 で full balance（A2C の約5倍のサンプル効率）だが、直後に定数行動アトラクタへ崩壊（trust region 踏み越え＋クランプ後行動保存のバグ） | 部分的（負の結果） |
| **PPO v3：KL 早期停止**（§23.10） | (iv) ノイズ床込み KL>0.02 でバッチの残エポック打ち切り、(v) クランプ前行動を score から復元 | upd75–250 の8連続 ckpt で 1.000 維持。ただし σ_e が 0.1 までアニールし切る終端で KL 暴騰・崩壊 | 部分的 |
| **PPO v4（canonical）**（§23.10） | v3 ＋ 探索 σ_e の下限 0.2 | **upd75–300 の10連続 ckpt すべて 1.000・崩壊なし**。best-ckpt 選択不要、value R² 0.89–0.92 で安定、A2C 比 3–5 倍速。振動問題は解消 | 成立（canonical） |
| **SAC v1：素朴移植**（§23.11） | twin-Q NNN critic（cov_jac TD 回帰）＋ score-function actor ＋ running 目標標準化 | bootstrap 目標が脱標準化した critic 自身を通り正帰還で TD 発散（Qmin→−10¹⁴）。修正＝固定 reward scale ＋ TD 目標クランプ | 否定（負の結果） |
| **SAC v2：バッチ平均 baseline**（§23.11） | 重み $w=Q(s,a)-\overline Q_{\rm batch}$ で actor 更新 | baseline が状態価値の差に支配され actor が学習しない。修正＝状態ごとに K 本サンプルの状態内 baseline（leave-one-out） | 否定（負の結果） |
| **SAC v3/v4：fresh サンプル方式**（§23.11） | 現在方策から新規サンプルした行動で actor 信号を作る（antithetic 対・Q の 2 パス平均も試行） | ハングは脱出するが壁アトラクタで停滞（best 0.21–0.38）。μ 近傍の Q 行動感度が評価ノイズ＋回帰残差に埋もれる SNR 限界 | 否定（負の結果） |
| **SAC v5：replay 行動係留 ＋ noise-deadband 比**（§23.11） | actor 信号を実行済み replay 行動の soft advantage $A=Q(s,a_{\rm replay})-\overline{Q(s,a_k)}_K$ に係留し、off-policy 補正は §23.10 処方 (ii) の noise-deadband 付き重要度比 | **epi1000（~40 万 steps）で last100_up = 1.000**。backprop・転置重み・reparameterization なしの SAC が実現可能と確認。ただし持続性未確認・PPO 比約 4 倍遅い | 成立（feasible） |
| **PPO vs SAC 比較**（§23.12） | 同一タスク・同一の完全 NNN 構成での対決 | PPO が全面で優位（到達 ~9 万 vs ~40 万 steps）。理由は SAC の効率の源泉 reparameterization が backprop フリー制約で禁じられるため。**NNN では PPO 系が構造的に有利**が主結論 | 結論 |
| **SAC 再訪 v6/v6.1（内部ノイズ＋実測分散＋α 自動＋温度レギュレータ）**（§25.8） | PPO を勝たせた §25 の全部品を SAC v5 に統合し、公平条件（エントロピー調整あり）で壁なし swing-up を再検証。4 アーム（fixa/auto/regfixa/regauto） | best 0.517/0.173/0.163/0.347 と全アームで PPO（1.000）に遠く及ばず。**新現象：SAC 更新下で内部温度が暴走**（0.35→0.9、α では制御不能＝価格は物理ハンドルでない）。**物理レギュレータ ρ_T は温度制御に成功**（0.32–0.37 維持、Stage 3 の setpoint 形）が性能は回復せず。PPO の勝因（実測分散）は信頼領域機構の中で効くと確定し、§23.12.3 の指針を公平条件で再確認 | 結論（PPO 優位再確認） |

**横断的な実装知見**（表の複数行に共通）: (1) forward-mirror 系は **SGD が堅く Adam は不安定**（§20.13/§21.3）。(2) **観測正規化が必須**（§20.13）。(3) mirror は**永続 EMA ＋ KP 追跡**が正しい設計で、per-step 単発推定は RL では律速になる（§23.9）。(4) NNN を ratio ベース RL に載せる際は「**アンサンブル平均 μ が確率的**」であることの統計的会計（周辺分散・ノイズ不感帯・KL ノイズ床）が本質的な移植作業（§23.10 処方 (i)–(v)、SAC でも再利用）。

---

## 25. 探索温度の NNN 内部化（2026-08-01 着手）

§22.2(c) で「場の唯一の未踏 DOF」とした探索温度の学習に、§23.13 の実測（壁なし保持は σ_e フロアが律速）を受けて着手する。本節はアイデアの定式化と段階計画の記録である。

### 25.1 問題：探索だけが外付けのまま残っている

現行の連続方策（§23.1 以降のすべての実タスク実験）は $a\sim\mathcal N(\mu,\sigma_e^2)$ で、$\mu$ は NNN のアンサンブル平均だが **$\sigma_e$ は手でアニールする外付けの大域スカラー**である。credit・方策・価値・option が NNN 内部で閉じた今、これが最後の外部 scaffolding であり、「計算に必要なノイズがそのまま探索を担う」（§2.2）という自然統合の中心主張が、行動レベルでは未実現だったことを意味する（注：この切り離しは §23.1 の実装上の割り切りとして導入されたが、以後の実験でも無自覚に維持されてきた）。

実用面の代償は §23.13 で顕在化した。大域一律の $\sigma_e$ は下限 0.2 を要し（下げると score $(a-\mu)/\sigma^2$ が発散、§23.10 v3）、その ±4N のノイズ床の下では頂点保持の微細ゲインが学習できない。「pumping では広い探索、頂点近傍では精密制御」という要求は、**大域スカラー温度では構造的に満たせない**。

### 25.2 設計空間の制約（確定済みの否定的結果より）

1. 行動ノイズを readout サンプル分散に**素朴に**直結すると、分散の縮小とともに score が発散し方策が inaction に崩壊する（§23.1）。→ 分散フロアの会計が必須。
2. per-unit σ の policy-score credit は ill-posed：$\mathbb E[\partial\log\pi/\partial\sigma]\approx0$（§21.6）。bulk の σ は動作点で平均方策を動かさず、分散＝探索幅だけを変えるため、尤度スコアには映らない（§17.2-2）。
3. 場の学習可能な自由度は per-unit 値でなく**低次元座標**（§21.5–21.6）。

したがって内部化は次の 3 点セットでのみ成立する：**(a) 温度の発生源を NNN の物理ノイズにする、(b) 温度を場の低次元座標で制御する、(c) 温度の学習信号は尤度スコアでなくリターンの統計（分散チャネル）から取る**。

### 25.3 構成

**(a) 発生源**：外付け Gaussian を撤廃し、実行行動を readout アンサンブルの実サンプル $a=o^{(m^*)}$（§2.1 の原形）とする。行動分散は物理量 $\sigma_a^2(s)=\mathrm{Var}_m(o^{(m)})$ となり、score・比・KL の会計は周辺分散 $\mathrm{var}_t=\sigma_a^2+\sigma_\mu^2$（$\sigma_\mu^2=\sigma_a^2/T$ は μ 推定ノイズ）で行う（§23.10 の処方の流用）。**〔Stage 1 v1 の実測による訂正〕**当初この周辺分散を「常に正のフロア」と述べたが、これは誤りだった：$\sigma_a^2(1+1/T)$ は spread とともに 0 に潰れる**相対**フロアであり、spread の小さい状態で score が発散して §23.1 の崩壊が再現した（実測：KL 1.46・clip 0.52・学習停滞）。正しくは**絶対温度フロア** $\mathrm{var}_{\min}$（v2 では std 0.1 相当）を要する。フロア未満の状態では実行行動に不足分のディザを加えて挙動分布とスコアモデルの整合を保つ。絶対フロアの存在は欠点でなく、決定論的崩壊を構造的に防ぐ特徴であり、物理的にも交差ノイズが完全には消えないことに対応する。

**(b) 制御**：場が per-unit ノイズを決めるので、**探索温度＝場が制御する物理量**になる。Stage 1 は §23.2 と同型の固定文脈ゲート $g=\sigma(k\cos\theta)$ で hot 場（pump 用、σ 倍率 1）と cold 場（balance 用、σ 倍率 ~0.3）をブレンドする。頂点に近づくほど行動温度が物理的に下がり、§23.13 の律速に直接答える。これは §20.16 の SR 対立（計算忠実度は低 σ、制御は中〜高 σ）への解でもある：§21.1 で一様タスクの空間配分は効かなかったが、ここでは**文脈（時間）軸**が加わるため配分が意味を持つ。スキル保護まで要る場合は σ 単独でなく動員ダイヤル $\rho_k$（$\sigma_k=\rho_k\sigma_0,\ h_k=h_0/\rho_k$。§23.7）に一般化し、**cold 場において pump 側ユニットを厳密沈黙させる**ことで pump ノイズから保持ゲインを隔離できる（§23.7 の ρ olap アームと同じ配置：pump 場では A・B 双方を動員し、balance 場でのみ B を厳密沈黙させる）。

**※ 交絡の注記（`docs/idea_core.md` §3.3・§4.6・§4.7）。** 本レポートの実測はすべて $h=0.15$ 固定の体制 (b) なので、cold 場で $\sigma_k$ を 0.3 倍すると $h_k/\sigma_k$ が約 3.3 倍になり、**行動温度だけでなく交差率 $\nu_k$（動員量）も同時に下がる**。この交絡は偶発的ではなく構造的である：$z_k$ は二値なので $\mathrm{Var}(z_k)=\nu_k(1-\nu_k)$ であり、行動分散は $\sigma_a^2=\sum_k W_{ok}^2\,\nu_k(1-\nu_k)+(\text{相関項})$ と書ける。$\nu_k\in[0,0.5]$ ではこれが $\nu_k$ の単調増加関数なので、**行動温度と動員量は同一の $\nu$ 場の単調な関数**であり、場 $(\sigma,h)$ の操作だけでは独立に動かせない。$(\log\sigma,\log h)$ 平面で $\nu$ を保つ方向はゲージ方向 $(+1,+1)$ だけだが、それは（$w,b$ も同時にスケールする限り）出力がビット単位で不変なので温度も変えない。$\nu$ を保ったまま温度だけを変えられるのは readout 側の $\theta$（$W_{ok}$ のゲイン）である。

したがって **Stage 1 の go/no-go は「文脈依存の温度制御」と「文脈依存の recruitment」を分離できない** ―― 両者は同一の介入の 2 つの言い換えになっている。最低限の対処は、(i) 各アームで $\sigma$ の設定値ではなく **$\nu_k$ を実測して報告**し、温度低下がどれだけの動員低下を伴ったかを明示すること、(ii) 分離が必要なら、$\nu$ を揃えたうえで readout ゲインだけを変える対照アームを置くこと、である。$\rho_k$ に一般化する場合は、この交絡は設計として意図されたもの（動員を落とすこと自体が目的）になるので、$\sigma$ 単独アームと $\rho$ アームは別条件として扱う。

**(c) 学習**：per-unit σ の尤度 credit は死んでいる（§21.6）が、**行動レベルの分散スコア**は well-posed：$\partial\log\pi/\partial\sigma_a=((a-\mu)^2-\sigma_a^2)/\sigma_a^3$。これを advantage で変調し、$\partial\sigma_a/\partial(\text{場座標})$（forward 推定または §19.5 型の解析 Jacobian）で低次元座標へ写す：
$$\Delta(\text{場座標})\propto A_t\cdot\frac{(a-\mu)^2-\sigma_a^2}{\sigma_a^2}\cdot\frac{\partial\sigma_a}{\partial(\text{場座標})}$$
直観は「予定より大きく外れた行動が良い結果を生んだら温度を上げ、悪ければ下げる」。§21.6 の ill-posed 性は、(i) スコアを σ_a が観測可能な行動レベルで取り、(ii) per-unit でなく低次元座標に写す、の 2 段で回避される。別経路として max-entropy 目的があり、σ_e 固定ゆえ無意味だった SAC の α 自動調整（§23.11）が「エントロピー＝物理ノイズ量」として意味を持ち直す（§23.12 残り札 (iii) と接続）。

**神経科学的対応**：Doya (2002)（§18-F）の「ノルアドレナリン＝探索温度」を NNN の場で実体化するものであり、ノイズ場＝神経修飾場という描像（`docs/idea_neuromod.md` §2.4）において、場が行動モード選択（§21–23）に加えて探索状態の制御まで担うことになる。神経修飾の主張を RL 側から最も強く支持する断面である。

### 25.4 段階計画

- **Stage 1（機構の証明・温度は固定ゲート）**：σ_e 撤廃＋実サンプル行動＋hot/cold σ 場の cosθ ブレンドを PPO v4 に載せ、壁なし swing-up（§23.13 設定）で保持が改善するかを判定。**go/no-go**：§23.13 ベースライン（tail 平均 0.52・完全成功 ~1/12）に対する改善。対照として温度一定（hot のみ）の internal-noise アームを併走し、「内部ノイズ化」と「文脈依存温度」の寄与を分離する。
- **Stage 2（ゲートの学習）**：固定ゲートを modulatory core（§9/§23.3）に置換し、場ゲートを潜在行動として学習。§23.3 の知見から、温度分化が「必要」なタスク（壁なし保持）でのみ分化が創発するはず。
- **Stage 3（温度レベルの学習）**：(c) の分散スコアまたはエントロピー目的で ρ レベル・場座標を報酬学習。§21.6 の否定を裏返す最後のピース。

リスク：分散スコアの高分散（バッチ平均・baseline 必須）、hot→cold 遷移点の方策不連続（§23.2 のブレンドで緩和）、cold 場での探索不足による局所解、実サンプル行動の重い裾（Gaussian 近似比の歪み）。

### 25.5 Stage 1 実施結果（2026-08-02、v1–v2）

実装：`tmp/rl/policy_cont.py`（`noise_mode="internal"`・絶対フロア＋不足分ディザ）、`tmp/rl/ppo.py`（`internal_noise`・`temp_fields` の per-step ゲート適用〔rollout・epoch 再 forward・評価〕・物理分散による score/比/KL 会計）、ドライバ `tmp/rl_ppo_itemp_swingup.py`。タスクは §23.13 の壁なし swing-up（終了壁・生存ボーナス・厳格評価）。アームは **gated**（hot σ×1 / cold σ×0.3 を cosθ ゲートでブレンド）と **const**（温度一定、内部化のみの ablation）、各 300 updates・seed 0。

- **v1 = 負の結果（§25.3(a) の訂正の由来）**。相対フロアのみでは spread の小さい状態で score が発散（KL 1.46・clip 0.52）し学習停滞。**絶対温度フロア**（std 0.1）＋実行行動への不足分ディザで解決（v2 は KL 0.02–0.08 で全区間安定）。
- **v2 = 機構は成立**。σ_e を完全撤廃した内部ノイズ PPO が壁回避（upd 100 で壁死 ~0）から振り上げまで学習。**探索温度が測定可能な物理量になった**（batch の $\sqrt{\overline{\mathrm{var}}}$ を毎 update 記録）。
- **性能は同予算のベースライン未満**：厳格評価 best last100_up は gated **0.387**（終盤単調上昇 0.207→0.337→0.387）、const **0.303**（変動）、外付け σ_e ベースライン 0.647。**gated > const** は方向として §25.3(b)（文脈依存温度）を支持するが単一 seed・小差。
- **診断（温度軌跡の実測が示した明確な原因）**：内部温度は訓練の大半で **0.10–0.20**（gated 終盤のみ 0.31）に留まり、ベースラインの σ_e スケジュール 0.4→0.2 に対して**大幅な探索不足**だった。readout のアンサンブル spread は body の σ（0.6）に対して自然には ~0.1–0.3 しか出ず、しかも学習が進むと縮む。つまり Stage 1 の敗因は内部化そのものではなく**温度の絶対レベルが低すぎた**ことにあり、これは場の設計変数（hot 倍率 > 1）で直接、あるいは Stage 3 の温度学習で自動的に、修正可能である。温度が観測・制御可能な物理量になったからこそ、この診断が 1 行のログで可能になった点は内部化の便益そのものである。

**Stage 1 の判定（v2 時点）**：機構（go）／性能（未達、原因特定済み）。次の一手は温度較正後の再比較。

### 25.6 Stage 1 較正 run（2026-08-02）＝ 完全達成：内部ノイズ方策が壁なし swing-up を全エピソードで解いた

**温度レバーの較正で第二の設計知見**。body σ の場倍率は行動温度にほぼ効かない（hot 倍率 1.5→3.0 で温度 0.23→0.24。crossing 活動が [0,1] に有界で readout spread が飽和する）。正しいレバーは **readout ユニット自身のノイズ場エントリ $\sigma_{\rm out}$**（$o^{(m)}\leftarrow o^{(m)}+\sigma_{\rm out}\xi^{(m)}$）である。readout も NNN の 1 ユニットでありその固有ノイズは他ユニットの σ と同じ物理量なので、これは外付けスケジュールの復活ではなく**ノイズ場の定義域の readout への拡張**であり、温度と body の計算（μ）が綺麗に分離する利点を持つ。実装 `policy_cont.sigma_out`・`ppo.temp_out`。プローブで温度 0.37（ベースライン σ_e 相当）を確認して再 run した（outgate: hot 0.35/cold 0.05 の cosθ ゲート、outconst: 0.35 一定。各 300 updates・seed 0）。

**結果 = outconst が完全達成**（厳格評価：接触即終了 env・greedy・下垂れから・3 env seeds）：

| アーム | best last100_up | late-half mean | 備考 |
|---|---|---|---|
| outconst（σ_out 0.35 一定） | **1.000**（upd 275・300 の連続 2 ckpt） | 0.592 | 12 エピソード統計検証（torch 4 × env 3 seeds）：**tail=1.0 が 12/12・壁接触 0・最長保持 461 step** |
| outgate（hot/cold ゲート） | 0.530（終盤単調上昇） | 0.329 | 壁接触 0 |
| （参考）外付け σ_e ベースライン §23.13 | 0.647 | ~0.36 | tail=1.0 は 0/12 |

**§23.13 の未達成目標（壁不使用の無期限保持）は、σ_e を完全撤廃した内部ノイズ方策によって統計的に完全解決された。** 振り上げ→キャッチ→エピソード終端までの保持（最長 461/500 step）が全 12 エピソードで再現し、壁接触は一切ない。

**知見（正直な整理）**：

1. **内部化は「機構として閉じた」だけでなく「性能で外付けを上回った」**（tail 平均 1.00 vs 0.52）。§25.1 の理論的動機（自然統合）が実利で裏づけられた。
2. **予想の逆転：勝ったのは文脈ゲートでなく一定温度**。§23.13 では「頂点近傍の低ノイズ化が精密ゲインに必要」と仮説したが、実際には**頂点でも σ_out=0.35 のノイズを浴びながら訓練した方策の方が頑健な保持を獲得**した（ノイズ下訓練＝ロバスト化。greedy 実行時はノイズが消えるので保持は容易になる）。cold ゲートはむしろ頂点での頑健化訓練を奪い 0.530 に留まった。「精密さは低温訓練から」ではなく「頑健さは高温訓練から」が正しい描像だった。
3. **帰属の確定（対照実験・2026-08-02）**：外付け σ_e=0.35 一定（カリキュラム同一・他条件同一）の対照は **best 0.370・late-half 0.286** に留まった。整理すると、外付けアニール 0.647 / 外付け一定 0.370 / **内部一定 1.000**。したがって勝因は「温度一定スケジュール」では**なく**（それ単独ではむしろ悪化）、内部ノイズ設計にある。**下位機構の分解（同日実施）**：(A) score・比・KL の分散を定数 0.37² に固定し実行はそのまま実サンプルとした **outfixv は 0.570 に劣化**、(B) 実行を「同じ per-state 実測分散の Gaussian ドロー」に置換し会計はそのままの **outgauss は 1.000 を維持**。すなわち**決定的な機構は (i) per-state 実測分散による統計会計**（状態適応的な実効温度・信頼領域）であり、(ii) 実サンプル実行の構造化成分は必須でない（(iii) μ 揺らぎは両勝者に共通で単独分離は未実施）。**核心の言い換え：NNN のアンサンブルは「状態ごとの分散計」であり、その実測分散で score・比・KL を正規化することが勝因**。外付け σ_e 設計はこの測定値を捨てていた。「μ の確率性の会計」（§23.10）の系譜の最終形である。
4. **多 seed 再現（2026-08-02 追記）**：outconst を seed 1・2 で再現実行した。best last100_up は **seed 0: 1.000（連続 2 ckpt、12/12 検証済み）／seed 1: 0.593／seed 2: 1.000（4 ckpt で 1.000、late-half 0.924）**。3 seed 中 2 つが完全達成で、最弱 seed 1 でも外付け一定対照（0.370）を大きく上回り、終盤上昇傾向。**内部ノイズの優位は seed を跨いで再現する**（完全達成は 2/3。全 seed 保証にはまだ届かず、seed 1 型の遅い獲得の解析は残課題）。

**成果物**：`tmp/out/swingup_itemp_outconst_s0.pt`（勝者 checkpoint）、獲得過程アニメ `tmp/out/rl_itemp_demo.gif`、actor/critic ラスター動画 `tmp/out/rl_itemp_raster.gif`、学習曲線 `tmp/out/itemp_outconst_curves.png`。

### 25.7 計画の改訂（2026-08-02）：Stage 2 の縮小・Stage 3 の SAC への統合

§25.6 の分解結果を受けて、§25.4 の段階計画を次のとおり改訂する。

- **Stage 2（ゲートの学習）は動機が低下したため縮小・保留**。理由：(i) 一定温度が文脈ゲートに勝った（cold 化はロバスト化訓練を奪う、§25.6 知見 2）、(ii) §23.3 の前例（場の分化はタスクが必要とするときのみ創発）から、学習ゲートは定数解に収束する見込みが高い。温度の文脈依存性が genuinely 必要なタスク（例：精密操作と大域探索が同一エピソードで交替する課題）が現れた時点で再訪する。
- **Stage 3（温度レベルの学習）は SAC の α 自動調整として実現するのが最も自然**。max-entropy 目的の温度パラメータ α の自動調整は「探索温度の報酬学習」そのものであり、§25 でエントロピー＝実測可能な物理量（per-state の実測分散）になったため、§23.11 で「σ_e 固定ゆえ無意味」だった α 調整が初めて意味を持つ。すなわち **SAC 再訪が Stage 3 の実行形態**である。
- **次段 = SAC 再訪**。§23.11 v5（replay 行動係留 + noise-deadband 比）を土台に、(i) 行動サンプリングを内部ノイズ化（σ_out 場）、(ii) score・重要度比・soft-Q 目標の log π 項をすべて per-state 実測分散で計算（replay に収集時の μ と実測 var を保存）、(iii) α 自動調整（目標エントロピーに対する標準の dual ascent）を導入する。§23.12 の比較で SAC が負けた副次的要因 (b)「entropy 自動調整の不在」がここで解消されるため、PPO との再比較は「backprop フリー制約下での構造的相性」の測定をより公平な条件で更新することになる。

---

### 25.8 SAC 再訪の実施結果（2026-08-02〜03）＝ 温度暴走の発見と物理レギュレータ、ただし PPO 優位は覆らず

§25.7 の計画どおり、SAC v5（§23.11：replay 係留＋deadband 比）に (i) 内部ノイズ（σ_out=0.35）、(ii) 全 logπ・score・比の per-state 実測分散化（replay に収集時 μ・σ_μ・var を保存）、(iii) α 自動調整（実測エントロピーへの dual ascent）を統合した **SAC v6** を、NNN-PPO 決定版と同じ壁なし swing-up（§23.13 env）で検証した。実装 `tmp/rl/sac.py`（`internal_noise`/`alpha_auto`/`temp_reg`）、ドライバ `tmp/rl_sac_itemp_swingup.py`、各アーム 600 episodes（≈24 万 env steps）・seed 0。

**結果（厳格評価 best last100_up / late-half）**：

| アーム | best | late-half | 備考 |
|---|---|---|---|
| v6 fixa（α=0.1 固定） | 0.517 | 0.127 | epi 300 がピーク、以後温度暴走で劣化 |
| v6 auto（α 自動） | 0.173 | 0.024 | 終盤 checkpoint は greedy でも壁死 |
| v6.1 regfixa（＋温度レギュレータ） | 0.163 | 0.108 | 温度は 0.32–0.37 に制御成功 |
| v6.1 regauto | 0.347 | 0.216 | 終盤上昇傾向はあるが低水準 |
| （参照）NNN-PPO 決定版 §26 | **1.000** | 1.000 | 同タスク・同程度の env steps |

**知見**：

1. **内部温度の暴走（新現象）**。SAC の更新下では実測温度が 0.35→0.9 へ単調成長し（actor 更新で重みが育ち ensemble spread が増える）、収集が壁死で崩壊する。PPO では温度が自然に ≤0.3 に収まっており（KL 制約が重み成長を抑えるためと推測）、この現象は SAC で初めて顕在化した。**温度が創発量であることの代償**であり、内部化の設計に「温度の閉ループ制御」が原理的に必要であることを示す。
2. **α は温度を制御できない**。α 自動調整は符号どおり動作した（実測エントロピー＞目標 → α→下限）が、**α はエントロピーの「価格」であって物理的ハンドルではない**。標準 SAC では σ_θ が方策パラメータなので価格が分散を動かすが、本設計の分散は重みからの創発量であり、価格をゼロにしても下がらない。
3. **物理レギュレータは機能する（Stage 3 の setpoint 形）**。大域ダイヤル ρ_T（body 場 × σ_out を同時スケール）を実測温度の目標 0.35 への log 比例制御で回すと、**温度は全区間 0.32–0.37 に制御された**（v6.1）。「温度制御は価格でなく場のダイヤルで行う」という §25 の主張が機構として動いた。ただし性能は回復せず（regfixa 0.163）、温度安定は必要条件であって十分条件ではない。
4. **PPO 優位（§23.12）は公平条件でも覆らない**。エントロピー調整の不在（§23.12.2(b)）を解消し、PPO を勝たせた実測分散会計を注入しても、SAC は 0.5 未満に留まる。**PPO の勝因（per-state 分散）は ratio/clip/KL という信頼領域機構の中で効く**ものであり、SAC の律速は依然として score-function Q 勾配の SNR（§23.11 v3/v4 の診断）にある、というのが整合的な解釈。§23.12.3 のアルゴリズム選択指針（backprop フリー制約下では PPO 系が構造的に有利）は、より公平な条件で再確認された。
5. 残り札：訓練予算の延長（SAC v5 は簡単なタスクでも 40 万 steps を要した）、n-step 目標、報酬駆動の温度目標（setpoint の一般化）。ただし §23.12 と同様、これらは可能性検証を超えた最適化の領域とし、本再訪はここで一区切りとする。

**成果物**：`tmp/out/swingup_sacit_{fixa,auto,regfixa,regauto}_s0.pt`・`sacit_*_curves.png`（温度・α・ρ_T の軌跡を含む）。

---

## 26. NNN-PPO 決定版アルゴリズム（2026-08-02 整理）

§20 から §25 までの全反復を経て確定した、**最新・正式版の NNN-PPO** の仕様。実装は `tmp/rl/policy_cont.py`（`noise_mode="internal"`）・`tmp/rl/critic.py`（`NNNCritic`）・`tmp/rl/credit.py`（`MirrorEMA`）・`tmp/rl/ppo.py`（`train_ppo_nnn(internal_noise=True, temp_out=(σ_out, σ_out))`）。壁なし swing-up での検証は §25.6（3 seed 中 2 で完全達成）。

### 26.1 構成要素

**方策（actor）**：
- NNN body：`SimpleNNNSample`、構造 [obs, 128, 128]、per-unit ノイズ σ=0.6、crossing 幅 h=0.15、内部サンプル数 T=64。
- 線形 readout ＋ **readout ノイズ場** $o^{(m)} \leftarrow o^{(m)} + \sigma_{\rm out}\,\xi^{(m)}$（σ_out = 0.35 一定。温度＝場の成分）。
- 方策平均 $\mu(s)$ = クリーン readout のアンサンブル平均。
- **per-state 分散（本質）**：$\mathrm{var}_t(s) = \max\!\big(\mathrm{Var}_m(o^{(m)})\,(1+1/T),\ \mathrm{var}_{\min}\big)$、絶対フロア var_min = 0.01（std 0.1）。
- 実行行動：実サンプル $a=o^{(m^*)}$（フロア未満では不足分ディザ）。※分解実験（§25.6-3）より $a\sim\mathcal N(\mu,\mathrm{var}_t)$ でも等価に機能する。決定版は NNN 的に自然な実サンプル実行を正とする。
- greedy 実行：$a=\mu$。

**価値（critic）**：独立の NNN（[obs, 64, 64]、同 σ/h/T）。線形 value readout のアンサンブル平均を $V$ とし、標準化 GAE リターンへの回帰誤差を top-level score として cov_jac で更新（バッチごとに複数エポック）。

**credit（両ネット共通・backprop ゼロ）**：cov_jac forward-mirror 再帰（`cov_weight` mirror × KDE crossing slope）。mirror は**永続 EMA（β=0.1）＋ Kolen–Pollack 追跡**（§23.9。per-step 単発推定は不可）。転置重みはどこにも使わない。

### 26.2 更新則（1 update）

1. **収集**：3 エピソード × horizon 400（壁死で短縮時は総 step 数が揃うまで追加収集）。各 step で $(s, a, \mu_{\rm old}, \mathrm{var}_t, \sigma_\mu)$ を保存。開始角カリキュラム（bottom 50% / 頂点近傍 20% / 一様 30%）。観測は RunningNorm で正規化。
2. **GAE**：γ=0.99、λ=0.95。advantage はバッチ標準化。リターンは running 標準化して critic 目標に。
3. **actor（最大 4 エポック、KL 早期停止）**：各サンプルで再 forward して $\mu_{\rm new}$ を取得し、
   - 比 $r=\exp[(-(a-\mu_{\rm new})^2+(a-\mu_{\rm old})^2)/(2\,\mathrm{var}_t)]$（**収集時の実測 var_t を使用**）
   - **noise-deadband clip**：閾値 $\epsilon_t=\epsilon+2\sigma_{\log r}$（$\sigma_{\log r}=\sigma_\mu|a-\mu_{\rm old}|/\mathrm{var}_t$）。advantage の符号方向に $r$ が $1\pm\epsilon_t$ を超えたサンプルは棄却
   - 勾配 = $\mathrm{clamp}(r,1\pm\epsilon_t)\cdot A_t\cdot$ cov_jac score $\big((a-\mu_{\rm new})/\mathrm{var}_t\big)$
   - **KL 早期停止**：$\overline{(\mu_{\rm new}-\mu_{\rm old})^2/(2\,\mathrm{var}_t)} > 0.02$ で残エポック打ち切り
   - 更新は Adam（lr 0.01）、mirror は更新量を KP 追跡
4. **critic（4 エポック）**：凍結した標準化リターンへの cov_jac 回帰（lr 0.02）。

### 26.3 系譜（各要素がどの失敗から来たか）

| 要素 | 由来 |
|---|---|
| cov_jac + EMA mirror + KP | §20 Step A–B、§23.9（単発 mirror は律速） |
| 周辺分散・deadband・KL ノイズ床 | §23.10 v1–v3（μ の確率性の会計） |
| σ_e 下限（→撤廃の動機） | §23.10 v4 / §23.13（壁なし保持の律速） |
| 実サンプル行動・内部ノイズ化 | §25 Stage 1（§2.1 の原形の復権） |
| 絶対温度フロア＋ディザ | §25.5 v1 の崩壊（相対フロアは不十分） |
| readout ノイズ場 σ_out | §25.6（body σ は飽和して温度に効かない） |
| **per-state 実測分散の統計会計** | §25.6 分解（outfixv 0.570 vs outgauss 1.000 ＝ 勝因の本体） |

### 26.4 NNN 固有部分と標準部分の切り分け

- **NNN 固有**：(i) 勾配供給が forward mirror（転置重み・backprop ゼロ）、(ii) 方策分布が物理的アンサンブル（外付け分布なし）、(iii) 温度がノイズ場の成分（σ_out）、(iv) **アンサンブルが状態ごとの分散計として機能し、その実測値が PPO の全統計（score・比・clip・KL）を正規化する**。標準 PPO には (iv) に相当する測定値が存在しない（決め打ちの σ しかない）点が、本アルゴリズムの実利的優位の源泉である（§25.6-3）。
- **標準流用**：GAE・advantage 標準化・clipped surrogate・KL 早期停止・Adam・観測/リターン正規化。

---
