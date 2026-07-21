# Noise-modulated Neural Network に基づく報酬変調型強化学習とノイズ場による行動モード形成

## 0. 概要（このレポートの読み方）

本レポートは、Noise-modulated Neural Network（NNN）を強化学習（RL）へ **自然に**（外付けの方策分布・探索スケジュール・転置重み backward・外部 RL アルゴリズムに頼らずに）統合する構想と、その最小実証の記録である。「性能で既存手法に勝つ」ことは目的ではない（§20.14）。核心は、RL の構成要素（方策・探索・局所感度・層間 credit・時間方向 credit・サブネットワーク動員）が NNN の同一ノイズ機構から立ち上がることを示すことにある。

**構成の層**（読む順序）:

- **Part I ― 構想（§1–16）**: 元の設計・理論。forward fluctuation を方策・探索・credit・eligibility に一貫利用し（§2–6）、ノイズ場を行動モード/option として用いる（§7–12）。統合アルゴリズムと実験計画、中心的主張（§13–16）。
- **Part II ― 評価と位置づけ（§17–19）**: 査読的フィードバック（§17）、既存研究との位置づけ（§18）、最大の未定義部だった「場生成パラメータへの credit 経路」の具体化（§19）。
- **Part III ― 実装と実測（§20–21）**: 学習則の自然統合を CartPole で検証（§20：Step A–C ＋ Task #1 の critic 統一）。ノイズ場を行動モードとして検証（§21：L1 addressing・報酬による自律選択・L2 多重化・連続場形成・per-unit σ 検証）。実装はすべて `tmp/`（`tmp/rl/` 共通、`tmp/rl_*` 検証）。
- **結論（§22）**: 到達点の総括と、**最も NNN-native な RL 方式**の同定と詳説。

**主要な到達点**（詳細は各節と §22）:

1. 転置重み backward・外部 critic・外部 RL アルゴリズムなしに、**forward fluctuation の credit だけで CartPole を学習**（Step A–B, §20.12–13；単一 NNN 統合版 Task #1, §20.17）。credit は autograd の $\nabla_W\log\pi$ を cosine ~0.95 で復元（§20.12）。
2. **単一のノイズ強度 $\sigma$ は計算・探索・制御を同時最大化しない**（SR 対立、§20.16）が、RL は低忠実度 credit に頑健。
3. **ノイズ場が行動をアドレスし**（L1, §21.2）、**報酬で場を自律選択**でき（§21.3）、**共有ユニットに 2 行動が多重化**され（L2, §21.4）、**連続場座標の補間で未学習の中間行動**が現れる（§21.5）。
4. **per-unit σ eligibility は policy-score では ill-posed**（$\mathbb E[\partial\log\pi/\partial\sigma]\approx0$）＝場は低次元 recruitment 座標として学習するのが正しい（§21.6）。

否定的結果（§20.14 の較正、§21.1 の SR 対立、§21.6 の σ ill-posed）も含めて正直に記録している。なお §1–16 は当初の構想であり、§17 以降の評価・実測で修正・限定された箇所は各節に明記した（例：§18-C の分散優位の主張は §20.14 で撤回）。

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

したがって、ノイズ場 $P$ を切り替えると、重み集合 $W$ を共有したまま異なるサブネットワークを動員できる。

$$
\pi_o(a\mid s)
=
\pi(a\mid s;W,P_o)
$$

ここで $P_o$ は行動モード $o$ に対応するノイズ場である。

重みが各モード内の技能を保持し、ノイズ場が現在利用する技能の組合せを選択する。

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

少数のユニットには常に小さな基礎ノイズを与える。この部分は常時動作し、観測と現在のfield stateから次のfield座標候補を生成する。

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

スケール族のnoise distributionでは、条件を整えることで、

$$
\frac{\partial\bar\phi}{\partial\sigma}
=
-\frac{d}{\sigma}\bar\phi'(d)
$$

を利用できる。

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
3. active-unit数、計算量、消費電力との対応を明確にできる。
4. field間の違いを絶対scaleではなく相対的な動員patternとして解釈できる。

field分化を示すためにoverlap penaltyを直接入れると、分化そのものが正則化の自明な帰結になる。主結果では、

- 固定noise budget
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
- overlap penaltyあり／なし
- 固定noise budgetあり／なし

を設ける。

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

このセクションは、既存実装（`nnn/credit.py`, `nnn/activation.py`, `nnn/noise_field.py`）と3論文プログラム（`draft_nce.md` / `draft_front_comp_neuro.md` / `draft_tcds.md`）に照らした査読的コメントである。構想の骨格そのものは支持できる。以下は「どこが実装済み資産で強く、どこに理論的な穴と設計上の緊張があるか」を切り分けるためのものである。

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

8. **「gating を避ける」の正直な位置づけ（§9）。** modulatory core は結局 $s,c\to\tilde c$ の学習写像であり、soft-gating であること自体は否めない。mixture-of-experts との **真の差** は「core が出力するのは per-expert 重みではなく低次元 field 座標であり、field は出力を混合せずノイズ（recruitment / gain）を変調する」点にある。「gating network がない」と主張するより、「**gating が neuromodulatory field の形をとる（出力混合でなくノイズ変調で作用する）**」と正直に述べる方が、`draft_front_comp_neuro.md` の主張（単一物理量が選択と計算を兼ねる）とも噛み合って強い。

### 17.3 実験計画への補足（§14 に足すべきもの）

- **RL 版 SR 曲線を第1段階に追加。** $\sigma$ を掃引し、(i) weight mirror の autograd 勾配に対する cosine similarity と (ii) 到達報酬が、同じ $\sigma$ 領域で最適化されるかを見る。§17.2-3 の二律背反の直接検証であり、`draft_front_comp_neuro.md` の SR 図（§4）の RL 版になる。`--model sample` と `--model analytic` の対照もそのまま持ち込める。
- **node perturbation（pooled reward covariance）との variance-vs-network-size 比較を主軸に。** Fiete–Seung 型の $O(N)$ 分散劣化に対し、層別 credit がどれだけ改善するかが最も説得力のある定量結果になる。これを §14.1 の評価項目の筆頭にすべき。
- **「重み固定・field だけ切替で行動様式が変わる」（§14.2 の強い主張）は front_comp の L1 と同一実験。** RL 文脈では「学習後に field を固定 vs 切替」で示せる。`draft_front_comp_neuro.md` と相互引用でき、証拠を共有できる。

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
- 差分: §7–9 の noise field は、Doya の「神経修飾物質が学習・探索のメタパラメータを制御する」という枠組みの **具体的な計算実体** とみなせる。field 座標が探索と動員を制御する低速内部状態である、という構成はこの系譜に真っ直ぐ乗る。`draft_front_comp_neuro.md` の「ノイズ場＝神経修飾場」と同じ位置づけを RL 側から補強できる。

### G. options / 階層強化学習
- 代表: Sutton, Precup & Singh (1999) options、Bacon, Harb & Precup (2017) option-critic、Vezhnevets et al. (2017) FeUdal、Eysenbach et al. (2018) DIAYN。
- 差分: 通常の option は **別パラメータの sub-policy $\pi_\omega$** を持つ。NNN-RL は option を、**共有重みを連続 noise field で addressing した点** として表現する（§7.3 の連続 field 座標）。したがって option 間は field 空間で連続に補間でき、未学習の中間 option が滑らかに現れる（§14.2 の主張）。これは離散 option の option-critic とは異なる「連続 option 埋め込み」であり、§17.2-5 で示した通り、field への credit を option-critic の intra-option / termination gradient として定式化すれば、この文献群と直接比較できる。

### H. context-dependent gating（継続学習）
- 代表: Masse et al. (2018 PNAS) context-dependent gating。
- 差分: `draft_front_comp_neuro.md` §5 で既述の通り、Masse らはランダム割当のゲート＋別途のシナプス安定化という二機構だが、NNN は単一物理量 $(\sigma,h)$ が選択と計算の両方を担う。RL 文脈では「同一物理量が option 選択と方策計算を兼ねる」として同じ対比が使える。

### 3論文プログラム内での位置づけ（相互引用の設計）

この RL 構想は、既存3論文の **第4の断面であると同時に、それらを統合する capstone** に当たる。

- `draft_nce.md`（forward 統計から credit を作る学習則）を **そのまま actor/critic の空間方向 credit に使う**（§3・§17.1-1）。
- `draft_front_comp_neuro.md`（noise field が共有重み上の複数方策を addressing する）を **option 機構として使う**（§7・§17.3）。
- `draft_tcds.md`（$\rho=(\sigma,h)$ を資源として制御する）を **noise budget として使う**（§11）。
- そして **報酬 $\Delta_t^R$ が、この3つに共通する変調子** になる。学習則の credit も、場による動員も、資源配分も、すべて同一の reward-modulated eligibility 原理で更新される（§10.3・§15）。

したがって論文戦略上は、RL を単独で立てるより「NCE の学習則と front_comp の addressing が報酬のもとで出会う地点」として、3論文へのポインタ付きで位置づけるのが最も強い。逆に言えば、RL 論文が成立するには §17.2 の未定義部（特に 5: field への credit 経路）を埋めることが前提になる。

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
- **決定的な差**: option-critic は両級を backprop で学習するが、NNN-RL は **両級とも forward-covariance 推定** で学習し、かつ option を別パラメータの sub-policy でなく **共有重み上の noise-field addressing** として実現する（§18-G、`draft_front_comp_neuro.md` の L1）。

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

ノイズ強度 $\sigma$ を掃引し、M1 の cosine（mirror 推定精度）と M2 の到達 return を**同一図に重ねる**。両者の最適 $\sigma$ 領域が重なるかで、探索ノイズと推定ノイズの二律背反の有無を直接見る。`--model sample`（機構）と `--model analytic`（平均場）の対照は `draft_front_comp_neuro.md` §4 の RL 版。これは第1段階の最も概念的に重要な補助実験。

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
3. **探索と option を同一ノイズが担うノイズ場（§7–19、最も NNN 固有の frontier）**。計算に必要なノイズ＝探索ノイズ＝サブネットワーク動員ノイズ、という三位一体を行動モード形成（Foraging/Avoidance/Sheltering）で示す。`draft_front_comp_neuro.md` の addressing と相互に支え合う。
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

**含意（Sub-B の前提が崩れる → 場の価値の置き直し）**: 「報酬が対立を抜ける場を選ぶ」という Sub-B の動機は成立しない（抜ける場が無い）。ただしこれはノイズ場方向を否定しない。ノイズ場本来の価値は「**共有重み上で異なる場が異なる行動を実体化する**」（§7.2 / §14.2 / front_comp の L1）であって、SR 対立の解消ではない。そしてそれを示すには **複数の行動が必要な環境**が要る ―― CartPole は単一行動なので原理的に場の価値を示せない。したがって場/option の検証は CartPole を離れ、最小の multi-mode 環境へ移すのが筋。

### 21.2 Sub-B: 報酬による prototype 選択（§7.2 の option 機構、Task #2）

**Sub-A の否定を受けて再設計**: 「報酬が SR 対立を抜ける場を選ぶ」という当初の動機は崩れた（CartPole に抜ける場が無い）。ノイズ場本来の問い ―― **共有重みのまま場を切り替えると行動様式が変わるか（§14.2 / front_comp L1 の RL 版）** ―― は、複数行動を要する環境でしか示せない。よって CartPole を離れ、最小の multi-mode 環境で「場が行動をアドレスする」ことを RL で示す。

**環境 `MultiModeReach`**（`tmp/rl/envs_multimode.py`）: 1 次元・2 ターゲット（±1）。各エピソードのレジーム（どちらを目指すか）は**観測に含めない**。したがって行動を選べるのは**ノイズ場だけ**で、場の役割が観測と冗長にならない。場は recruitment field（`field.recruit`：半分のユニットを σ、残り半分を 0＝分離サブネット動員、§7.1）で、P_0 と P_1 が共有重み上の別サブネットを担う。学習は **episodic REINFORCE ＋ per-timestep baseline ＋ advantage 白色化**、actor credit は forward weight-mirror（転置重みなし）。

**結果（`tmp/rl_multimode.py`、`tmp/out/rl_multimode.png`）= 成立**:

- 学習は収束（return −71→−5.9、ターゲット到達）。
- **決定的テスト（場を固定、レジームは隠れたまま、3000 ep）**: 場 P_0（target −1）→ 終点 x = **−1.01 ± 0.09**、場 P_1（target +1）→ 終点 x = **+1.03 ± 0.08**。**終点は場だけで決まり、隠れたレジームに依らない。** すなわち **同一重みのまま、ノイズ場を切り替えると行動（目指すターゲット）が切り替わる** ―― front_comp の L1 addressing を RL で実現した最小実証（§14.2）。軌道図 `rl_multimode.png` は、同一初期条件から P_0 固定で全軌道が −1 へ、P_1 固定で +1 へ収束することを示す。

**実装上の要点**: 当初の TD critic は全負の dense 報酬でベースラインが機能せず学習が崩壊した（missing-baseline 問題）。critic を捨て REINFORCE ＋ per-timestep baseline にして解決。credit 側（forward mirror）はそのまま。

**限定と次**: `recruit(quiet=0)` は左右で概ね **disjoint なパラメータ**を使うため、厳密には「多重化」でなく「分割」寄り（L1 addressing は成立、L2 の overlapping multiplexing は未検証）。次段は (i) **報酬による自律的な場選択**（規制信号 → 場、§9 の modulatory core）、(ii) **重なりを持つ場**での多重化（front_comp L2 の RL 版）、(iii) 場そのものを報酬で学習（σ eligibility、§10）。

### 21.3 自律的な場選択（§7.2・§9 の option 機構、Task #2 = 完了）

Sub-B は場を「与えれば」行動をアドレスすることを示した。ここでは場を**報酬で自律的に選ばせる**。選択器（softmax 選好 `theta[context, field]`）が文脈から場 prototype をエピソード毎に選び、行動本体は選ばれた場の下で x のみを見て動く。**prototype には意味を事前付与せず**（§7.2）、選択器と本体を**同時学習**する。実装 `tmp/rl/multimode_select.py` + `tmp/rl_multimode_select.py`。

**結果（5000 ep, `tmp/out/rl_multimode_select.png`）= 成立**:

- **選択器が自律的に分化**: 学習後 π(field|context) は完全な対角（ctx0→P0=1.00、ctx1→P1=1.00）。報酬だけで「文脈→場」の一貫した routing を獲得。
- **合成行動が正しい**: context 0（target −1）→ 選択 P0 → 終点 −0.96、context 1（target +1）→ 選択 P1 → 終点 +1.03。両文脈とも正しいターゲットへ。
- **2 文脈は異なる場へ**（縮退なし）。prototype の意味は与えていないので、この対応は**報酬による自己組織化**の結果である。

すなわち、報酬が「**文脈 → ノイズ場 → 行動**」の三段を自己組織化し、**ノイズ場が option として自律的に機能する**ことを RL で実証した（§7.2 の狙いの最小達成）。

**実装上の要点**: 選択器と本体の同時学習は後期に不安定化しうる。本体を Adam にすると ep3000 以降で片方の行動が崩壊（return −6→−22）。**本体を SGD にすると危険域を越えて安定**（選択器はいずれも正しく分化する）。CartPole での Adam-不安定と同型で、forward-mirror REINFORCE は SGD が堅い。

**到達点**: ノイズ場方向は、(§21.1) 単一タスクでは SR 対立を抜けない一方、(§21.2) 複数行動環境では場が行動をアドレスし、(§21.3) 報酬が場を option として自律選択できる、というところまで最小実証された。次段は front_comp の L2（overlapping 場での多重化・共有ユニット損傷）と、場そのものを報酬で形成する σ eligibility（§10、§19 の field credit 経路）。

### 21.4 重なり場での多重化 vs 分割（front_comp L2 の RL 版、Task #3 = 完了）

§21.2/21.3 は **disjoint** な recruitment 場だった（2 行動が別ユニット群＝分割寄り）。ここでは **重なりを持つ場**（`field.overlapping_pair`、recruit_frac=0.7、P_0 と P_1 が中央 26 ユニットを共有、Jaccard(active)=0.41）で 2 行動を学習し、ユニット群を損傷（actor readout 列をゼロ化）して、行動が分割されているか多重化されているかを直接調べる。実装 `tmp/rl_multimode_lesion.py`。

**設計上の落とし穴と修正（NNN の重要な性質）**: recruitment（σ=0→不活性）が厳密に効くのは、**入力が T 方向に一定な層だけ**である。2 隠れ層だと、最終層の σ=0 ユニットも上流の揺らぎから発火するため、名目上の「共有／片側」分類が実使用と一致せず損傷テストが交絡する（実際、最初の 2 層版は名目 shared 損傷が無影響という辻褄の合わない結果になった）。**単一隠れ層**にすると場が readout 直前のユニットを直接ゲートし、σ=0 ユニットが真に dead になる（検証済み: P_1 下で p0_only ユニットの mean|z|=0）。以下は単一隠れ層のクリーンな結果。

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

もし行動が**分割**されていれば、どのユニット群を壊しても高々一方しか劣化しないはず。共有群が両方を同時に劣化させることは、**区画分割仮説への直接の反証**であり、front_comp の L2（分割でなく多重化）を RL で実証したことになる。§21.1 の限定（disjoint 寄り）への回答にもなっている。

**ノイズ場 RL のまとめ（§21）**: (21.1) 単一タスクでは per-unit σ 配分は SR 対立を抜けない、(21.2) 複数行動では与えた場が行動をアドレスする（L1）、(21.3) 報酬が場を option として自律選択する、(21.4) 重なり場で 2 行動が共有サブネットに多重化される（L2）。front_comp の L1・L2 が RL 側で最小実証され、かつ報酬による自律選択まで到達した。残る本丸は場そのものを報酬で形成する σ eligibility（§10 / §19 の field credit 経路）。

### 21.5 場を報酬で形成：連続場中心の学習と補間（§7.3・§19、Task #4 = 完了）

これまでは場（prototype）を与えるか離散選択していた。ここでは **prototype を与えず、連続の場中心 $c\in[0,1]$（Gaussian recruitment bump、`field.bump`）を報酬で形成する**。場レベルの policy が文脈ごとの中心 $\mu_c[\text{ctx}]$ を持ち、各エピソードで $c=\mu_c+\xi$ を sample、bump(c) の下で本体が動く。報酬が場中心（場レベルの policy gradient、§19.3 の $u_{\mathrm{mod}}=\Xi^{-1}\xi$ に相当）と本体（forward-mirror REINFORCE）を同時に動かす。実装 `tmp/rl/multimode_field.py` + `tmp/rl_multimode_field.py`。単一隠れ層（§21.4 のクリーンなゲート）、本体 SGD。

**結果（5000 ep, `tmp/out/rl_multimode_field.png`）= 成立**:

- **対称初期から報酬が対称性を破り、2 つの連続場中心を自己組織化**（両中心 0.5 スタート → $\mu_c=[0.97, 0.18]$）。prototype に意味を与えていないので、この分離は報酬による自己組織化。
- 各中心で正しい行動：ctx0（target −1、中心 0.97）→ 終点 −1.05、ctx1（target +1、中心 0.18）→ +1.01。
- **連続 option の決定的テスト（補間）= 成立**: 学習後に場中心 $c$ を 0→1 に掃引すると、終点が **滑らかな sigmoid 状に +1 から −1 へ遷移**する（$c{=}0.55$ 付近で終点 ≈ 0 という**未学習の中間行動**が現れる）。すなわち場は離散スイッチでなく**連続 option 座標**であり、場空間で近い位置は部分的に重なる部分網を動員して近い行動を生む（§7.3 / §14.2）。

**正直な範囲**: これは §19 の「**場を連続潜在行動とみなし場レベルの REINFORCE で中心を動かす**」最小版（提案した手堅い入口）である。§10 の純粋な **per-unit σ eligibility**（$\psi_\sigma=g\,\phi_T'(-d/\sigma)$ を forward 統計から作り σ を直接 credit する）は、より NNN-native な次段として残る。ここで動かしたのはスカラーの場中心であって per-unit の σ を forward-covariance credit で更新してはいない。

**§21 の最終まとめ**: ノイズ場方向は最小実証を一通り達成した ―― (21.1) 単一タスクでは σ 空間配分は SR 対立を抜けない（否定）、(21.2) L1 addressing、(21.3) 報酬による場の自律選択、(21.4) L2 多重化（共有ユニット損傷）、(21.5) 場を報酬で連続形成＋補間で未学習の中間行動。front_comp の L1・L2 に加え、報酬による自律選択・連続場形成・option 補間まで RL 側で示せた。最も NNN-native な残件は per-unit σ eligibility（§10）と、§19 の field credit 経路を weight mirror で forward 推定する版。

### 21.6 per-unit σ eligibility の実装と検証（§10）= 明確化的な否定

§10 の per-unit ノイズ場 eligibility $\psi_\sigma$ を実装し（`credit.sigma_grad` = 教科書の $-d/\sigma$ 形、`credit.sigma_grad_forward` = crossing 自身のノイズから $\partial z/\partial\sigma$ を局所推定する一般形。後者は `kde_slope` と同じく転置重み不使用）、autograd の $\partial\log\pi/\partial\sigma$ に対して cosine 検証した（`tmp/rl_sigma_credit.py`）。

**検証結果 = policy-score の σ 勾配は ill-posed**:

- 単一 pass の $\partial\log\pi/\partial\sigma$ は norm ~60 と大きいが、**独立な 2 pass 間の cosine ≈ −0.04**。すなわち per-pass の σ 勾配は**ノイズ支配**で、安定な per-unit 構造を持たない（重み勾配が Step A で cosine ~0.95 と安定だったのと対照的）。
- **200 pass 平均で $\partial\log\pi/\partial\sigma$ の norm は ~0**（単一 pass の約 $10^5$ 分の 1）。つまり **$\mathbb E[\partial\log\pi/\partial\sigma]\approx 0$**。
- $-d/\sigma$ 形も forward 推定形も、この（ノイズないしゼロの）gold に対し cosine ≈ 0。

**解釈（構想にとって重要）**: fixed recruitment の下では、**per-unit のノイズ量 σ は行動尤度に系統的な効果を持たない**（σ は分散＝探索/変動を変えるだけで、平均方策 $\bar\phi$ をこの動作点では動かさない）。ゆえに policy-score から σ を credit する §10 の枠組みは、動作点近傍では系統信号がなく成立しない。これは §17.2-2 の指摘（$\psi_\sigma$ は探索共分散でなく平均応答の感度）を実測で強めたもの。

**これは §21.5 の選択を裏づける**: 場の有用な自由度は「per-unit の σ 量」ではなく「**どのユニットを動員するか（recruitment）**」であり、それを動かすのは場中心のような**低次元座標**である（§19）。場中心を動かすと動員される部分網が系統的に変わり（→ REINFORCE で学習可能、§21.5）、一方 fixed recruitment で per-unit σ を掃いても平均方策は変わらない（→ 勾配 ≈ 0）。したがって **場は §10 の per-unit σ でなく §19 の低次元 recruitment 座標として報酬学習するのが正しい**、という設計判断が実証された。

**残る可能性**: σ の効果は recruitment 境界（σ=0↔σ>0）でのみ系統的に現れうる（境界ユニットの σ を上げると動員が増える）。すなわち per-unit σ credit は bulk では ~0 でも境界で非零の可能性がある。ただし主要な場の学習は低次元座標で足りることが §21.5 で示されており、per-unit σ を forward-covariance で直接学習する路線は、少なくとも policy-score 経由では優先度が低い。σ の探索/変動への効果（探索温度の学習）は policy-score でなく return 分散を通じた別チャネルを要する（今後）。

---

## 22. 結論：到達点と最も NNN-native な RL

### 22.1 到達点の総括（二本柱）

本プログラムは、構想の二本柱をそれぞれ最小実証した。

- **学習則の自然統合**（§20）: online で forward mirror が policy gradient 方向を復元し（Step A, cosine ~0.95）、forward path 内の credit だけで CartPole を学習し（Step B）、単一 NNN・単一ノイズで policy+value+探索+credit を担う（Task #1）。ただし単一 $\sigma$ は計算忠実度と制御を同時最大化せず（SR 対立, §20.16）、RL は低忠実度 credit に頑健、という限定が付く。
- **ノイズ場による行動モード**（§21）: 与えた場が行動をアドレスし（L1）、報酬が場を option として自律選択し（§21.3）、重なり場では 2 行動が共有サブネットに多重化され（L2, §21.4）、連続場座標の補間で未学習の中間行動が現れる（§21.5）。per-unit σ でなく低次元 recruitment 座標が正しい学習対象である（§21.6）。

### 22.2 最も NNN-native な RL：ノイズ場を option 変数とする階層 RL

これまで試した方式のうち、**NNN からRLを考える必然性に最も富み、NNN の定義的機構に最も密に統合されている**のは、**ノイズ場（recruitment 場）を行動モード＝option 変数として用いる方式**である。とりわけその最密結合形は、**連続場座標を報酬で学習する（§19 / §21.5）** ことと、**重なり場での多重化（§21.4）** の組合せである。

**なぜこれが「必然性に最も富む」か。** 判定の基準は「RL の構造が NNN に *押し付けられて* いるか、NNN から *示唆されて* いるか」である。

- **forward-covariance credit（§3, §20）は RL→NNN の方向**である。出発点は標準 RL（policy gradient / actor-critic）で、「その勾配を NNN の forward ノイズで実装できるか」を問う。RL の構造は既存のまま、NNN は backprop の代替を提供する。実際これは backprop と同等以上でも以下でもなく（§20.14）、しかも $\sigma$ の二役に二律背反を抱える（§20.16）。有用な *配管* だが、NNN 固有の RL を生まない。
- **ノイズ場 option（§7, §21）は NNN→RL の方向**である。NNN の定義的機構は「ノイズ強度がどのサブネットワークを機能させるかを決める（確率共鳴／recruitment）」ことにある。ここから出発すると、自然に次を問うことになる：**「ノイズ場を制御可能な内部状態にしたら何が起きるか」**。答えは、場が下位行動の選択子になる ―― これはそのまま **options / 行動モードの階層 RL 構造**である。標準 RL では option を別パラメータの sub-policy として *後付け* する（option-critic 等）が、NNN では **同一重み集合の上に多重化された複数方策を、ノイズ場が addressing する**（§21.2, §21.4）。すなわち option 構造を発明する必要がなく、NNN のノイズ機構に *既に埋め込まれている*。

**この方式が密結合である具体的な理由**（実証と対応づけて）：

1. **機構が NNN 固有**。recruitment（ノイズがサブネットワークを機能ゲートする）は標準ネットには存在しない。σ=0 のユニットは（入力が T 方向に一定な層で）真に不活性化する（§21.4 で検証）。この物理量がそのまま option 選択子になる。
2. **多重化 vs 分割を実証**（§21.4, L2）。重なり場で 2 行動を学習し、**共有ユニットを損傷すると両行動が同時に崩壊**、片側専用ユニットは片側のみ崩壊。これは「複数方策が共有サブネットに多重化されている」ことの直接証拠であり、front_comp の L2 を RL で示したもの。標準の option 分割では出得ない署名である。
3. **場は連続の内部状態**（§21.5）。報酬が場中心を動かし（場レベルの policy gradient, §19.3）、対称初期から 2 つの連続場座標を自己組織化する。学習後に座標を補間すると終点が滑らかに遷移し（$c\approx0.55$ で中間行動）、場は離散スイッチでなく **連続 option 埋め込み** である。連続的な行動モード間の滑らかな遷移という、NNN の graded recruitment に固有の性質が RL の option 補間として現れる。
4. **報酬による自律選択**（§21.3）。文脈から場を選ぶ選択子が、意味を事前付与していない prototype に対し、報酬だけで一貫した「文脈→場→行動」対応を自己組織化する（選択器 π(field|context) が完全な対角に収束）。option の獲得が報酬から創発する。
5. **同じノイズが探索も担う**。行動 sample のばらつきは NNN 内部ノイズそのもので、探索を外付けしない。場は「どのサブネットワークで探索するか」まで規定する。
6. **§21.6 が答えを鋭くする**。per-unit の σ 量は policy-score に系統効果を持たず（$\mathbb E[\partial\log\pi/\partial\sigma]\approx0$）学習対象として ill-posed。有用な自由度は「どのユニットを動員するか＝recruitment」であり、それを動かすのは **低次元の場座標**である。つまり NNN-native な RL 変数は raw な per-unit ノイズ強度ではなく、**recruitment 場の低次元座標**だと実測が特定した。これは option 変数としての場という描像を、さらに具体化・限定する。

**神経科学的な含意**。この方式は、`draft_front_comp_neuro.md` の「ノイズ場＝神経修飾場」という描像の RL 版に当たる。神経修飾物質が回路の実効的参加（ゲイン／興奮性）を変えて行動モードを切り替える、という計算原理が、ここでは「報酬で学習・選択される低次元ノイズ場が、共有重み上の多重化方策を option として addressing する」という形で実装される。option 変数が抽象的な離散 index ではなく **物理量（ノイズ場）** である点が、NNN 起点の RL としての特異性である。

**位置づけの要約**。forward-credit（§20）は、場も重みも forward-native に学習可能にする **不可欠な配管**であり、実際 §21 の場学習もこの credit の上に乗っている。しかし **NNN でなければ生まれない RL 構造**、すなわち「thinking RL from NNN の必然性」が最も濃いのは、**ノイズ場を連続 option 座標として報酬学習し、共有重み上の多重化方策を addressing する** 方式である。

**限定と今後**。(a) 実証はいずれも最小環境（CartPole／1D 2-target）であり、規模・行動数を上げた検証（§14.2 の Foraging/Avoidance/Sheltering）が要る。(b) L2 は重なり場だが、学習された多重化が recruitment 境界にどこまで依存するかの解析が残る。(c) 場の唯一の未踏 DOF は「探索温度としての $\sigma$」で、これは policy-score でなく **return 分散を通じた別チャネル**を要する（§21.6）。(d) §19 の場-credit 経路を weight mirror で完全に forward 推定する版（場中心すら autograd を使わない）は、自然統合を実装レベルで最後まで閉じる課題として残る。

---

## 23. 補遺：CartPole 振り上げ安定化への挑戦（部分的成功・2026-07-20）

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

- **連続力 NNN 方策**（§3.1 の自然形）：readout 平均 $\mu$ を NNN が出し、行動 $a\sim\mathcal N(\mu,\sigma^2)$、力 = $F_{\max}\cdot a$。スコア $u=(a-\mu)/\sigma^2$ を **cov_jac** が body へ伝播（NNN の貢献）。
- **外付け critic**：標準 MLP を backprop で GAE リターンに回帰。内製 critic の破綻を排除（これが唯一の統合上の割り切り）。
- **GAE(γ=0.99, λ=0.95)** ＋ advantage 正規化 ＋ 下垂れ寄りカリキュラム。
- **探索 σ は固定＋アニール**（0.4→0.1）。サンプル分散に σ を結ぶと $(a-\mu)/\sigma^2$ が発散して inaction へ崩壊するため、探索だけ外付けの固定 Gaussian にした（§21.6 の σ ill-posed と整合する設計判断）。

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

### 23.4 スキル再利用の試み（balance を先に学習 → swing-up で再利用）― 漏れ recruitment の限界

ノイズ場 option の**本来価値＝共有重み上でのスキルの合成・保護**（§7.1 / `draft_tcds.md`）を実タスクで示すべく、二段の継続学習を試みた（ユーザ提案）。Phase 1: 場を $P_\text{balance}$（サブネットワーク A=units[0:64]）に固定し **balance を A に事前学習**。Phase 2: **A を凍結**し、場ゲート（下=$P_\text{pump}$（B=[64:128])、頂点近く=$P_\text{balance}$（A））で swing-up を下垂れから学習、**pump を B に追加**しつつ **balance は凍結 A を再利用**する。実装は `tmp/rl/a2c_swingup.py`（`freeze_mask`・`fixed_field`・`norm_obj`・`energy_reward`）＋ `tmp/_consolidate_p1.py`/`_p2.py`。

**結果 = 部分的成立、しかしクリーンでない**:

- **Phase 1 成功**：A が balance を学習（現実的な handoff 開始 θ∈[0.1,0.3] から **last100_up = 1.00**）。純 cos 報酬・2 層で収束（64 ユニットの A では aggressive な θ̇=±1 摂動には弱いが、pump が届ける穏やかな状態は保持できる）。
- **Phase 2**：**A の重みは凍結（不変を検証）**、pump を B に学習。swing-up は **部分的**（last100_up ≈ 0.38）。
- **再利用はクリーンでない（漏れ recruitment）**：Phase 2 の方策の頂点 balance は last100_up ≈ 0.34 で、**A 単体の 1.00 から劣化**。原因は §21.4 の知見どおり **2 層では recruitment が漏れる**こと：balance フェーズ（$P_\text{balance}$）でも layer-1 の B ユニットが上流から発火し、**Phase 2 で pump 用に学習した readout-B 列がそれを読んで balance を汚染**する。損傷実験も分離しない（A/B いずれの ablation も両行動を劣化）。

**含意（正直な限界）**: ノイズ場 recruitment の**深い層での漏れ**が、**層をまたぐモジュラーなスキル合成・保護をクリーンには許さない**。厳密な分離には (i) 単一隠れ層（漏れ無し・しかし balance を学習しにくい、§23.4 診断）、または (ii) 場（ノイズ）だけでなく **readout も gate する**（純粋なノイズ場 recruitment を超える明示的 gating）が要る。すなわち「balance を保護しつつ pump を追加」というスキル保護は、現行の**ノイズ場 recruitment（漏れあり）単独では部分的にしか実現しない**。これは §21.1（不要なら場は分化しない）・§21.4（recruitment は入力が T 一定な層でのみ厳密）と一貫する、機構の適用限界の明確化である。

**見通し（重要）: この漏れは sample-level モデル `SimpleNNNSample` に固有であり、`SimpleNNNStatistic` / `SimpleNNNAnalytic` を使えばクリーンに解決するはずである。** 理由は forward の構造にある。`SimpleNNNSample` は per-sample の揺らぎ $[N,T,H]$ を層をまたいで伝播するため、深い層の $\sigma=0$ ユニットも**上流の揺らぎ**を入力に受けて閾値交差し、発火してしまう（漏れ）。一方 `SimpleNNNStatistic` / `SimpleNNNAnalytic` は各層の応答を「**その層で加えたノイズに対する期待／統計値**」として計算し、層への入力は前層の**決定論的な期待活性**（T 揺らぎを伝播しない）である。$\sigma=0$ なら（radius/幅 = 0、あるいは CDF が step 化して $2P(1-P)=0$）ノイズ由来の交差が一切生じず、**出力も局所微分も厳密に 0**、深さに依らず dead になる（`activation.py`: 「radius = 0 makes both the output and the derivative exactly zero」）。したがって **§23.4 の漏れ限界は sample モデル固有**であり、statistic/analytic 系では層をまたぐスキルの分離・保護（§23.4 の consolidation）は問題なく実現するはずである。

**ただし役割分担の注意**: `cov_jac` の weight mirror（`cov_weight`）は $T$ sample の共分散を要するため **sample モデル専用**である。statistic/analytic 系ではクリーンな recruitment が得られる代わりに、credit は解析的な局所微分 $\phi_T'$（`phi_prime`）経由で構成する必要がある。すなわち理想的には「**場によるクリーンな分離＝statistic/analytic、勾配＝解析形**」という組合せで、漏れなしのスキル保護 ＋ forward-only credit を両立できる見込みである。

**到達点**: 方向3 の swing-up は、(3a) 固定文脈ゲートの場 option で full balance（§23.2）、(3b) 学習ゲート＋完全 NNN actor で full balance（§23.3、ただし自律分化はしない）まで示せた。スキル保護（§23.4）は漏れ recruitment の限界により部分的。クリーンなスキル保護は「readout も gate する」拡張か非漏れ機構が必要、という次の課題を残す。

### 23.5 critic の NNN 化：単一 NNN（backprop ゼロ）で解く ― 学習するが critic が律速

当初計画の締めくくりとして、外付け MLP critic を **NNN critic（cov_jac、GAE リターンへの回帰）**に置換し、**actor・critic とも forward-mirror、転置重み backward をどこにも使わない単一 NNN システム**にした（ユーザ要望「全体を1つのネットワークとして見る」）。実装 `tmp/rl/critic.py`（`NNNCritic`）＋ `tmp/rl/a2c_nnncritic.py`。value 誤差 $(V-\text{GAE リターン})$ を top-level score として actor と同じ cov_jac 再帰で回帰し、リターンは running 標準化。

**結果（`tmp/out/swingup_nnnac.pt`、eval-from-bottom）= 学習するが full balance 未達**:

- 完全 NNN actor-critic は swing-up を**学習する**（mean cos −0.81 → +0.2〜0.29、last100_up はピーク ≈ 0.44）。**backprop を一切使わずポールを頂点まで振り上げ、時々保持する**。
- しかし **full balance（last100_up = 1.0）には届かない**（方向1の外付け MLP critic は 1.0 に到達）。critic が律速。

**診断（正直な限界）**: cov_jac の value 回帰は、backprop MLP critic より **advantage の質が低い**。理由は (i) forward-mirror 推定が backprop より高分散、(ii) バッチ 1 パス更新（MLP は 8 epoch backprop）で critic の学習が遅い、(iii) bootstrap する GAE リターン（critic 自身に依存）を高分散 critic で回帰するため誤差が乗る。これは本取り組みを通じた一貫観察 ―― **actor 側の policy credit は cov_jac が backprop 相当に機能する（§23.1 で full balance 達成）が、critic（価値回帰）は cov_jac だと質が落ち、full balance を支えきれない** ―― を追認する。

**含意（主張の較正）**: 「NNN cov_jac が backprop 相当」という主張は **actor（policy gradient）については実タスクで検証された**（§23.1）。一方、**完全 backprop フリー（critic も NNN）にすると、価値関数回帰の質が律速となり full balance は未達で partial に留まる**。したがって現時点の堅い到達点は「**cov_jac actor ＋（信頼できる）critic で full balance を解く**」であり、critic まで NNN 化した単一システムは「解きつつあるが critic 強化が課題」である。

**critic 強化の道**: (i) critic の cov_jac 更新を複数 epoch（各 epoch で再 forward、低速）にする、(ii) `cov_jac_full`（readout 誤差も forward 統計から）や pooled mirror で分散を下げる、(iii) critic を actor と body 共有（§6/§20.17）にして表現を強化、(iv) §23.4 の見通しに沿い statistic/analytic 系で漏れなく安定化。いずれも次段の課題。

### 23.6 次段の優先課題（作業再開時のメモ）

本セッションはここで一区切り（主目標＝NNN cov_jac で swing-up + full balance を解く、は §23.1/3a/3b で達成）。再開時の優先順位は以下。

**最優先 (A): NNN critic を強化し、完全 backprop フリー版（§23.5）を full balance へ届かせる。**
現状、actor（cov_jac）は backprop 相当だが、critic まで NNN 化すると価値回帰の質が律速で last100_up がピーク 0.44 に留まる。原因に直接効く候補（推奨順）:
1. critic の cov_jac 更新を **複数 epoch**（各 epoch で再 forward。低速だが MLP の 8-epoch backprop に相当する学習量を与える）。
2. **`cov_jac_full`**（readout 誤差も forward 統計から）や **pooled mirror**（`cov_weight(..., pool=True)`）で mirror 分散を下げる。
3. critic を actor と **body 共有**（§6 / §20.17 の統合 critic）にして表現を強化。
4. §23.4 の見通しに沿い **statistic/analytic 系**で漏れなく安定化（credit は解析 $\phi_T'$ 経由）。
目標: last100_up = 1.0 を **backprop ゼロ**で達成し、「単一 NNN で難タスクを解く」を完成させる。

**次点 (B): 素朴な A2C でなく、同系のより進んだ RL アルゴリズムを NNN（cov_jac）へ統合する。**
現状は on-policy A2C＋GAE の素朴構成。cov_jac が $\nabla_W\log\pi$ を forward-only で与えるので、より進んだ policy-gradient 系は自然に載る:
1. **PPO（最有力・直接の発展）**: clipped surrogate ＋ 複数 epoch の再利用。重要度比 $\pi_\text{new}/\pi_\text{old}$ のクリッピングで安定化・sample 効率向上。既存の GAE ＋ 連続 Gaussian-from-samples 方策（§3.1）にそのまま乗る。cov_jac は ratio の対数勾配 $\nabla\log\pi$ を供給。
2. **SAC（より野心的・off-policy）**: max-entropy ＋ replay ＋ twin critics。**entropy 項は NNN の内部ノイズ（探索）と概念的に直結**し、NNN 固有の魅力がある。ただし off-policy と cov_jac の整合（mirror は方策非依存だが score は importance 補正が要る）と replay 下の mirror 推定は要検討。
目的: swing-up+balance を超え、より難しい連続制御へ NNN RL を広げ、「NNN RL が実用的に competitive」を強める。cov_jac actor はこれらの上位アルゴリズムの policy-gradient 部品として差し替え可能な位置にある。

**実装メモ**: 完全 NNN 版は `tmp/rl/a2c_nnncritic.py`（`critic.NNNCritic`）。critic 強化は critic 更新ループの epoch 化から。PPO 化は `train_a2c` に old-policy log-prob 保存＋ratio クリップを足すのが最小変更。
