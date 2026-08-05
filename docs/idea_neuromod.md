# 神経修飾場の研究ノート — 修飾物質を「何に」結びつけるか

**この文書の役割**: NNN のノイズ場 $\mathcal P=\{(\sigma_k,h_k)\}$ を神経修飾場のモデルとして扱う
研究線の**単一の文書**。主張の骨格と限定（第 1 節）、修飾物質を計算論的に何に結びつけるのが
生物規範的かの検討（第 2〜5 節）、それを支える実測（第 6 節）、そこから導かれるデモ案（第 7 節）を置く。
投稿想定は Frontiers in Computational Neuroscience の Research Topic
「Computational Models of Neuromodulation」。記事種別は **Hypothesis & Theory**（次点：Methods /
Technology & Code）。Original Research は避ける。主張は機構仮説とその最小実証であり、
新規の神経データ解析ではないため。

> **投稿情報（※メモ由来・要再確認）**: IF 3.3、締切 2026-09-11、編集者に Tomoki Kurikawa
> （公立はこだて未来大学）ほか。投稿前に Research Topic ページで最新の締切・スコープ・
> 記事種別を確認すること。

**正典**: 記号・定義は `docs/idea_core.md` に従う。特に効くのは §2.6（SR は sample 水準にしか
存在しない）、§3.3–3.5（交差率 $\nu$、$\sigma=0$ リーク、kill 三点操作）、§4（ゲージ対称性）。

**関連文書**: `idea_consolidation.md`（$\rho$ による資源制御、圧縮・逐次格納）、
`idea_rl.md` §25（探索温度の内部化）、`draft_nce.md`（forward-only 学習則）、
`idea_coding.md`（同じ SR 論法の時間軸版）。
**実装の配置**: 標準問題の共通部は **`tmp/neuromod/`** パッケージ（`world` 問題設定・感覚符号化・
行動目標・閉ループ駆動 / `fields` ノイズ場と交差率による参加度 / `protocol` 学習と採点 /
`viz` 描画・アニメ・保存）。個別のチャレンジは `tmp/` 直下のスクリプトで、共通部を import して
自分の問いだけを変える。現存するのは `tmp/neuromod_behavior_modes.py`（行動デモの参照ドライバ、L1）、
`tmp/neuromod_sr_curve.py`（SR 曲線、L3）、`tmp/neuromod_baseline_shift.py`（ベースライン依存の
効果反転、§7.2）、`tmp/neuromod_lesion.py`（損傷・重なり、L2 / §7.3）。

> `tmp/neuromod/` にはこの問題に属するものだけを置く。学習則は `nnn/`、RL・コンソリデーション・
> リザバーは各々のパッケージにあり、神経修飾の取り組みはそれらにほとんど依存しないので、
> それらの機構をここに持ち込まないこと。

**改稿**: 2026-08-02。旧稿（SR 図の作成手順）と、別稿にあった投稿戦略
（主張の限定・L1–L4・批判への応答・損傷実験の定義）を本稿に統合し、生物対応づけの議論を
主軸に据え直した。本稿がこの研究線の単一の文書である。SR レシピは付録 A・B に圧縮して残してある。

---

## 0. この文書の要点

1. **現デモの結びつけ方（修飾物質 → どの対象カテゴリに向かうか）は、生物学的な裏付けが最も薄い部類である。**
   モノアミン系に「食物モジュレータ」「シェルターモジュレータ」の対応物は無い（§4）。
2. **しかもこの結びつけは「結局 one-hot の鍵では？」という批判を自ら誘発する。**
   SR（内点最適）を主張の芯に据えるなら、修飾物質は**スカラーの計算メタパラメータ**に結びつけ、
   行動の違いはその帰結として出すほうが、生物側の逆 U 字文献と主張が直結する（§2, §3）。
3. **理論側から結びつけ先が制約される。** $\sigma$ の絶対値はゲージ依存量なので「修飾物質 $=\sigma$」は
   単独では意味を持たない。結びつけるべきは $\rho$（動員ダイヤル）または $\nu$（交差率）、
   実質的には「閾値までの距離を膜電位ゆらぎの標準偏差で測った量」である（§5）。
4. **L1・L2・L3 は実験で揃った。残る空白は L4（連続制御の定量化）だけである。**
   下の到達点を参照。次に作るのは行動レベルの SR（§7.1）で、RL には踏み込まない（§7.8）。

### 到達点（2026-08-02 時点）

| 段 | 結果 | 図・データ |
|---|---|---|
| **L1** アドレッシング | 共有重み + 3 場 → 3 行動（実装済み） | `neuromod_pure_panels.png` |
| **L2** 多重化 | 共有ユニットの損傷が両行動を同時に劣化させる（3 seed）。**$\alpha_\text{mix}>0$ が条件**（§7.3） | `lesion_bars.png`, `lesion.csv` |
| **L3** 資源性 | 学習水準スイープで内点最適 $\sigma^*\approx0.8$–$1.2$、平均場では消える（§6）。さらに**同じ用量がベースライン次第で逆向きに効き、反転点が用量とともに移動する**（3 seed、§7.2） | `baseline_shift_s3_curve.png`, `baseline_shift_analytic_*` |
| **L4** 連続制御 | 場の補間は動画で見せているが未定量（§8(c)） | — |
| §7.1 行動レベル SR | 濃度スライダー動画は完成。**厳密版は示唆的だが未確立**（多 seed が必要、§7.1） | `behavior_sr_slider.gif`, `behavior_sr_train*.csv` |

批判 (a)〜(e)（§1.3）のうち、実験を要する (b)(c)(e) はすべて裏付けが付いた。残るのは
(d) の文献整備（§8(a)）である。

---

## 1. 主張と論理の骨格

### 1.1 中心的主張と 4 段の論理

> **神経修飾物質様のノイズ場は、単一の重み集合に多重化された複数の方策を選択的にアドレスする。
> ノイズはシンボリックな「鍵（look-up key）」ではなく、最適強度をもつ機能的資源（確率共鳴）である。**

論理は「ノイズ場が方策を選ぶ」→「その選択は鍵ではなく資源である」→「ゆえに神経修飾の計算モデル
として意味がある」の順に閉じる。以後この 4 段を **L1–L4** と呼ぶ。

| 段 | 主張 | 根拠 | 状態 |
|---|---|---|---|
| **L1. アドレッシング** | 重みを一切変えずに、ノイズ場だけで異なる行動が選択される | 行動デモ：1 つの共有重み集合 + 3 つのノイズ場 → 3 つの識別可能な行動ベクトル場 | **実装済み** |
| **L2. 多重化（分割ではない）** | 各行動は互いに重なりうるサブネットに蓄えられており、排他的な区画への分割ではない | サブネット重なり（Jaccard、$S_i=\{k\mid\nu_k>0\}$）と共有ユニットの損傷実験（kill 三点操作） | **実装済み（§7.3、$\alpha_\text{mix}>0$ で成立）** |
| **L3. 資源性（鍵ではない）** | 動員されたサブネットの性能はノイズ強度に対して**内点最適**を持つ。鍵に最適強度は存在しない | SR 曲線：`--sweep train --model sample` で内点最適 $\sigma^*\approx0.8$–$1.2$（$h=0.2$ 大域固定の下で。§6） | **実装済み・主要結果** |
| **L4. 連続制御** | ノイズ場は離散スイッチではなく連続場であり、場の補間が未学習の中間行動を生む | 場の補間（デモの `cycle` モード）を定量化 | **部分的（§8(c)）** |

**なぜ L3 が科学的な芯なのか。** L1 だけでは「ノイズ場は one-hot なスイッチ ＝ 記号的な鍵に過ぎない」
という解釈を排除できない。鍵ならば、閾値を超える限りどんな強度でも同じ行動が選ばれるはずで、
**最適強度は存在しない**。したがって内点最適の存在は、この対抗仮説に対する直接の反証になる。
ノイズは「どれを選ぶか」を運ぶだけでなく、「選ばれた計算を成立させる」ために必要な量として振る舞っている。

### 1.2 主張しないこと（重要な戦略判断）

- **行動の創発は主張しない。** デモの 3 行動（Foraging / Avoidance / Sheltering）は教師付き
  ベクトル場回帰で設計的に与えたものであり、「創発した」と述べた瞬間に「行動は設計されている」
  という批判が正当になる。創発は今後の課題として明示的に繰り延べる。
- **神経修飾物質の生理学的モデルであるとは主張しない。** 主張するのは「ゲイン／興奮性の修飾によって
  機能サブネットが選択される」という**計算原理の対応**であり、特定の伝達物質の薬理学的再現ではない。

この 2 つを先に手放すことで、想定される 2 大批判が安価に回避でき、しかも残る主張
（アドレッシング + SR）は現有の実験だけで支えられる。

### 1.3 想定される批判と応答

| 批判 | 応答 | 必要な結果 |
|---|---|---|
| **(a) 行動は設計されたものであり、創発していない** | そのとおりであり、創発は主張しない。主張は「設計された複数方策が単一重み集合に多重化され、ノイズ場でアドレスされる」ことにある | なし（§1.2 の限定で回避） |
| **(b) ノイズ場は単なる one-hot の鍵ではないか** | 鍵に最適強度は存在しない。内点最適（SR）の存在が直接の反証。さらに**同じ介入がベースライン次第で逆向きに効き**、しかも**反転点が用量とともに移動する**（§7.2、3 seed で確認） | **L3（済）＋ §7.2（済）** |
| **(c) 単に区画分割されたサブネットのルックアップでは** | 場は重なりを持ち、共有ユニットの損傷は複数行動を同時に劣化させる（区画分割なら独立に壊れるはず） | **L2（済、§7.3）**。ただし $\alpha$ が近似 one-hot だと多重化は要請されないので、図は $\alpha_\text{mix}>0$ で作る |
| **(d) 神経修飾のモデルとしてのスコープ適合性** | 記事種別を Hypothesis & Theory とし、§2–§3 の対応表と引用で計算原理レベルの主張であることを明示 | Introduction の文献整備（§8(a)） |
| **(e) 決定論モデルでも同じことが言えるのでは** | 平均場（analytic）では SR が消え、低 $\sigma$ 端最適になる。閾値つきサンプル機構でのみ内点最適が現れる | **済**（§6 の analytic 対照） |

### 1.4 現デモの立ち位置と問題

現デモ `tmp/neuromod_behavior_modes.py` は、`ALPHA_STATES` を近似 one-hot に取り、
3 つのノイズ場が食物 / 脅威 / シェルターの各 drive に一対一で対応する構成になっている。
これは L1 を最短で見せるには良いが、二つの問題を抱えている。

- **生物学的な対応物が無い。** モノアミン（DA, NE, 5-HT, ACh）は「どの対象に向かうか」を
  labeled line で指定してはいない。指定に近い働きをするのは視床下部のペプチド性状態系であって、
  モノアミンではない（§4）。
- **主張の芯（L3）と噛み合わない。** 近似 one-hot の $\alpha$ の下では、場は事実上ルックアップキーとして
  働く。§1.3 の批判 (b) を、デモ自身が呼び込む形になっている。

したがって本稿の問いは「ノイズ場を神経修飾場と呼ぶとき、**修飾物質のどの計算的役割**に
対応づければ、L1 と L3 が同じデモの中で両立するか」である。

---

## 2. A. 生物規範的な結びつけの候補

### 2.1 計算論の背骨：メタパラメータとしての修飾物質

現在の標準的な立て方は **Doya (2002)** の metalearning 枠組みである。
DA = TD 誤差、ACh = 学習率、NE = 逆温度（探索）、5-HT = 割引率、という対応を提案しており、
「修飾物質 = 学習/決定のメタパラメータ」という見方が以後の基準になっている（Doya 2008 も同様）。

**本稿にとっての含意**: 修飾物質はまず**スカラー**であり、行動のラベルではない。
NNN で言えば、$\rho$ という単一のダイヤルに対応づけるのが自然であって、
3 本の独立な場に対応づけるのは筋が悪い。

### 2.2 ゲイン・SNR 変調（最も古典的で最も安全）

**Servan-Schreiber, Printz & Cohen (1990)** はカテコールアミンを活性化関数のゲイン変化として
定式化し、それを**そのまま SNR の変化として**扱い、逆 U 字を導いた。
一般論としては Salinas & Thier (2000)、Ferguson & Cardin (2020)。

**本稿にとっての含意**: NNN の実効ゲイン $\gamma_k=\lVert a_k\rVert/\sigma_k$（正典 §1.1）は
この量そのものである。対応は比喩ではなく形式的で、しかも $\gamma_k$ はゲージ不変なので
§5 の制約とも整合する。**引用として最も安全な一本**。

### 2.3 不確実性

**Yu & Dayan (2005)**: ACh が予期された不確実性、NE が予期されない不確実性を運ぶ。

### 2.4 探索 ↔ 利用（逆温度）

**Aston-Jones & Cohen (2005)** の adaptive gain theory。LC の tonic 活動と課題成績の間に
**逆 U 字**があり、tonic 高＝探索的、phasic ＝利用的という整理。

**本稿にとっての含意**: `idea_rl.md` §25 の「内部化された探索温度」がこれそのものである。
外付けの $\sigma_e$ を消して、方策自身の $T$ サンプルの広がりを探索温度にした構成は、
「修飾物質が逆温度を設定する」という命題の直接の実装になっている。ただしデモの順序としては
RL に踏み込む前に §7.1–7.3 を先に置く（§7.8）。

### 2.5 符号化 vs 想起の経路切り替え

**Hasselmo (2006; 1999)**: ACh 高で求心性入力を通し反回性結合を抑える、低で逆になる。
文字通り「同じ解剖構造の中でどの経路が計算に参加するかを修飾物質が決める」話であり、
**ノイズ場＝動員という主張の最も近い生理学的対応物**である。

**本稿にとっての含意**: `idea_consolidation.md` の $\rho$ ダイヤル（$\sigma_k=\rho_k\sigma_0$,
$h_k=h_0/\rho_k$）がそのまま「ACh 高＝新規学習に動員 / 低＝既存機能を保持」に読める。
ただし本稿でデモ化すると資源制御の話と主張が混ざる（§7.7）。

### 2.6 同一回路・複数出力（デモの最強の先行例）

**Marder (2012)**、**Marder & Bucher (2007)**、**Nusbaum & Beenhakker (2002)**。
甲殻類の胃腸神経節（STG）では、解剖学的に固定された一つの回路が、修飾物質しだいで
質的に異なる複数のリズムを出す。「重みは不変、場だけで行動が変わる」という L1 の主張の
生物学的先例としてこれ以上のものはない。**Bargmann (2012)** "Beyond the connectome" は
同じ論点を一般化しており、Introduction の掴みに使える。

関連して **Edelman & Gally (2001)** の degeneracy、**Marder & Taylor (2011)** の
「同じ振る舞いを複数のパラメータ集合が実現する」議論は、L2（多重化・重なり）の生物側の
受け皿になる（§7.3）。

> [new idea — 修飾物質が「読むモーメントの次数」を切り替える（`idea_core.md` §2.7）。§2.2 のゲイン変調
> $\gamma_k=\lVert a_k\rVert/\sigma_k$ は、idea_core §2.7.2 では**閾値へのマハラノビス計量の逆長さ**であり、
> 「修飾物質＝ゲイン」は形式的には「修飾物質が各閾値超平面のマハラノビス尺度を設定する」と読める（既存の
> 最も安全な対応の幾何的言い換え）。さらに一段強い仮説：活性は同じ二値化信号 $b=\mathbf 1[d+\eta>h]$ の
> **1 次モーメント（平均＝単調・閾値活性）**か**2 次モーメント（分散＝バンプ・交差活性）**かで質が変わる（§2.7.1）。
> もし修飾状態がこの**読みの次数**を偏らせるなら、§2.6 の「同一回路・複数出力（STG）」や §2.5 の「符号化↔想起の
> 経路切り替え」に、重み不変のまま**単調モード↔チューニング（バンプ）モード**という新しい計算的読みを与えられる
> （例：ACh 高＝鋭い特徴選択＝2 次バンプ／低＝滑らかな伝達＝1 次単調）。デモ候補として「場だけでモーメント次数を
> 動かし出力の質を切り替える」を §7 に検討する価値がある。]

### 2.7 容積伝達（「場である」ことの根拠）

Agnati・Fuxe らの **volume transmission**。放出が点対点でなく拡散性・勾配的であること、
つまり修飾は場だという主張の一次資料。「空間的に広がるノイズ場（バンプ状の動員パターン）↔
拡散性・容積伝達による広域かつ勾配的な放出」という対応を支える。

### 2.8 逆 U 字の実例（SR 主張の生物側の受け皿）

- DA と作業記憶: **Vijayraghavan et al. (2007)**、**Cools & D'Esposito (2011)**。
- 覚醒と検出成績: **McGinley, David & McCormick (2015)** が瞳孔径（NE 系覚醒の代理）に対する
  検出成績の逆 U 字を膜電位レベルで示した。**SR 曲線の直接の対応物として最も使いやすい実験**。
- 古典: **Yerkes & Dodson (1908)**。元の法則は単なる逆 U 字ではなく、
  **最適覚醒度が課題難度に依存して移動する**という 2 次元の主張である（§7.6）。

**本稿にとっての含意**: L3 の SR 曲線に「生物でも同じ形が測られている」という受け皿がつく。
これが無いと SR はモデル内部の性質にとどまる。

---

## 3. B. ノイズ場に結びつけると特に相性が良い題材

修飾物質をゲインでなく**ゆらぎ**に結びつけるとき、他の定式化では出てこない固有の主張は
「最適強度がある」「弱信号を通す」「探索を生む」の 3 つである。それぞれ生物側に実例がある。

### 3.1 歌鳥の LMAN（最良の例）

**Ölveczky, Andalman & Fee (2005)**、**Kao, Doupe & Brainard (2005)**、**Fee & Goldberg (2011)**。
大脳基底核系の LMAN が歌に**変動性を能動的に注入**し、しかもその注入量が社会的文脈
（メスの有無）で数十ミリ秒スケールで切り替わる。

つまり「**ノイズは除去すべき外乱ではなく、専用回路が生成し文脈で制御する機能的資源である**」という
NNN の第一原則（正典の 4 つの核の 1 つ）が、そのまま実験事実として存在している。

### 3.2 弱信号検出の SR

ザリガニ機械受容器 **Douglass et al. (1993)**、コオロギ尾葉 **Levin & Miller (1996)**、
ヒト触覚 **Collins et al. (1996)**、高齢者の姿勢制御 **Priplata et al. (2003)**、
総説は **McDonnell & Ward (2011)**。

閾値下の信号がノイズ無しには伝わらないという点で NNN の障壁と機構が同一なので、
検出課題（$d'$ 対 $\sigma$ の逆 U 字）は**最も正直な SR デモ**になる（§7.4）。

### 3.3 ノイズの生理学的実体は背景シナプス入力であり、それ自体が修飾される

**Chance, Abbott & Reyes (2002)** は釣り合った背景入力（＝ノイズ）がゲインを乗算的に変えることを
実験で示した。**Destexhe, Rudolph & Paré (2003)** の高コンダクタンス状態、
**Ho & Destexhe (2000)** の応答性増強がこれを支える。
つまり「$\sigma$ を変える」は生理学的に実在する操作である。

### 3.4 修飾物質は実際に変動性・相関を変える

**Goard & Dan (2009)**、**Polack, Friedman & Golshani (2013)**、
**Minces, Pinto, Dan & Chiba (2017)**。注意については **Cohen & Maunsell (2009)**。
空間的に構造化されたノイズ場を「注意のスポットライト」として見せる案の根拠になる。

### 3.5 行動状態の二値切り替え（配線が完全に分かっている系）

**Flavell et al. (2013)**: C. elegans の roaming/dwelling がセロトニンと神経ペプチドで切り替わる。
現デモに近い「採餌モードの切り替え」を、コネクトームが既知の系で正当化できる（§7.5）。

---

## 4. C. 現デモの結びつけ方をどう扱うか

labeled-line 構成（食物 / 脅威 / シェルターに 1 つずつ場）を残すなら、根拠はモノアミンではなく
**視床下部のペプチド性状態系**に求めるべきである。

- **AgRP ニューロン**は採餌を駆動し、しかも食物を「見た」瞬間に活動が落ちる
  （**Betley et al. (2015)**、**Chen et al. (2015)**）。内的欲求が感覚で更新されるという
  現デモの閉ループ規則（`neuromod_weights`）とよく合う。
- **オレキシン**は覚醒・探索、**オキシトシン**は社会行動。
- 防御行動側は **Mobbs et al. (2007)** の threat imminence（脅威の切迫度に応じて前頭前野優位から
  PAG 優位へ切り替わる）が使える。これは現在の `threat_gain` / `threat_range` による
  **連続的な緊急度**の実装と実はよく一致している。

**処方**: $\alpha$ が近似 one-hot である限り、場はキーとして働く。少なくとも次のどれかを行う。

1. $\alpha$ を明確に混合的にして、場の補間が中間行動を生むことを定量化する（L4、§8(c)）。
2. 行動選択そのものではなく、**同一行動の遂行品質**に場を効かせる図を足す（§7.1）。
3. 対応づけ自体を roaming/dwelling のような**状態**の軸に置き換える（§7.5）。

---

## 5. D. 理論側の制約 — 結びつけ先は $\sigma$ ではなく $\rho$ / $\nu$

### 5.1 「修飾物質 $=\sigma$」は単独では意味を持たない

正典 §4 のゲージ構造から、$\sigma_k$ の絶対値はゲージ依存量である。
$h/\sigma$ を保ったスケーリングは交差率 $\nu$ を変えず、出力もビット単位で不変になる
（`idea_duality.md` §12.6 の実測: 出力不変のまま素朴な overlap が $0.8305\to0.8720$ に動いた）。
したがって生物学的に結びつけるべきは **$\rho$（動員ダイヤル、制御量）**か
**$\nu_k=\mathbb E[z_k]$（交差率、観測量）**であり、実質的には

> **閾値までの距離を、膜電位ゆらぎの標準偏差で測った量**

である。

### 5.2 生物側にも二つのつまみが実在する

これは生物側にも実体があるのが良いところで、修飾物質は実際に二つのつまみを別々に回す。

| NNN | 生理学的対応 | 文献 |
|---|---|---|
| $h_k$（交差しきい値の半幅） | 平均膜電位・興奮性・後過分極電流の変調 | McCormick (1992) |
| $\sigma_k$（注入ノイズ強度） | 背景シナプス入力の分散、高コンダクタンス状態 | Chance et al. (2002)、Destexhe et al. (2003) |
| $\rho_k=$ 両者の連動 | 「ゆらぎ駆動レジームにおける閾値までの実効距離」 | Shadlen & Newsome (1998) 以降 |

**「修飾物質は興奮性とゆらぎを同時に動かし、機能的に効くのはその比である」**という言い方は、
NNN のゲージ理論と in vivo の fluctuation-driven regime の議論の両方に同時に支えられる。
報告時はゲージ不変な比 $h/\sigma^*\approx0.17$–$0.25$ を併記すること（§6）。

### 5.3 参加度を $\sigma$ のしきい値判定で測ってはいけない

行動デモのノイズ場は `--theta`（**以下 θ_cut と読む**。正典 §1.1 の $\theta=\{W,b\}$ と衝突するため。
CLI 引数名は変更しない）で強度を切り捨て、$\sigma_k=0$ のユニットを作る。
しかしこれを「ユニットが未動員／detach された」と読んではいけない。理由は 2 つ。

1. **層のスコープ**: 実装は `net(obs, stds=[field, field])` と**同一の場を隠れ 2 層の両方に**適用する
   （`structure=[6,H,H,2]`、$h$ は両層共通の大域定数）。$\sigma_k=0$ が理想形どおり $z\equiv0$ を
   与えるのは**第 1 隠れ層だけ**であり、第 2 隠れ層では上流のサンプルゆらぎが $\pm h$ をまたぐため
   $\sigma_k=0$ でも交差が残る（$\sigma=0$ リーク、実測 $\nu=0.11$–$0.17$ のプラトー。正典 §3.4）。
   厳密に沈黙させるには **kill の三点操作**（$\sigma_k\leftarrow0$、$h_k\leftarrow H_{\mathrm{DEAD}}=10^6$、
   $W^{(l+1)}[:,k]\leftarrow0$）が要る（正典 §3.5）。
2. **測り方**: 参加／不参加はゲージ不変な $\nu_k$ で測る（正典 §3.3, §4.3, §4.8）。

したがって θ_cut は「場の空間的な支持を切り出すためのバンプ裾の切り捨て」であって、
ユニットの退役操作ではない。`--model analytic` では期待応答が $\sigma\to0$ で（$d\neq0$ なら）
層によらず 0 に収束するので理想形と一致するが、本命図の `--model sample` は上の限定を受ける。

---

## 6. 支持証拠 — SR は sample 水準にしか存在しない（実測済み）

L3（資源性）を支える実験はすでに手元にある。

`--sweep train`（各ノイズ水準で新規ネットを学習し、その水準で採点）において:

- **`--model analytic`（平均場）は低 $\sigma$ 端で最適**。低 $\sigma$ でも重みの再スケールで学習でき、
  真の SR 障壁を持たない。
- **`--model sample`（実ノイズ注入＋閾値 $h>0$）は内点最適**（$\sigma^*\approx0.8$–$1.2$）で、
  低 $\sigma$ 側が実際に崩壊する（閾値下は伝達不能）。

すなわち **SR は平均場近似では消え、サンプルレベルの機構でのみ現れる**。
厳密な SR 図は `--sweep train --model sample`、`analytic` はその対照である。
テスト時スイープ（1 ネットで学習水準を固定し、テスト時のノイズだけを掃引）は全モデルで
逆 U 字を示すが、「学習水準の近傍で誤差が最小になるのは構造上当然」という交絡があるため、
補助図に留める。

**体制の明示が必須**（正典 §4.7）。`--sweep train` は `--crossing-h`（既定 0.2）を
**大域ハイパーパラメータとして固定したまま** $\sigma$ だけを掃引し、各点で $\theta=\{W,b\}$ を
再学習する ＝ **体制 (b)**（$h$ 固定 + $\theta$ 学習、ゲージ不在、$\sigma$ は本物の自由度）である。
もし $h\propto\sigma$ と連動させれば掃引は純粋な**ゲージ軌道**になり（$h/\sigma$ 一定 → $\nu$ 不変、
正典 §4.6）、曲線は平坦化して**内点最適そのものが定義上消える**。
報告時はゲージ不変な比 $h/\sigma^*\approx0.17$–$0.25$ を併記し、$\sigma^*$ の絶対値だけを
主張の担い手にしないこと。

**実測例（seed 7、`--sweep train --model sample`、$h=0.2$ 固定）**:

| 指標 | 端点 | 最適点 |
|---|---|---|
| 刺激依存信号 | 0.063 / 0.790 | **0.803**（$\sigma_{\text{train}}=1.19$、$h/\sigma^*=0.17$） |
| タスク誤差 | 0.414（低 $\sigma$ 崩壊）/ 0.030 | **0.014** |
| 場の分離度 | — | 1.430 |

※ 単一 seed の例。投稿には複数 seed の mean±std が要る（付録 A の R4）。

**別系統からの裏付け**: `idea_reservoir.md` §13.1（2026-07-30）で、**同一重み**の診断
（analytic で学習・凍結 → 同一重みのまま大域ノイズ強度 $\kappa_\sigma$ を掃引して両応答を採点）が
実装され、**sample は逆 U 字**（$\kappa_\sigma^*\approx0.20$–$0.23$、6 seed × 2 タスクすべて）、
**analytic は平坦なプラトー**が確認された。本稿の train スイープ（各水準で再学習）とは
別系統の手法であり、互いに補強する。本稿の文脈での同一重み診断は未実施。

---

## 7. デモ案

### 7.0 選定基準

良いデモの条件は 3 つある。**(i) L1–L4 のどれかの空白を埋めること**、
**(ii) 生物側に対応する実験事実があること**、**(iii) 既存資産で作れること**。
以下は RL を使わない案を優先度順に並べたものである（RL 系を後段に置く理由は §7.8）。

### 7.1 【推奨 1】行動レベルの SR ＝ 濃度スライダー・デモ

同一重み・同一環境で修飾物質「濃度」だけをスライドさせ、**両端で行動能力が壊れ、中間で最も
上手くやる**ところを見せる。曲線ではなく**動画として見せられる**のがこの案の核心である。

- 低濃度: 交差が起きず出力が沈黙し、エージェントが動けない（閾値下の崩壊）
- 適正: 採餌・回避・帰巣がきれいに回る
- 高濃度: 交差が飽和して出力が定数化し、行動が無方向・不規則になる

**指標**は既存の閉ループから素直に取れる。`tmp/neuromod_behavior_modes.py` はすでに
`food_strengths` / `hunger` / シェルター滞在 / 脅威接近を持っているので、
「1000 フレームあたりの摂食数」「シェルター到達までの時間」「脅威との最接近距離」を
$\sigma$ の関数にするだけでよい。

**なぜ良いか。** (1) L3 の縦軸が「ベクトル場回帰の MSE」という機械学習の指標から
**行動の指標**に変わり、神経科学の読者には Yerkes–Dodson (1908) の逆 U 字そのものに見える。
(2) §1.2 の自己限定（行動は設計であって創発ではない）と衝突しない。設計された行動の
**遂行品質**にノイズの最適値があると言っているだけだからである。
(3) 特集の主眼（修飾物質濃度と行動能力の関係）に最も近い。

**実装（2026-08-02）**: `tmp/neuromod_behavior_sr.py`。共通部に閉ループの 1 フレーム
`world.advance_frame` とヘッドレス測定 `world.rollout` を置き、**動画と測定が同じ力学を使う**ことを
保証した（`viz.animate` も同じ関数を通る）。動画は `viz.concentration_slider` の 4 パネル
（世界＋軌跡、現在濃度のノイズ場、速度トレース、行動曲線上を動くカーソル）。

**急性投薬版の実測（sample、2500 エポック、9 濃度 × 3 エピソード × 1500 フレーム）**:

| 濃度 | 0.15 | 0.24 | 0.38 | 0.60 | 0.95 | 1.50 | 2.39 | 6.00 |
|---|---|---|---|---|---|---|---|---|
| foods/1k | 0.22 | 1.11 | 4.67 | 8.67 | **9.56** | 4.67 | 0.00 | 0.00 |
| 移動距離 | 8.6 | 11.4 | 18.5 | 31.1 | 32.9 | 22.5 | 14.8 | 2.0 |

ピークは濃度 0.95（$h/\sigma=0.211$、§6 の帯 0.17–0.25 に一致）。動画では低濃度で速度がゼロに
張り付いて採餌が止まり、最適濃度で脅威を迂回する採餌ループが回り、高濃度で無方向になる。

> **この曲線は SR 主張を支持しない**（§6 の交絡）。ある濃度で育った動物を別の濃度で走らせれば、
> 平均場でも「実家から遠い」ほど成績が落ちるので、test スイープは構造上逆 U 字を出す。
> これは動画が見せている**急性投薬**の用量反応であって、主張には `--sweep train`
> （各濃度で新規に学習）と analytic 対照が要る。スクリプトは test スイープのとき明示的に警告する。

**厳密版（train スイープ、各濃度で新規学習）の実測 — 示唆的だが未確立**。
6 濃度、2000〜2500 エポック、各濃度 1 ネット × 3 エピソード:

| 濃度 | 0.150 | 0.314 | 0.656 | 1.372 | 2.869 | 6.000 | 端/ピーク比 |
|---|---|---|---|---|---|---|---|
| sample | 3.33±1.44 | 7.11±1.13 | 6.89±1.37 | **7.56**±1.13 | 6.44±1.91 | 4.22±0.63 | 0.44 / 0.56 |
| analytic | 4.22±0.63 | 6.89±0.31 | **7.56**±0.31 | 6.89±0.83 | 6.89±1.13 | 6.22±1.37 | 0.56 / 0.82 |

**方向は正しいが、主張はできない。** sample のほうが両端で落ちる（特に高端 4.22 対 6.22）ものの、
各濃度 1 ネットで、示している幅は**同一個体のエピソード間ばらつき**にすぎず、個体間変動を含まない。
低端の差（3.33±1.44 対 4.22±0.63）は有意でない。**`--seeds 3` で回し直すまでは結果と呼べない。**

**なぜ行動指標では弱いのか（2 つの仮説、どちらも重要）**:

1. **交差活性は低 $\sigma$ でも計算できる。** $\phi$ は前活性が帯 $\pm h$ の**内側**にあるとき発火するので、
   $\sigma$ が小さいネットワークは「狭帯域検出器」として学習し直せる。各動物が自分の濃度で育つ
   train スイープでは、この適応が働いて崩壊しない。§6 の低 $\sigma$ 崩壊は task error で測ったもので、
   行動では回避されうる。
2. **行動指標が粗く、頭打ちになっている。** `foods_per_1k` は 7 前後で飽和して見える（移動時間律速）。
   方策の劣化が採餌数に出にくい。到達時間や経路効率のような鋭い指標のほうが適する可能性がある。
   ただし**望む答えが出る指標を後から選ぶのは禁じ手**なので、指標を変えるならその理由を先に固定すること。

したがって現時点では、**§7.1 は動画としては完成、主張としては保留**である。

**この実験で判明した落とし穴（いずれも設定の問題で、ノイズの効果ではない）**:

1. **平均場では原理的に出ない。** `SimpleNNNSample` への載せ替えが前提（§6）。
2. **学習不足が「壊れたデモ」に見える。** analytic は task MSE をほぼ 0 にするのに 2500 エポック
   必要で、500 エポックの sample では採餌ゼロ・脅威に接近と、ノイズと無関係な理由で破綻する。
   `--epochs` の既定を 2500 にし、`--net-path` / `--load-net` で学習を再利用できるようにした。
3. **`--speed-gain` は cruise 用の値では小さすぎる。** learned モードの歩幅は $|v|$ に比例し、
   閉ループでは対象物の近くにいる時間が長く学習した減速則が効くため $|v|$ が小さい。既定 0.9 では
   移動距離が cruise の 4 割しかなく採餌できない。2.7–3.6 が適正。
   なお「動物の通常速度で正規化する」対策は、学習グリッド上の $|v|$ 中央値が 0.99 とほぼ飽和値で
   閉ループの実効値と食い違うため**効果がなく、採用しなかった**。
4. **安全指標は移動距離で正規化すること。** `close_frac`（脅威 0.2 以内のフレーム比）は最適点で
   0.027、低濃度で 0.036 だが、距離あたりでは 0.00087 対 0.0042 と**動いている方が 5 倍安全**である。
   止まっている動物は接近機会自体が少ないだけで、生の比率は交絡する。

### 7.2 【推奨 2】ベースライン依存の効果反転（同じ「薬」が逆に効く）

**Cools & D'Esposito (2011)** の中核的発見は、単なる逆 U 字ではなく、**同一のドーパミン作動薬が、
ベースラインの低い個体では成績を上げ、高い個体では下げる**ことである。逆 U 字の左右どちらに
乗っているかで効果の符号が反転する。ヒトの神経修飾薬理で最も頑健で有名な現象のひとつ。

**実装**: `tmp/neuromod_baseline_shift.py`（2026-08-02 作成）。設計の要点は 2 つ。

- **ベースラインは形質**。各ネットを**自分の $\sigma_\text{base}$ で学習**する（学習水準スイープ）。
  こうして初めて「そのベースラインで出せる最良の性能」どうしの比較になる。
- **薬は急性の摂動**。重みを凍結し、$\sigma\to g\cdot\sigma$ とノイズ場だけを掃く。
  **$h$ は固定する**こと。$\sigma$ と $h$ を一緒にスケールするのは純粋なゲージ変換で、
  何も変わらない（正典 §4.6）。薬は $\sigma$ 単独を動かさなければ意味を持たない。

**対照がそのまま予測になる**: 符号反転には内点最適が要るので、`--model sample` では出て
`--model analytic`（平均場、閾値下の障壁なし）では出ないはずである。両方を走らせるのが誠実な形。

図は 2 段で、上段が学習水準の逆 U 字に各ベースラインからの投薬を矢印で重ねたもの（矢の根は
曲線上、先は曲線外＝凍結ネットを別の $\sigma$ で評価した点）、下段が改善量の符号反転である。

**なぜ良いか。** L3 の証拠として素の逆 U 字より**強い**。鍵仮説では「同じ介入が状況次第で
助けにも害にもなる」ことを説明できないからである（§1.3 の批判 (b) への決定打）。

**実行結果（2026-08-02、成立）**。$h=0.2$ 固定、7 ベースライン $\in[0.3,2.2]$、
300 エポック、grid 13、$T=24$。まず単一 seed・$g=1.4$ の内訳:

| $\sigma_\text{base}$ | 0.30 | 0.62 | 0.93 | 1.25 | 1.57 | 1.88 | 2.20 |
|---|---|---|---|---|---|---|---|
| sample: $\Delta$err | **−0.059** | **−0.031** | **−0.006** | +0.007 | +0.014 | +0.016 | +0.020 |
| analytic: $\Delta$err | +0.016 | +0.036 | +0.055 | +0.061 | +0.063 | +0.064 | +0.061 |

- **sample は符号が反転する**。同じ用量が $\sigma_\text{base}\le0.93$ では成績を上げ、
  $\ge1.25$ では下げる。反転点は $\sigma\in(0.93,1.25)$、すなわち
  $h/\sigma\in(0.16,0.22)$ で、§6 が別系統（学習水準スイープ）で得た
  $h/\sigma^*\approx0.17$–$0.25$ の帯にそのまま入る。**独立な 2 つの測り方が同じ最適点を指した**。
- **analytic は反転しない**。しかも「効果なし」ではなく**全ベースラインで単調に悪化**する
  （task error はベースラインとともに 0.018→0.210 と単調増加、すなわち低 $\sigma$ 端最適）。
  これは §6 の「平均場は閾値下の障壁を持たない」の直接の帰結であり、**予測どおりの帰無対照**である。
  帰無が「何も起きない」ではなく「符号が一方向に固定される」形で出るのが、この対照の強いところ。

**多 seed・多用量版（3 seed、用量 4 点、同一の学習済みネットに後付けで投薬）**。
用量は凍結ネットへの急性摂動なので**再学習が要らず、用量軸はほぼ無料**である。

| 用量 $g$ | 1.2 | 1.4 | 1.8 | 2.5 |
|---|---|---|---|---|
| 反転点 $\sigma_\text{base}$ | 1.381 | 1.196 | 0.844 | 0.481 |
| 反転点での $h/\sigma$ | 0.145 | 0.167 | 0.237 | 0.416 |

- **符号反転は 4 つの用量すべてで、3 seed を通して成立する**（seed 帯は図で確認できる程度に狭い）。
- **反転点は用量とともに単調に低ベースライン側へ移動する**。これは逆 U 字が要求する第 2 の署名である。
  用量が大きいほど、より低いベースラインから出発しても最適点を通り越してしまうからである。
  鍵仮説はこの用量依存性も説明できない。
- $g=1.4$ の反転点 $h/\sigma=0.167$ は §6 の $h/\sigma^*\approx0.17$–$0.25$ の下端に一致する。

**残**: 課題難度を振ったときの反転点の移動（§7.6 の Yerkes–Dodson 2 次元版）は未着手。

### 7.3 【推奨 3】損傷・重なり実験（L2 の空白を埋める）

派手さはないが、**L1–L4 で唯一まだ埋まっていない主張**であり、必要な道具は全部揃っている。

**手順と定義**（素朴な定義を採るとどちらも成立しないので、ここで固定する）:

- **動員ユニット集合**は $S_i=\{k\mid \nu_k>0\}$（交差率の支持）と定義する。
  $\{k\mid\sigma_k>\text{閾値}\}$ で定義してはいけない。$\sigma_k$ の絶対値はゲージ依存量なので、
  集合そのものがゲージ変換で動き、Jaccard 係数が関数の量でなくなる（§5.1 の実測）。
  $\nu_k$ は 0 次同次なのでゲージ不変。$\cos(\mathcal P_A,\mathcal P_B)$ 型の場の重なりを
  指標にしないこと。
- **損傷（lesion）**は **kill の三点操作**（$\sigma_k\leftarrow0$、$h_k\leftarrow H_{\mathrm{DEAD}}=10^6$、
  $W^{(l+1)}[:,k]\leftarrow\mathbf 0$）で実装する。$\sigma_k\leftarrow0$ だけでは第 2 隠れ層のユニットは
  沈黙せず（$\sigma=0$ リーク、§5.3）、「損傷したのに壊れない」という結果になって実験自体が成立しない。
- **対照**: 損傷の効果は行動ごとの task error の増分で測り、$S_{\text{shared}}=S_A\cap S_B$ の損傷と
  $S_A\setminus S_B$ の損傷を対照する。区画分割仮説なら前者は空、または片方の行動しか劣化しない。

**生物側の受け皿**: Marder の STG（同一回路・複数出力）が直接の先行例で、さらに
Edelman & Gally (2001) の degeneracy、Marder & Taylor (2011) の「同じ振る舞いを複数の
パラメータ集合が実現する」議論に接続できる。損傷 + 重なりという実験の形自体が神経科学の
標準的な考え方なので、査読者に伝わりやすい。

**実行結果（2026-08-02、成立）**。`tmp/neuromod_lesion.py`。analytic・**隠れ 1 層**
（$\sigma$-only 動員が厳密に効く条件）、1200 エポック、grid 17、3 seed、リング半径 0.28。

> **群のサイズを揃えること**。shared は private より小さいので、生の損傷量を比べると
> 「どこを切ったか」と「どれだけ切ったか」が交絡する。全群を $|S_A\cap S_B|$ 個に揃え、
> private とランダムは複数回の抽選で平均する。
>
> **私有ユニット損傷の特異性は証拠ではない**。$\sigma$-only 動員では、ある行動の支持の外にある
> ユニットはその行動を評価する時点で既に沈黙しているので、除去しても厳密に 0 である。
> private 群は「k ユニット分の価値」を測る**物差し**であって、結果ではない。
> 証拠は「shared 損傷が**両方**を壊すか、その量が物差しに対して有意か」だけである。

$\alpha_\text{mix}=0.6$（3 seed 平均、task error の増分）:

| ペア | shared | private の物差し | random 対照 | 判定 |
|---|---|---|---|---|
| food\|threat | +0.115 / +0.084 | 0.079 / 0.078 | +0.068 / +0.040 | 多重化 |
| food\|shelter | +0.174 / +0.169 | 0.107 / 0.110 | +0.048 / +0.064 | 多重化 |
| threat\|shelter | +0.095 / +0.106 | 0.057 / 0.066 | +0.034 / +0.039 | 多重化 |

**共有ユニットの損傷は両行動を同時に劣化させ、しかもその量は各行動の私有ユニットの物差しを
上回り、同サイズのランダム損傷も上回る**。区画分割仮説ではこの署名は出ない（分割なら共有ユニットは
高々一方にしか属さない）。§1.3 の批判 (c) への直接の回答である。

**さらに重要な副産物: §4 の処方が L2 の成否を決めている。** $\alpha_\text{mix}$ を振ると
food|threat の判定が単調に変わる（shared の損傷量 vs ランダム対照、3 seed 平均）:

| $\alpha_\text{mix}$ | shared | random | 判定 |
|---|---|---|---|
| 既定（近似 one-hot） | +0.050 / +0.043 | +0.061 / +0.070 | **曖昧**（shared がランダムを下回る） |
| 0.3 | +0.060 / +0.058 | +0.052 / +0.075 | 多重化（弱） |
| 0.6 | +0.115 / +0.084 | +0.068 / +0.040 | **多重化** |

近似 one-hot では 3 つのタスクが**別々の入力次元**を読むので、場が重なっていてもネットワークは
共有する必要がない。「重なり」は使えるユニットの重なりであって、使われ方の重なりではない。
$\alpha$ を混合すると同じ入力次元を奪い合うようになり、初めて多重化が要請される。
**§4 で指摘した近似 one-hot の弱点が、L2 の実験結果として定量的に確認された**ことになる。
したがって L2 を主張する図は $\alpha_\text{mix}>0$ の条件で作ること。

### 7.4 弱信号検出（$d'$ 対 $\sigma$）

SR 文献の canonical パラダイムそのもの（§3.2）。閾値下刺激の検出感度を $\sigma$ の関数にする。
対応が比喩でなく**同一のパラダイム**なのが強みで、縦軸が $d'$ という心理物理の標準単位になる。
コストは低いが、行動デモ（§7.1）ほどの視覚的訴求力はない。掃引の骨格は
`tmp/neuromod_sr_curve.py` を流用できる。

### 7.5 roaming / dwelling への結び替え

**Flavell et al. (2013)** の C. elegans は、セロトニンと神経ペプチドで採餌の 2 状態
（速く直進的に探索する roaming と、遅く局所的に留まる dwelling）が切り替わる。
現デモの labeled-line を、この**状態**の軸に置き換えると、§4 で指摘した生物学的裏付けの薄さが
一気に解消される。環境もアニメーションもそのまま使え、しかも roaming↔dwelling は生物でも
連続的に遷移するので、場の補間による中間行動 ＝ **L4 の定量化**にそのまま乗る。
RL を使わずに探索↔利用の話ができるのも利点。

### 7.6 Yerkes–Dodson の難度依存性（novelty は最も高い）

元の法則は「難しい課題ほど最適覚醒度が低い」という**2 次元の予測**である。既存の掃引に
課題難度の軸を足すだけで、$\sigma^*$ が難度とともに移動するかを見られる。

**ただし「難度」の定義を先に決める必要がある。** 弱信号性を上げるとノイズが要る方向、
要求精度を上げるとノイズが害になる方向で、予測の向きが逆になる。裏を返すと、軸の定義を
明示することで NNN が**反証可能な予測**を出せる、という書き方ができる。

### 7.7 やらないほうがよいもの

**ACh の符号化 vs 固定化**（§2.5、Hasselmo）は生物学的には申し分ないが、$\rho$ による資源制御
（`idea_consolidation.md`）と主張が重なる。ここでデモ化すると 2 本の研究線の主張が混ざるので、
本稿では Discussion で一段落触れて `idea_consolidation.md` を指すに留める。

### 7.8 RL 系を後段に置く理由

§2.4 の探索温度デモ（`idea_rl.md` §25 の内部化温度をそのまま神経修飾の枠で提示し直す）は
生物学的根拠も既存資産も厚いが、**RL の機構そのものを説明する負担**が加わる。
§7.1–7.3 はいずれも教師付き学習のままで成立し、しかも L1–L4 の空白を直接埋める。
先に非 RL の 3 本で骨格を固め、探索温度は「同じ原理が RL でも成り立つ」という拡張として
後段に置くのが安全である。

### 7.9 まとめと着手順

| # | 案 | 埋まる段 | 生物側の根拠 | 再利用できる資産 | コスト |
|---|---|---|---|---|---|
| 1 | ベースライン依存の効果反転（§7.2） | L3 を決定的にする | Cools & D'Esposito (2011) | `tmp/neuromod_baseline_shift.py` | **実施済み（符号反転を確認）** |
| 2 | 損傷・重なり（§7.3） | **L2** | Marder、Edelman & Gally | `tmp/neuromod_lesion.py` | **実施済み（多重化を確認）** |
| 3 | 行動レベル SR（§7.1） | L3 を行動で見せる | Yerkes–Dodson、McGinley | `tmp/neuromod_behavior_sr.py` | **動画は完成、主張は保留** |
| 4 | 弱信号検出 $d'$（§7.4） | L3 の別証拠 | SR 文献直系 | `neuromod_sr_curve.py` の骨格 | 小 |
| 5 | roaming/dwelling（§7.5） | L4 ＋ §4 の解消 | Flavell et al. (2013) | 行動デモの環境・アニメ | 中 |
| 6 | Yerkes–Dodson 難度依存（§7.6） | L3 の新規予測 | Yerkes & Dodson (1908) | 既存掃引 + 難度軸 | 小（設計に注意） |
| 7 | 探索温度（§2.4、§7.8） | L3 の RL 版 | Aston-Jones & Cohen、LMAN | `tmp/rl_ppo_itemp_swingup.py` | 中（説明負担大） |

**着手順の推奨**: 1 → 2 → 3。**1（§7.2）と 2（§7.3）は完了**、3（§7.1）は動画のみ完成。
L1–L4 のうち残る空白は L4（連続制御の定量化、§8(c)）だけになった。

### 7.9.1 次の一手の判断（2026-08-03）

**判断: §7.1 の厳密版に追加投資せず、動画は「機構の直感的な提示」として使い、
資源は L4 の定量化に回す。**

根拠は 3 点である。

1. **L3 はすでに独立な 2 系統で成立している。** SR 曲線の内点最適（§6）と、ベースライン依存の
   符号反転（§7.2、3 seed、用量依存性つき）。§7.1 は L3 の**3 番目**の証拠であって、
   無くても主張は立つ。
2. **§7.1 の厳密版は費用が重く、成功も保証されない。** `--seeds 3` で 6 濃度 × 2000 エポックは
   **約 10 時間**。しかも現状の 1 seed 結果は sample と analytic の差が小さく（端/ピーク比
   0.44,0.56 対 0.56,0.82）、多 seed で誤差が縮んでも差が残るとは限らない。§7.1 に記した
   2 つの仮説（交差活性は帯の内側で発火するので低 $\sigma$ でも学習し直せる／`foods_per_1k` が
   移動時間律速で頭打ち）が正しければ、**行動指標では原理的に SR が見えにくい**。
3. **L4 は唯一の空白で、着手コストが低い。** `cycle` モード（場の補間）と `--alpha-mix` が
   すでにあり、`world.rollout` で行動指標も取れる。補間した場が生む行動を対応する補間目標に
   対して評価すれば、未学習の中間行動が滑らかに現れることを曲線にできる（§8(c)）。

**この判断が変わる条件**: 査読で「行動レベルの証拠が要る」と言われた場合、または L4 が早く
片付いて時間が余った場合。その際は §7.1 に戻り、まず**指標を替える**こと（到達時間・経路効率）を
検討する。ただし §7.1 に書いたとおり、望む答えが出る指標を後から選ぶのは禁じ手なので、
指標変更の理由を先に固定してから回すこと。

**動画の扱い**: `behavior_sr_slider.gif` は Results ではなく、Introduction か補足動画に置く。
キャプションには「急性投薬の用量反応であり、SR の主張は §6 と §7.2 が支える」と明記する。

> **コストの実測**（2026-08-02）: `--model sample` の学習は **約 1 秒/エポック**で、
> グリッド数や $T$ にほとんど依存しない（テンソルサイズではなくステップあたりの
> オーバヘッドが律速）。つまり **エポック × ベースライン × seed がそのまま秒数**になる。
> 掃引系のスクリプトを設計するときはこれを前提に規模を決め、重い実行は背景に回すこと。

---

### 7.10 実装済みの前提整備（2026-08-02）

§7 のデモに着手する前に、標準問題の側で 3 つの設定を直した。いずれも「主張を潰していた設定」である。

1. **場の中心をリング配置にした**（`fields.ring_centers`、`--field-radius`）。旧設定は直角三角形配置で、
   food–threat だけが対角線上に離れ、共有 2 ユニット（Jaccard 0.043）に対し他ペアは 8（0.200）だった。
   L2 の損傷実験で最も試したいペアが**ほぼ分割**されていたので、そのまま実験すれば反証したい区画分割
   仮説のほうを支持してしまう。リング配置後は 0.227 / 0.227 / 0.182 とほぼ対称になり、
   **重なり量が `--field-radius` という制御変数になった**（L2 の掃引軸として使える）。
   旧配置は `--corner-centers` で再現できる（0.043 / 0.200 / 0.200 を確認済み）。
2. **参加度を交差率 $\nu_k$ で測るようにした**（`fields.overlap_report`）。旧実装は
   `(field > 0).sum()` という $\sigma$ しきい値で「動員ユニット数」を数えており、§5.3 と §7.3 で
   自ら禁じた定義だった。なお analytic・第 1 隠れ層では両定義は一致する（$\sigma_k=0 \Rightarrow \nu_k=0$）。
   $\nu$ 定義が効くのは sample 水準・深い層・ゲージ変換下である。
3. **閉ループが出力の大きさを使うようにした**（`world.step_agent`、`--speed-mode learned`）。
   旧実装は予測ベクトルを正規化して向きだけを使い、学習した $\tanh$ 速度則を捨てていた。
   これでは低 $\sigma$ の沈黙が「凍りつく」ではなく単なる jitter にしか見えず、**逆 U 字の左腕が
   隠れる**。アニメに速度パネルを追加し、崩壊が乗るチャンネルを可視化した。旧挙動は
   `--speed-mode cruise`。

あわせて、**kill 三点操作を `fields.kill_units` として実装**した（$\sigma_k\leftarrow0$、
$h_k\leftarrow H_\mathrm{DEAD}$、$W^{(l+1)}[:,k]\leftarrow0$）。共有ユニットに適用すると
$\nu=0$・下流列 0 になることを検証済みで、§7.3 の損傷実験はこれを呼ぶだけで書ける。
`--hidden-layers 1` と `--model sample` も選べるようにしたので、$\sigma=0$ リークを避ける条件で
実験できる。

`--alpha-mix` は §4 の処方 1（$\alpha$ を混合的にする）を掃引可能にしたもので、既定は旧来の
近似 one-hot のままである。

## 8. 残作業・未解決

- **(a) 文献の一次確認**: §9 の文献はいずれも記憶からの引用であり、書誌情報（巻号・ページ）は未確認。
  主張の骨格を支える 5 本（Doya 2002、Servan-Schreiber et al. 1990、Marder 2012、
  McGinley et al. 2015、Ölveczky et al. 2005）は実物を当たること。
- **(b) SR 図の完成**: 高 $\sigma$ 側の低下が緩いため掃引を広げる（$\sigma\sim4$–5 まで）。
  複数 seed（付録 A の R4）で内点最適と低 $\sigma$ 崩壊の seed 非依存性を示す。
  任意で analytic と $\mathbb E[\text{sample}]$ の重ね描き。
- **(c) 連続制御の定量化（L4）【次の一手。判断は §7.9.1】**: デモの `cycle` モードは場の補間を動画で見せるが定量化されていない。
  補間した場が生む行動場を対応する補間目標に対して評価し、**未学習の中間行動が滑らかに現れる**
  ことを曲線で示す。§7.5 がこれを兼ねうる。
- **(d) 同一重み診断**: 各モデルを別々に学習する現行方式は厳密な同一重み比較ではない。
  リザバー文脈では実装済み（§6）。本稿の文脈では未着手。
- **(e) 実装の配置**: `tmp/neuromod/` と `tmp/neuromod_*.py` を `examples/` に移すか `tmp/` のままにするか
  （投稿時にコード公開するなら前者）。
- **(f) $\alpha$ 混合の下での行動デモ**: §7.3 で $\alpha_\text{mix}\in\{0.3,0.6\}$ の学習が成立し、
  多重化の署名が強まることは確認済み。残るのは、混合目標の下で行動デモ（`cycle` / `scripted`）が
  見た目にも成立するか、および L4 の中間行動が滑らかに出るか。

---

## 9. 文献（すべて未確認。§8(a) 参照）

**枠組み**
- Doya K (2002) Metalearning and neuromodulation. *Neural Networks* 15:495–506.
- Doya K (2008) Modulators of decision making. *Nat Neurosci* 11:410–416.
- Bargmann CI (2012) Beyond the connectome: how neuromodulators shape neural circuits. *BioEssays* 34:458–465.

**ゲイン・SNR・容積伝達**
- Servan-Schreiber D, Printz H, Cohen JD (1990) A network model of catecholamine effects: gain, signal-to-noise ratio, and behavior. *Science* 249:892–895.
- Salinas E, Thier P (2000) Gain modulation: a major computational principle. *Neuron* 27:15–21.
- Ferguson KA, Cardin JA (2020) Mechanisms underlying gain modulation in the cortex. *Nat Rev Neurosci* 21:80–92.
- Agnati LF, Fuxe K ら, volume transmission（一次資料を確認すること）。

**不確実性・探索・符号化**
- Yu AJ, Dayan P (2005) Uncertainty, neuromodulation, and attention. *Neuron* 46:681–692.
- Aston-Jones G, Cohen JD (2005) An integrative theory of locus coeruleus-norepinephrine function. *Annu Rev Neurosci* 28:403–450.
- Hasselmo ME (2006) The role of acetylcholine in learning and memory. *Curr Opin Neurobiol* 16:710–715.

**同一回路・複数出力・degeneracy**
- Marder E (2012) Neuromodulation of neuronal circuits: back to the future. *Neuron* 76:1–11.
- Marder E, Bucher D (2007) Understanding circuit dynamics using the stomatogastric nervous system. *Annu Rev Physiol* 69:291–316.
- Nusbaum MP, Beenhakker MP (2002) A small-systems approach to motor pattern generation. *Nature* 417:343–350.
- Edelman GM, Gally JA (2001) Degeneracy and complexity in biological systems. *PNAS* 98:13763–13768.
- Marder E, Taylor AL (2011) Multiple models to capture the variability in biological neurons and networks. *Nat Neurosci* 14:133–138.

**逆 U 字の実例**
- Yerkes RM, Dodson JD (1908) The relation of strength of stimulus to rapidity of habit-formation. *J Comp Neurol Psychol* 18:459–482.
- Vijayraghavan S et al. (2007) Inverted-U dopamine D1 receptor actions on prefrontal neurons engaged in working memory. *Nat Neurosci* 10:376–384.
- Cools R, D'Esposito M (2011) Inverted-U-shaped dopamine actions on human working memory and cognitive control. *Biol Psychiatry* 69:e113–125.
- McGinley MJ, David SV, McCormick DA (2015) Cortical membrane potential signature of optimal states for sensory signal detection. *Neuron* 87:179–192.

**ノイズ＝機能的資源**
- Ölveczky BP, Andalman AS, Fee MS (2005) Vocal experimentation in the juvenile songbird requires a basal ganglia circuit. *PLoS Biol* 3:e153.
- Kao MH, Doupe AJ, Brainard MS (2005) Contributions of an avian basal ganglia-forebrain circuit to real-time modulation of song. *Nature* 433:638–643.
- Fee MS, Goldberg JH (2011) A hypothesis for basal ganglia-dependent reinforcement learning in the songbird. *Neuroscience* 198:152–170.
- McDonnell MD, Ward LM (2011) The benefits of noise in neural systems. *Nat Rev Neurosci* 12:415–426.
- Douglass JK et al. (1993) Noise enhancement of information transfer in crayfish mechanoreceptors by stochastic resonance. *Nature* 365:337–340.
- Levin JE, Miller JP (1996) Broadband neural encoding in the cricket cercal sensory system enhanced by stochastic resonance. *Nature* 380:165–168.
- Collins JJ, Imhoff TT, Grigg P (1996) Noise-enhanced tactile sensation. *Nature* 383:770.
- Priplata AA et al. (2003) Vibrating insoles and balance control in elderly people. *Lancet* 362:1123–1124.
- Faisal AA, Selen LPJ, Wolpert DM (2008) Noise in the nervous system. *Nat Rev Neurosci* 9:292–303.

**ノイズの生理学的実体・変動性の修飾**
- Chance FS, Abbott LF, Reyes AD (2002) Gain modulation from background synaptic input. *Neuron* 35:773–782.
- Destexhe A, Rudolph M, Paré D (2003) The high-conductance state of neocortical neurons in vivo. *Nat Rev Neurosci* 4:739–751.
- Ho N, Destexhe A (2000) Synaptic background activity enhances the responsiveness of neocortical pyramidal neurons. *J Neurophysiol* 84:1488–1496.
- Shadlen MN, Newsome WT (1998) The variable discharge of cortical neurons. *J Neurosci* 18:3870–3896.
- McCormick DA (1992) Neurotransmitter actions in the thalamus and cerebral cortex. *Prog Neurobiol* 39:337–388.
- Goard M, Dan Y (2009) Basal forebrain activation enhances cortical coding of natural scenes. *Nat Neurosci* 12:1444–1449.
- Polack PO, Friedman J, Golshani P (2013) Cellular mechanisms of brain state-dependent gain modulation in visual cortex. *Nat Neurosci* 16:1331–1339.
- Minces V, Pinto L, Dan Y, Chiba AA (2017) Cholinergic shaping of neural correlations. *PNAS* 114:5725–5730.
- Cohen MR, Maunsell JHR (2009) Attention improves performance primarily by reducing interneuronal correlations. *Nat Neurosci* 12:1594–1600.

**状態依存の行動切り替え（ペプチド系）**
- Betley JN et al. (2015) Neurons for hunger and thirst transmit a negative-valence teaching signal. *Nature* 521:180–185.
- Chen Y, Lin YC, Kuo TW, Knight ZA (2015) Sensory detection of food rapidly modulates arousal-related AgRP neurons. *Cell* 160:829–841.
- Flavell SW et al. (2013) Serotonin and the neuropeptide PDF initiate and extend opposing behavioral states in C. elegans. *Cell* 154:1023–1035.
- Mobbs D et al. (2007) When fear is near: threat imminence elicits prefrontal-periaqueductal gray shifts in humans. *Science* 317:1079–1083.

**比較対象**
- Masse NY, Grant GD, Freedman DJ (2018) Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization. *PNAS* 115:E10467–E10475.
  ※ 最近傍の先行研究だが、あちらはランダム割当のゲート + 別途のシナプス安定化という**二機構**であり、
  本稿は**単一の物理量が選択と計算の両方を担う**点で異なる。詳しい対比は `idea_consolidation.md` §14.4。

---

## 付録 A. SR 曲線の再現手順（最小）

§6 の実測を再現するための操作。対象コードは
[`tmp/neuromod_sr_curve.py`](../tmp/neuromod_sr_curve.py)（行動デモ
[`tmp/neuromod_behavior_modes.py`](../tmp/neuromod_behavior_modes.py) の学習・データ・
ノイズ場をそのまま再利用する診断スクリプト）。依存は `numpy`, `torch`, `matplotlib` のみ、
実行はリポジトリ直下から、CPU 既定。

**測る量**（CSV は列順 `x, separation, signal, task_err`。1 列目名は `sigma_train` or `test_s`）:

- `separation` — 3 つのノイズ場が生む出力ベクトル場どうしの平均ペア距離（各場がどれだけ
  別々の行動を動員するか）。
- `signal` — 出力の**刺激依存成分**。SR で単峰になるべき本命量。低ノイズ端（閾値下で交差が
  起きず出力が消える）と過大（交差飽和で定数化）の両側で 0 に向かう。
- `task_err` — 学習目標への MSE（参照用）。

**R1. 厳密な SR（本命）— sample × 学習水準スイープ**
```bash
python tmp/neuromod_sr_curve.py \
    --sweep train --model sample \
    --grid-side 25 --epochs 1500 --train-steps 15 \
    --s-min 0.1 --s-max 4.0 --samples 96 --crossing-h 0.2 --seed 7 \
    --save data/sr_train_sample_seed7.csv --no-show
```
重い（sample ネットを 15 個学習）。軽量プレビューは
`--grid-side 17 --epochs 700 --train-steps 9 --samples 32`。

**R2. 対照（平均場＝SR なし）— analytic × 学習水準スイープ**
```bash
python tmp/neuromod_sr_curve.py \
    --sweep train --model analytic \
    --train-steps 15 --s-min 0.1 --s-max 4.0 --seed 7 \
    --save data/sr_train_analytic_seed7.csv --no-show
```

**R3. テスト時スイープ（高速・補助図）**
```bash
python tmp/neuromod_sr_curve.py --model sample \
    --grid-side 21 --epochs 1500 --samples 96 --s-max 4.0 --s-steps 60 --seed 7 \
    --save data/sr_test_sample_seed7.csv --no-show
```

**R4. seed 頑健性 — R1 を seed 掃引**
```bash
for s in 0 1 2 3 4; do
  python tmp/neuromod_sr_curve.py --sweep train --model sample \
    --grid-side 21 --epochs 1000 --train-steps 11 --s-min 0.1 --s-max 4.0 --samples 64 \
    --seed $s --save data/sr_train_sample_seed$s.csv --no-show
done
```

**標準出力の読み方**: `INTERIOR optimum -> SR confirmed` は signal のピークが端点でない、すなわち
SR 成立。`at an endpoint` は端点最適（analytic の train スイープで想定どおり）。

**落とし穴**

- **`--sigma` は掃引変数ではない。** `--sigma`（既定 0.22）は 8×8 ユニットシート上の**バンプの
  空間的な広がり**であり、ノイズ強度ではない。ノイズ強度は `--base-std`（test スイープ）
  または `--s-min`〜`--s-max` の掃引変数（train スイープ）。内点最適 $\sigma^*\approx0.8$–$1.2$ は
  後者の軸上の値である。
- **`--samples`（コードの `t`）は時間ではない。** 正典の $T$、すなわち 1 入力あたりの確率的
  forward サンプル数（正典 §1.1, §3.1）。時間発展・時間相関ノイズは本スクリプトに存在しない。
- **`--crossing-h` は掃引中に固定する。** SR には $h>0$ が必須で、固定こそが体制 (b) の条件
  （§6）。$\sigma$ と連動させると内点最適が定義上消える。
- **test スイープの `task_err` は交絡する。** 学習水準の近傍で最小になるのは構造上当然なので、
  主張は `separation` / `signal`（非学習目標）の内点最適か、`--sweep train` で行う。
- **SR 判定はモデル非依存ではない**（正典 §2.6）。`analytic` で内点最適が出ないのは失敗ではなく
  予測どおりの対照である。`statistic` 水準での SR の有無は**未検証**であり、
  「内点にピークが出れば SR」という読み方をそのまま適用してはいけない。

---

## 付録 B. 図の枠組み

| 図 | 使う CSV | 描く列 vs 横軸 | 主張 |
|---|---|---|---|
| **Fig 1 厳密な SR** | `sr_train_sample_seed7` | `task_err` と `signal` vs `sigma_train` | 内点最適＋低 $\sigma$ 崩壊＝真の SR（**$h=0.2$ 固定＝体制 (b)** をキャプションに明記し、$h/\sigma^*\approx0.17$–$0.25$ を併記） |
| **Fig 2 機構 vs 平均場** | `sr_train_sample_*` ＋ `sr_train_analytic_*` | `signal` vs std を重ね描き | sample = 内点、analytic = 低 $\sigma$ 端（障壁なし） |
| **Fig 3 場の分離度** | 任意の train/test CSV | `separation` vs std | ノイズ強度が場の区別度を決め最適点をもつ |
| **Fig 4 テスト時の逆 U 字** | `sr_test_sample_*` | `separation` と `signal` vs `test_s` | 1 ネットでの逆 U 字（安価な補助図） |
| **Fig 5 seed 頑健性** | `sr_train_sample_seed{0..4}` | `task_err` の mean±std vs `sigma_train` | 傾向が seed に依存しない |

本文図は Fig 1（厳密 SR）、Fig 2（機構 vs 平均場）、行動デモのパネル図（3 場 × ベクトル場）、
§7.3 の損傷図の 4 枚程度に絞る。§7.2 のベースライン依存図を Fig 1 の隣に置くと L3 が最も強くなる。

CSV は純数値なので `numpy.loadtxt(path, delimiter=',')`（`#` 行は自動スキップ）または
`pandas.read_csv(path, comment='#', header=None, names=['x','separation','signal','task_err'])`
で読める。
