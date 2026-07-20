"""
spike_coding_regime.py — ノイズの多寡がスパイク列の符号化様式を決めるか

仮説
----
NNN の交差活性は、しきい値 ±h をまたぐイベントとしてスパイクを生成する。
時系列入力に対して、交差を駆動する源は 2 つある:

    (i)  決定論的な信号変化   d(t) が ±h を横切る    -> 精密なタイミングで発火
    (ii) 注入ノイズ           d(t)+eta(t) が揺らぐ   -> 確率的な発火 = レート

したがって「ノイズが少ない = タイミング符号 / ノイズが多い = レート符号」が
予測される。本スクリプトはこれを、同一の関数を異なるノイズ強度で回帰させた
ネットワークのスパイク列から、複数の統計量で検証する。

中心となる制御パラメータ (ゲージ不変)
------------------------------------
NNN は (w, b, sigma, h) の共通スケールに対してゲージ不変なので、sigma の
絶対値そのものには意味がない (docs/idea_consolidation.md §3)。時間軸を入れて
初めて意味を持つ無次元量は「1 ステップあたりの決定論的な変化 / ノイズ std」

    Gamma = |w_k * dx/dt| * dt / sigma

である (分子・分母がゲージ変換 alpha で同率に伸縮するので不変)。予測:

    Gamma >> 1 : 信号が 1 ステップでしきい値帯を通過 -> 交差は決定論的イベント
    Gamma << 1 : 帯の中で準静的に揺らぐ             -> 交差は確率的 = レート

sigma と信号速度という別々の物理ノブを振って、指標が Gamma の一軸に
collapse するかを見るのが本スクリプトの主眼 (--speeds で 2 次元掃引)。

測る量 (すべて同一のスパイク列から)
----------------------------------
1. 試行間再現性 rel(delta): 同じ刺激 x(t) を R 試行 (ノイズのみ独立) 提示し、
   幅 delta で平滑化したスパイク列の試行間相関。
   タイミング符号 -> delta=1 でも高い / レート符号 -> 粗い delta で初めて上がる。
2. 情報の分解: 窓内の (a) スパイク数 count と (b) サブビンのパターン pattern の
   それぞれと刺激との相互情報量。timing_gain = I(pattern) - I(count) >= 0 が
   「数を超えて時刻が運ぶ情報」。シャッフル減算でバイアス補正。
3. デコーダ比較: スパイク列からの線形デコード誤差を
   (a) レートデコーダ (窓平均) と (b) FIR デコーダ (時間カーネルを学習)
   で比較する。FIR はレートを部分集合として含むので、改善分が
   「線形デコーダが使えるタイミング情報」。さらに窓内で発火時刻を
   シャッフル (数は保存) した対照で、その寄与を直接確認する。
4. ISI 統計: CV と ISI ヒストグラム。決定論的交差 -> 信号周期にロックした
   低 CV / ノイズ駆動 -> 幾何分布的な CV ~ 1。
5. 交差の起源分解 (反実仮想): 同じ学習済みパラメータで
   (a) ノイズのみ (刺激を定数に固定) と (b) 信号のみ (sigma=0) の交差率を測り、
   実際の交差がどちらに由来するかを分解する。

使い方
------
    python tmp/spike_coding_regime.py --quick            # 動作確認 (1 分弱)
    python tmp/spike_coding_regime.py                    # sigma 掃引
    python tmp/spike_coding_regime.py --speeds 1,2,4     # Gamma collapse (2 次元)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from nnn.activation import Crossing  # noqa: E402
import fncl_driver as fncl  # noqa: E402
from fncl_driver import save_json, savefig, write_text  # noqa: E402


# ============================================================
# 刺激と目標
# ============================================================
def make_signal(T: int, speed: float, device, kind: str = "sine"):
    """時系列刺激 x(t) と目標 y(t) = g(x(t)) を返す。

    speed は 1 ステップあたりの変化量のスケール (= Gamma の分子側のノブ)。
    目標は非単調 (g(x) = sin(1.5 pi x)) にして、隠れ層のバンプが必要な形にする。
    """
    t = np.arange(T, dtype=np.float32)
    if kind == "sine":
        x = np.sin(2.0 * np.pi * speed * t / T * 4.0).astype(np.float32)
    elif kind == "rw":  # 帯域制限ランダムウォーク (広帯域の対照)
        rng = np.random.RandomState(0)
        w = rng.randn(T).astype(np.float32)
        k = max(1, int(round(T / (64.0 * speed))))
        ker = np.ones(k, dtype=np.float32) / k
        x = np.convolve(w, ker, mode="same")
        x = (x / (np.abs(x).max() + 1e-8)).astype(np.float32)
    else:
        raise ValueError(kind)
    y = np.sin(1.5 * np.pi * x).astype(np.float32)
    return (torch.tensor(x, device=device), torch.tensor(y, device=device))


def autocorr_time(sig: np.ndarray, thresh: float = 0.5) -> float:
    """刺激の自己相関時間 tau_x [ステップ]: 正規化自己相関が thresh を切るラグ。

    速度をまたぐ比較では、全ての解析窓をこの単位で指定する（§5.3(3)）。
    正弦波なら tau_x = P/6 (thresh=0.5) で周期 P に比例するので、速度比が
    そのまま tau_x の比になる。
    """
    s = np.asarray(sig, dtype=np.float64)
    s = s - s.mean()
    ac = np.correlate(s, s, mode="full")[len(s) - 1:]
    ac /= ac[0] + 1e-12
    below = np.flatnonzero(ac < thresh)
    return float(max(1.0, below[0])) if below.size else float(len(s) // 4)


def timescale_params(tau: float, args) -> dict:
    """全解析窓を tau_x の倍数として決める（項目1の中核）。

    絶対ステップで固定した窓は信号時定数が変わると必ず交絡する（§5.3(3)）。
    ここで決めた窓は「刺激が有意に変化する時間」を単位に揃っているので、
    速度をまたいだ Gamma collapse の比較が初めて意味を持つ。

    FIR デコーダは**タップ数を固定したまま時間スパンだけ tau に比例**させる
    （等間隔に間引く）。そうしないと tau とともに特徴次元が増えて、
    デコーダの表現力自体が条件間で変わってしまう。
    """
    def R(frac, lo=1):
        return int(max(lo, round(frac * tau)))

    if args.window_mode == "abs":       # 旧挙動（比較用の対照アーム）
        span = args.n_lag
        taps = np.arange(min(args.lag_taps, span))
        return {"tau": tau, "window": args.window, "lag_span": span,
                "lag_taps": taps,
                "sub_len": args.sub_len, "stride": args.stride,
                "jitters": args.jitter_list,
                "rel_widths": [1.0, 2.0, 4.0, 8.0, 16.0],
                "rel_floor_hit": 0, "info_win": args.n_sub * args.sub_len}

    span = R(args.span_frac)                      # FIR / rate の時間スパン
    taps = np.unique(np.linspace(0, span - 1, min(args.lag_taps, span))
                     .astype(int))
    info_win = R(args.info_frac)                  # 情報分解の窓
    sub_len = int(max(1, round(info_win / args.n_sub)))
    jitters = sorted({R(f) for f in args.jitter_fracs})
    rel_widths = [float(max(1.0, f * tau)) for f in args.rel_fracs]
    # 1 ステップの床に当たった窓は実効的に正規化されていない（§5.5 課題3）。
    # サンプリング解像度が正規化の下限を与えるので、当たったら記録して警告する。
    floor_hit = sum(1 for f in args.rel_fracs if f * tau < 1.0)
    return {"tau": tau, "window": R(args.window_frac), "lag_span": span,
            "lag_taps": taps, "sub_len": sub_len,
            "stride": int(max(1, sub_len)), "jitters": jitters,
            "rel_widths": rel_widths, "rel_floor_hit": floor_hit,
            "info_win": info_win}


# ============================================================
# 時間軸つき NNN (realtime 解釈: dim 1 = 時間)
# ============================================================
def causal_ma(z: torch.Tensor, w: int) -> torch.Tensor:
    """[N, T, H] の時間軸に沿った因果移動平均 (幅 w)。"""
    if w <= 1:
        return z
    N, T, H = z.shape
    zp = F.pad(z.transpose(1, 2), (w - 1, 0))
    ker = torch.ones(1, 1, w, device=z.device) / w
    out = F.conv1d(zp.reshape(N * H, 1, -1), ker)
    return out.reshape(N, H, T).transpose(1, 2)


class TemporalNNN(nn.Module):
    """d_k(t) = w_k x(t) + b_k -> ノイズ -> ±h 交差 -> 窓平均 -> 線形読み出し。

    交差は nnn.activation.Crossing をそのまま使う ([N,T,H] の dim 1 = 時間に
    沿った cyclic XOR、backward は要素ごとの KDE スロープ)。sample 版
    (CrossingSample) は backward で T 方向に平均してしまい時間的 credit が
    消えるため、時間符号の研究では realtime 版を使うのが正しい。
    """

    def __init__(self, hidden: int, sigma: float, h: float, window: int,
                 device):
        super().__init__()
        self.H, self.sigma, self.h, self.window = hidden, sigma, h, window
        # 入力側: 刺激レンジ [-1,1] を覆うようにしきい値交差点を配置
        centers = torch.linspace(-1.0, 1.0, hidden, device=device)
        mag = 1.0 + 0.5 * torch.rand(hidden, device=device)
        sign = torch.where(torch.rand(hidden, device=device) < 0.5, -1.0, 1.0)
        w1 = (mag * sign)
        self.w1 = nn.Parameter(w1)
        self.b1 = nn.Parameter(-w1 * centers)
        self.a = nn.Parameter(torch.zeros(hidden, device=device))
        self.b_out = nn.Parameter(torch.zeros(1, device=device))

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        """x [N, T] -> d [N, T, H]"""
        return x.unsqueeze(-1) * self.w1 + self.b1

    def spikes(self, x: torch.Tensor, sigma: float = None) -> torch.Tensor:
        s = self.sigma if sigma is None else sigma
        d = self.pre(x)
        if s > 0:
            d = d + s * torch.randn_like(d)
        return Crossing.apply(d, self.h)          # [N, T, H] in {0, .5, 1}

    def forward(self, x: torch.Tensor, sigma: float = None) -> torch.Tensor:
        z = self.spikes(x, sigma)
        r = causal_ma(z, self.window)
        return r @ self.a + self.b_out            # [N, T]


def train(net: TemporalNNN, x: torch.Tensor, y: torch.Tensor, trials: int,
          epochs: int, lr: float, log_every: int = 0):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xb = x.unsqueeze(0).repeat(trials, 1)
    yb = y.unsqueeze(0).repeat(trials, 1)
    losses = []
    for e in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(net(xb), yb)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if log_every and e % log_every == 0:
            print(f"      epoch {e:4d} mse={float(loss):.5f}")
    return losses


# ============================================================
# 解析 1: 試行間再現性 (タイミングの精度)
# ============================================================
def gauss_smooth(z: np.ndarray, width: float) -> np.ndarray:
    """[R, T, H] を時間方向にガウス平滑化 (width = std, ステップ単位)。"""
    if width <= 0.5:
        return z
    r = int(max(1, round(3 * width)))
    t = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (t / width) ** 2)
    k /= k.sum()
    out = np.empty_like(z)
    for h in range(z.shape[2]):
        for i in range(z.shape[0]):
            out[i, :, h] = np.convolve(z[i, :, h], k, mode="same")
    return out


def reliability(spk: np.ndarray, widths) -> dict:
    """試行間相関 vs 平滑化幅。spk [R, T, H] (binary)。"""
    R = spk.shape[0]
    out = {}
    for wdt in widths:
        sm = gauss_smooth(spk, wdt)
        cors = []
        for h in range(spk.shape[2]):
            m = sm[:, :, h]
            m = m - m.mean(axis=1, keepdims=True)
            nrm = np.linalg.norm(m, axis=1) + 1e-12
            if np.median(nrm) < 1e-8:
                continue
            c = (m @ m.T) / np.outer(nrm, nrm)
            iu = np.triu_indices(R, k=1)
            cors.append(float(np.mean(c[iu])))
        out[float(wdt)] = float(np.mean(cors)) if cors else 0.0
    return out


# ============================================================
# 解析 2: 情報分解 (count vs pattern)
# ============================================================
def _mi_plugin(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> float:
    """離散 a, b の plug-in 相互情報量 [bit]。"""
    joint = np.zeros((na, nb), dtype=np.float64)
    np.add.at(joint, (a, b), 1.0)
    joint /= joint.sum()
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log2(joint[nz] / (pa @ pb)[nz])))


def info_decomposition(spk: np.ndarray, stim: np.ndarray, n_sub: int,
                       sub_len: int, stride: int, n_lvl: int,
                       n_shuffle: int = 8, rng=None) -> dict:
    """窓内の count / pattern が刺激について持つ情報 (シャッフル補正つき)。

    spk [R, T, H] binary, stim [T]. 窓 = n_sub 個のサブビン x sub_len ステップ。
    """
    rng = rng or np.random.RandomState(0)
    R, T, H = spk.shape
    W = n_sub * sub_len
    starts = np.arange(0, T - W, stride)
    # 刺激レベル (窓中心の値を分位点で n_lvl 段に離散化)
    ctr = stim[starts + W // 2]
    edges = np.quantile(ctr, np.linspace(0, 1, n_lvl + 1)[1:-1])
    lvl = np.digitize(ctr, edges)
    lvl_all = np.tile(lvl, R)

    ic, ip = [], []
    for h in range(H):
        cnt, pat = [], []
        for r in range(R):
            tr = spk[r, :, h]
            seg = np.stack([tr[s:s + W] for s in starts])       # [S, W]
            sub = seg.reshape(len(starts), n_sub, sub_len).sum(axis=2)
            cnt.append(np.clip(sub.sum(axis=1), 0, n_lvl * 2))
            pat.append(((sub > 0).astype(np.int64)
                        * (2 ** np.arange(n_sub))).sum(axis=1))
        cnt = np.concatenate(cnt).astype(np.int64)
        pat = np.concatenate(pat).astype(np.int64)
        if cnt.max() == cnt.min() and pat.max() == pat.min():
            continue
        n_cnt, n_pat = int(cnt.max()) + 1, 2 ** n_sub
        i_c = _mi_plugin(lvl_all, cnt, n_lvl, n_cnt)
        i_p = _mi_plugin(lvl_all, pat, n_lvl, n_pat)
        # バイアス補正: 刺激ラベルをシャッフルした帰無の平均を引く
        bc, bp = [], []
        for _ in range(n_shuffle):
            sh = rng.permutation(lvl_all)
            bc.append(_mi_plugin(sh, cnt, n_lvl, n_cnt))
            bp.append(_mi_plugin(sh, pat, n_lvl, n_pat))
        ic.append(max(0.0, i_c - float(np.mean(bc))))
        ip.append(max(0.0, i_p - float(np.mean(bp))))
    if not ic:
        return {"i_count": 0.0, "i_pattern": 0.0, "timing_gain": 0.0,
                "timing_frac": 0.0}
    i_c, i_p = float(np.mean(ic)), float(np.mean(ip))
    gain = max(0.0, i_p - i_c)
    return {"i_count": i_c, "i_pattern": i_p, "timing_gain": gain,
            "timing_frac": gain / (i_p + 1e-12)}


# ============================================================
# 解析 3: デコーダ比較 (rate / FIR / 時刻シャッフル)
# ============================================================
def _ridge_fit_eval(X: np.ndarray, y: np.ndarray, lam: float = 1e-3):
    """前半で学習し後半で評価する線形リッジ回帰の MSE。"""
    n = len(y) // 2
    Xtr, ytr, Xte, yte = X[:n], y[:n], X[n:], y[n:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    A = Xtr.T @ Xtr + lam * len(Xtr) * np.eye(Xtr.shape[1])
    wgt = np.linalg.solve(A, Xtr.T @ (ytr - ytr.mean()))
    pred = Xte @ wgt + ytr.mean()
    return float(np.mean((pred - yte) ** 2))


def _lag_matrix(spk: np.ndarray, taps, span: int = None) -> np.ndarray:
    """[R,T,H] -> [R*T', H*len(taps)] のラグ設計行列。

    taps は使用するラグのリスト（等間隔に間引かれている）。span を tau に
    比例させつつタップ数を固定することで、**デコーダの表現力を条件間で
    一定に保ったまま時間スパンだけ信号時定数に合わせられる**。
    """
    R, T, H = spk.shape
    taps = np.asarray(taps, dtype=int)
    span = int(span if span is not None else taps.max() + 1)
    cols = [spk[:, span - 1 - int(l): T - int(l), :] for l in taps]
    X = np.concatenate(cols, axis=2)
    return X.reshape(-1, H * len(taps))


def jitter_spikes(spk: np.ndarray, J: int, rng) -> np.ndarray:
    """各スパイクを [-J, +J] の一様乱数だけ時間方向にずらす。

    数（レート）はほぼ保存し、精密な時刻だけを段階的に壊す摂動。窓内
    シャッフル（初版）は窓幅がレートの変動時定数より長いと「レートが窓内で
    変化していること」を壊してしまい、タイミング符号の指標にならない
    （§5.1 欠陥2）。ジッタなら J を小さく取ることで両者を分離できる。
    """
    if J <= 0:
        return spk
    R, T, H = spk.shape
    idx = np.array(np.nonzero(spk))
    if idx.size == 0:
        return spk
    off = rng.randint(-J, J + 1, size=idx.shape[1])
    tt = np.clip(idx[1] + off, 0, T - 1)
    out = np.zeros_like(spk)
    out[idx[0], tt, idx[2]] = 1.0
    return out


def decoder_comparison(spk: np.ndarray, stim: np.ndarray, taps, span: int,
                       jitters, tau: float, rng=None) -> dict:
    """rate / FIR デコーダと、ジッタ掃引によるタイミング依存性の測定。

    主指標は **ジッタ耐性 J_half**: FIR デコーダの説明率が無摂動時の半分に
    落ちるジッタ幅（ステップ）。タイミング符号なら小さく、レート符号なら
    大きい。デコーダはジッタ後のデータで学習し直すので、測っているのは
    「デコーダの不整合」ではなく「摂動後に残っている情報量」である。
    """
    rng = rng or np.random.RandomState(0)
    R, T, H = spk.shape
    tgt = np.tile(stim[span - 1:], R)
    var = float(np.var(tgt))

    X_fir = _lag_matrix(spk, taps, span)
    n_tap = len(taps)
    # rate = 同じタップ上の一様平均。FIR がレートを厳密に部分集合として
    # 含む（一様重み）ので、差分は純粋に「時間カーネルの自由度」に帰属する。
    X_rate = np.stack([X_fir[:, h::H].mean(axis=1) for h in range(H)], axis=1) \
        if n_tap > 1 else X_fir
    mse_rate = _ridge_fit_eval(X_rate, tgt) / var
    mse_fir = _ridge_fit_eval(X_fir, tgt) / var

    curve = {}
    for J in jitters:
        sj = jitter_spikes(spk, int(J), rng)
        curve[int(J)] = _ridge_fit_eval(_lag_matrix(sj, taps, span), tgt) / var

    # 説明率 P(J) = 1 - mse(J) が P(0)/2 を下回る J を線形補間で求める
    p0 = max(0.0, 1.0 - mse_fir)
    j_half = float("nan")
    if p0 > 0.02 and curve:          # light-metrics ではジッタ掃引を省く
        js = sorted(curve)
        prev_j, prev_p = 0, p0
        for J in js:
            p = max(0.0, 1.0 - curve[J])
            if p <= p0 / 2:
                if prev_p > p:
                    j_half = prev_j + (prev_j - J) * (p0 / 2 - prev_p) \
                        / (prev_p - p + 1e-12)
                else:
                    j_half = float(J)
                break
            prev_j, prev_p = J, p
        else:
            j_half = float(js[-1])          # 最大ジッタでも半減しない
    return {"mse_rate": mse_rate, "mse_fir": mse_fir,
            "timing_benefit": (mse_rate - mse_fir) / (mse_rate + 1e-12),
            "jitter_curve": {str(k): v for k, v in curve.items()},
            "j_half": j_half,
            # tau で正規化した値が、速度をまたいで比較できる唯一の形
            "j_half_norm": j_half / (tau + 1e-12)}


# ============================================================
# 解析 4/5: ISI 統計と交差の起源分解
# ============================================================
def isi_stats(spk: np.ndarray, max_isi: int = 64) -> dict:
    isis = []
    R, T, H = spk.shape
    for r in range(R):
        for h in range(H):
            idx = np.flatnonzero(spk[r, :, h])
            if len(idx) > 1:
                isis.append(np.diff(idx))
    if not isis:
        return {"cv": float("nan"), "mean_isi": float("nan"),
                "hist": [], "rate": 0.0}
    isis = np.concatenate(isis).astype(np.float64)
    hist, _ = np.histogram(isis, bins=np.arange(1, max_isi + 2))
    return {"cv": float(isis.std() / (isis.mean() + 1e-12)),
            "mean_isi": float(isis.mean()),
            "hist": (hist / hist.sum()).tolist(),
            "rate": float(spk.mean())}


def crossing_origin(net: TemporalNNN, x: torch.Tensor, trials: int,
                    sigma: float = None) -> dict:
    """交差の起源分解。

    signal_only : sigma=0 の決定論的交差 (信号が ±h を横切るイベント)。
    noise_only  : **準静的**な交差 — 各時刻の動作点 d(t) を固定したまま、
                  連続 2 ステップに独立なノイズを引いたときの交差率。
                  刺激を平均値で固定する素朴な対照は動作点が偏るため使わない
                  (d(t) の分布をそのまま保つのが正しい反実仮想)。
    """
    s = net.sigma if sigma is None else sigma
    with torch.no_grad():
        xb = x.unsqueeze(0).repeat(trials, 1)
        nu_full = float((net.spikes(xb, s) > 0).float().mean())
        nu_sig = float((net.spikes(xb, sigma=0.0) > 0).float().mean())
        d = net.pre(xb)
        if s > 0:
            sa = d + s * torch.randn_like(d)
            sb = d + s * torch.randn_like(d)
            h = net.h
            cr_hi = (sa > h).float() != (sb > h).float()
            cr_lo = (sa > -h).float() != (sb > -h).float()
            nu_noise = float((cr_hi | cr_lo).float().mean())
        else:
            nu_noise = 0.0
    return {"nu_full": nu_full, "nu_signal_only": nu_sig,
            "nu_noise_only": nu_noise,
            "signal_share": nu_sig / (nu_sig + nu_noise + 1e-12)}


def gamma_stat(net: TemporalNNN, x: torch.Tensor, sigma: float) -> float:
    """ゲージ不変の制御パラメータ Gamma = |d/dt d_k(t)| / sigma。

    プリ活性の時間微分 |w_k * dx/dt| を **ユニットと時刻の同時中央値**として
    直接測る（初版は median|w| と median|dx| を別々に取っており、しかも
    学習後の w を使っていたため 1/sigma スケーリングが壊れていた — §5.1
    欠陥3）。凍結エンコーダで掃引する場合、この量は sigma に反比例する
    純粋な条件側の制御パラメータになる。
    """
    if sigma <= 0:
        return float("inf")
    with torch.no_grad():
        dd = (torch.diff(x).unsqueeze(-1) * net.w1).abs()   # [T-1, H]
        return float(dd.median() / sigma)


# ============================================================
# 1 条件の実行
# ============================================================
def refit_readout(net: TemporalNNN, x: torch.Tensor, y: torch.Tensor,
                  trials: int, sigma: float) -> float:
    """読み出し（線形）のみをその sigma で当て直したときの正規化 MSE。

    凍結エンコーダを別の sigma で駆動すると、学習時の読み出し係数は最適で
    なくなる。符号化様式の測定はスパイク列に対して行うので読み出しは本来
    無関係だが、「その sigma でのレート読み出し性能」を公平に比較するために
    リッジで当て直した値を報告する。エンコーダは一切変更しない。
    """
    with torch.no_grad():
        xb = x.unsqueeze(0).repeat(trials, 1)
        r = causal_ma(net.spikes(xb, sigma), net.window)
        X = r.reshape(-1, net.H).cpu().numpy()
        t = y.unsqueeze(0).repeat(trials, 1).reshape(-1).cpu().numpy()
    var = float(np.var(t))
    return _ridge_fit_eval(X, t) / (var + 1e-12)


def analyze(net: TemporalNNN, x: torch.Tensor, y: torch.Tensor, sigma: float,
            speed: float, args, seed: int, tag: str, tp: dict,
            losses=None) -> dict:
    """凍結したエンコーダを sigma で駆動し、符号化指標一式を測る。

    tp = timescale_params(): 全解析窓は tau_x の倍数で与えられる（項目1）。
    """
    with torch.no_grad():
        xb = x.unsqueeze(0).repeat(args.trials, 1)
        task_asis = float(F.mse_loss(
            net(xb, sigma), y.unsqueeze(0).repeat(args.trials, 1)))
        spk = (net.spikes(xb, sigma) > 0).float().cpu().numpy()

    stim = y.cpu().numpy()
    rng = np.random.RandomState(seed)
    rate = float(spk.mean())
    dead = rate < 1e-6            # 発火なし = 測定不能な条件

    rel = reliability(spk, tp["rel_widths"])
    info = info_decomposition(spk, stim, args.n_sub, tp["sub_len"],
                              tp["stride"], args.n_lvl, rng=rng)
    jit = [] if getattr(args, "light_metrics", False) else tp["jitters"]
    dec = decoder_comparison(spk, stim, tp["lag_taps"], tp["lag_span"],
                             jit, tp["tau"], rng=rng)
    isi = isi_stats(spk)
    org = crossing_origin(net, x, args.trials, sigma)

    return {"sigma": sigma, "speed": speed, "seed": seed, "mode": tag,
            "dead": bool(dead), "tau": tp["tau"],
            "win_readout": tp["window"], "win_lag_span": tp["lag_span"],
            "win_sub_len": tp["sub_len"],
            "task_mse": task_asis,
            "task_mse_refit": refit_readout(net, x, y, args.trials, sigma),
            "final_loss": float(np.mean(losses[-20:])) if losses else None,
            "gamma": gamma_stat(net, x, sigma),
            "rel": rel, "rel_fine": rel[float(tp["rel_widths"][0])],
            "rel_coarse": rel[float(tp["rel_widths"][-1])],
            # ISI も tau で正規化しないと速度をまたげない
            "isi_norm": (isi["mean_isi"] / tp["tau"]
                         if np.isfinite(isi["mean_isi"]) else float("nan")),
            **{k: v for k, v in info.items()},
            **{k: v for k, v in dec.items()},
            "cv": isi["cv"], "mean_isi": isi["mean_isi"],
            "rate": rate, "isi_hist": isi["hist"],
            **org,
            "w_abs_median": float(net.w1.detach().abs().median()),
            "bump_width": float(sigma / (net.w1.detach().abs().median()
                                         + 1e-12))}


def encoder_stats(net: TemporalNNN, sigma_train: float) -> dict:
    """獲得されたエンコーダの特徴づけ（項目7 の下ごしらえ）。

    予測（§2.2）: 低ノイズ訓練 -> 鋭いエッジ検出器（|w| 大 = バンプ幅小）、
    高ノイズ訓練 -> 段階的トランスデューサ（|w| 小 = バンプ幅大）。
    しきい値交差点 -w_k/b_k の分布が刺激レンジをどう覆うかも見る。
    """
    with torch.no_grad():
        w = net.w1.detach()
        b = net.b1.detach()
        ctr = (-b / (w + 1e-12))          # d_k = 0 となる刺激値
        inside = ((ctr > -1.0) & (ctr < 1.0)).float().mean()
        aw = w.abs()
        return {"w_abs_median": float(aw.median()),
                "w_abs_iqr": float(aw.quantile(0.75) - aw.quantile(0.25)),
                "bump_width_train": float(sigma_train / (aw.median() + 1e-12)),
                "center_in_range": float(inside),
                "center_std": float(ctr.clamp(-3, 3).std()),
                "readout_l1": float(net.a.detach().abs().sum())}


def run_matrix(speed: float, args, device, seed: int):
    """項目(6): 訓練ノイズ x 評価ノイズ の転移行列。

    sigma_train ごとにエンコーダを 1 つ学習して凍結し、その 1 つを全ての
    sigma_test で駆動する。対角は「訓練した条件での性能」、非対角は転移。
    §5.3(1) で 2 点だけ示した非対称性（高ノイズ訓練は低ノイズでも使えるが
    逆は不可）を 2 次元に拡張し、「訓練ノイズが獲得表現を決める」(§2.2) を
    一枚で示す。
    """
    x, y = make_signal(args.steps, speed, device, kind=args.signal)
    tp = timescale_params(autocorr_time(x.cpu().numpy()), args)
    print(f"    [windows] tau_x={tp['tau']:.1f} readout={tp['window']} "
          f"lag_span={tp['lag_span']} info_win={tp['info_win']}")
    rows = []
    for s_tr in args.sigma_train_list:
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = TemporalNNN(args.hidden_dim, s_tr, args.crossing_h,
                          tp["window"], device)
        losses = train(net, x, y, args.trials, args.epochs, args.lr)
        est = encoder_stats(net, s_tr)
        print(f"    [train sigma={s_tr:<6g}] loss={np.mean(losses[-20:]):.4f} "
              f"|w|={est['w_abs_median']:.2f} "
              f"bump={est['bump_width_train']:.2f} "
              f"centers_in_range={est['center_in_range']:.2f}")
        for s_te in args.sigma_test_list:
            r = analyze(net, x, y, s_te, speed, args, seed, "matrix", tp,
                        losses)
            r["sigma_train"] = s_tr
            r.update({f"enc_{k}": v for k, v in est.items()})
            rows.append(r)
            print(f"      test sigma={s_te:<6g} refitMSE={r['task_mse_refit']:.3f}"
                  f" rel={r['rel_fine']:.3f} timfr={r['timing_frac']:.3f}"
                  f" CV={r['cv']:.2f} sig_sh={r['signal_share']:.2f}")
    return rows


def run_train_sweep(sigma: float, speed: float, args, device,
                    seed: int) -> dict:
    """各 sigma で新規に学習してから測る（学習の効果を含む = 実験B）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, y = make_signal(args.steps, speed, device, kind=args.signal)
    tp = timescale_params(autocorr_time(x.cpu().numpy()), args)
    net = TemporalNNN(args.hidden_dim, sigma, args.crossing_h, tp["window"],
                      device)
    losses = train(net, x, y, args.trials, args.epochs, args.lr,
                   log_every=args.epochs // 2 if args.verbose else 0)
    return analyze(net, x, y, sigma, speed, args, seed, "train", tp, losses)


def run_test_sweep(speed: float, args, device, seed: int):
    """学習は sigma_train で 1 回だけ行い、凍結して sigma を掃引する（実験A）。

    examples/regression_noiseless_result.py の流儀（学習は std>0、評価は
    stds=[0,0]）をそのまま制御実験に使う。エンコーダが全条件で同一なので、
    符号化様式の変化は純粋に sigma（= Gamma）だけに帰属する。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, y = make_signal(args.steps, speed, device, kind=args.signal)
    tp = timescale_params(autocorr_time(x.cpu().numpy()), args)
    warn = (f"  [WARN] rel 幅 {tp['rel_floor_hit']} 個が 1 ステップの床に接触"
            if tp.get("rel_floor_hit") else "")
    print(f"    [windows] tau_x={tp['tau']:.1f} -> readout={tp['window']}, "
          f"lag_span={tp['lag_span']} ({len(tp['lag_taps'])} taps), "
          f"info_win={tp.get('info_win')} (sub_len={tp['sub_len']}), "
          f"jitters={tp['jitters']}{warn}")
    net = TemporalNNN(args.hidden_dim, args.sigma_train, args.crossing_h,
                      tp["window"], device)
    losses = train(net, x, y, args.trials, args.epochs, args.lr,
                   log_every=args.epochs // 2 if args.verbose else 0)
    print(f"    [encoder] trained at sigma={args.sigma_train:g}: "
          f"loss={np.mean(losses[-20:]):.4f}, "
          f"|w| median={float(net.w1.detach().abs().median()):.3f}")
    rows = []
    for sigma in [float(v) for v in args.sigmas.split(",")]:
        r = analyze(net, x, y, sigma, speed, args, seed, "test", tp, losses)
        rows.append(r)
        print(f"    sigma={sigma:<6g} Gamma={r['gamma']:7.2f} "
              f"rate={r['rate']:.3f} refitMSE={r['task_mse_refit']:.3f} "
              f"rel(fine)={r['rel_fine']:.3f} "
              f"timing_frac={r['timing_frac']:.3f} "
              f"J1/2={r['j_half']:.1f} ({r['j_half_norm']:.2f}tau) "
              f"CV={r['cv']:.2f} sig_share={r['signal_share']:.2f}")
    return rows


# ============================================================
# 図
# ============================================================
def make_figures(rows, args):
    sig = np.array([r["sigma"] for r in rows])
    order = np.argsort(sig)

    # Fig 1: 符号化様式の指標 vs sigma
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for speed in sorted({r["speed"] for r in rows}):
        sel = [r for r in rows if r["speed"] == speed]
        sel.sort(key=lambda r: r["sigma"])
        s = [r["sigma"] for r in sel]
        lab = f"speed={speed}"
        axes[0, 0].plot(s, [r["rel_fine"] for r in sel], "-o", label=lab)
        axes[0, 1].plot(s, [r["timing_frac"] for r in sel], "-o", label=lab)
        axes[1, 0].plot(s, [r["j_half_norm"] for r in sel], "-o", label=lab)
        axes[1, 1].plot(s, [r["cv"] for r in sel], "-o", label=lab)
    for ax, ttl, yl in [
            (axes[0, 0], "trial-to-trial reliability (fine)", "corr"),
            (axes[0, 1], "info fraction carried by timing", "gain / I(pattern)"),
            (axes[1, 0], "jitter tolerance J(1/2) / tau", "tau units"),
            (axes[1, 1], "ISI coefficient of variation", "CV")]:
        ax.set_xlabel("noise std sigma")
        ax.set_ylabel(yl)
        ax.set_title(ttl)
        ax.set_xscale("log")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.suptitle("Coding regime vs injected noise")
    fig.tight_layout()
    savefig(fig, args.out_dir / "fig1_regime_vs_sigma.png")

    # Fig 2: Gamma collapse (2 つの物理ノブが 1 軸に乗るか)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for speed in sorted({r["speed"] for r in rows}):
        sel = [r for r in rows if r["speed"] == speed and
               np.isfinite(r["gamma"])]
        sel.sort(key=lambda r: r["gamma"])
        g = [r["gamma"] for r in sel]
        axes[0].plot(g, [r["rel_fine"] for r in sel], "-o",
                     label=f"speed={speed}")
        axes[1].plot(g, [r["timing_frac"] for r in sel], "-o")
        axes[2].plot(g, [r["j_half_norm"] for r in sel], "-o")
    for ax, ttl in [(axes[0], "reliability (fine)"),
                    (axes[1], "timing info fraction"),
                    (axes[2], "jitter tolerance J(1/2) / tau")]:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\Gamma = |w\,\dot x|/\sigma$")
        ax.set_title(ttl)
        ax.axvline(1.0, color="k", ls=":", lw=1)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Gauge-invariant collapse: two knobs, one parameter")
    fig.tight_layout()
    savefig(fig, args.out_dir / "fig2_gamma_collapse.png")

    # Fig 3: ISI ヒストグラム (低/中/高ノイズ)
    sel = [rows[i] for i in order]
    picks = [sel[0], sel[len(sel) // 2], sel[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, r in zip(axes, picks):
        hh = np.array(r["isi_hist"])
        if hh.size:
            ax.bar(np.arange(1, len(hh) + 1), hh, width=1.0)
        ax.set_title(f"sigma={r['sigma']:.3g}  CV={r['cv']:.2f}")
        ax.set_xlabel("ISI [steps]")
        ax.set_ylabel("prob")
    fig.suptitle("ISI distribution: locked events -> Poisson-like")
    fig.tight_layout()
    savefig(fig, args.out_dir / "fig3_isi.png")


def _matrix_of(rows, metric, trains, tests):
    """[len(trains), len(tests)] の行列（seed 平均）。"""
    M = np.full((len(trains), len(tests)), np.nan)
    for i, tr in enumerate(trains):
        for j, te in enumerate(tests):
            v = [r[metric] for r in rows
                 if r.get("sigma_train") == tr and r["sigma"] == te
                 and r.get(metric) is not None and np.isfinite(r[metric])]
            if v:
                M[i, j] = float(np.mean(v))
    return M


def matrix_tables(rows, args):
    """項目(6): 転移行列の表と図。行 = 訓練ノイズ、列 = 評価ノイズ。"""
    mr = [r for r in rows if r["mode"] == "matrix"]
    trains = sorted({r["sigma_train"] for r in mr})
    tests = sorted({r["sigma"] for r in mr})
    out = ["**転移行列** (行 = 訓練 sigma、列 = 評価 sigma；値 = 読み出し当て直し"
           "後の正規化 MSE。低いほど良い。太字 = 対角)", "",
           "| train \\ test | " + " | ".join(f"{t:g}" for t in tests) + " | 行平均 |",
           "|---" * (len(tests) + 2) + "|"]
    M = _matrix_of(mr, "task_mse_refit", trains, tests)
    for i, tr in enumerate(trains):
        cells = []
        for j, te in enumerate(tests):
            v = M[i, j]
            s = "—" if not np.isfinite(v) else (
                f"**{v:.3f}**" if abs(tr - te) < 1e-12 else f"{v:.3f}")
            cells.append(s)
        out.append(f"| {tr:g} | " + " | ".join(cells)
                   + f" | {np.nanmean(M[i]):.3f} |")

    # 転移の非対称性: 上三角 (train < test) と下三角 (train > test) の平均
    iu = np.triu_indices(len(trains), k=1)
    il = np.tril_indices(len(trains), k=-1)
    if len(trains) == len(tests):
        out += ["", f"転移の非対称性: 低ノイズ訓練→高ノイズ評価 "
                    f"{np.nanmean(M[iu]):.3f} / 高ノイズ訓練→低ノイズ評価 "
                    f"{np.nanmean(M[il]):.3f}（低いほど良い）"]

    # エンコーダの特徴づけ（項目7 の下ごしらえ）
    out += ["", "**獲得されたエンコーダ** (訓練 sigma ごと)", "",
            "| train sigma | median |w| | bump 幅 sigma/|w| | 交差点が範囲内 | "
            "readout L1 |", "|---" * 5 + "|"]
    for tr in trains:
        v = [r for r in mr if r["sigma_train"] == tr]
        g = lambda k: float(np.mean([q[k] for q in v]))  # noqa: E731
        out.append(f"| {tr:g} | {g('enc_w_abs_median'):.3f} "
                   f"| {g('enc_bump_width_train'):.3f} "
                   f"| {g('enc_center_in_range'):.2f} "
                   f"| {g('enc_readout_l1'):.2f} |")

    # ---- 図 ----
    metrics = [("task_mse_refit", "refit MSE (low=good)", "viridis_r"),
               ("timing_frac", "timing info fraction", "magma"),
               ("cv", "ISI CV", "magma"),
               ("signal_share", "signal-driven share", "magma")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.8))
    for ax, (key, ttl, cmap) in zip(np.atleast_1d(axes), metrics):
        Mk = _matrix_of(mr, key, trains, tests)
        im = ax.imshow(Mk, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(tests)))
        ax.set_xticklabels([f"{t:g}" for t in tests], rotation=45, fontsize=7)
        ax.set_yticks(range(len(trains)))
        ax.set_yticklabels([f"{t:g}" for t in trains], fontsize=7)
        ax.set_xlabel("test sigma")
        ax.set_ylabel("train sigma")
        ax.set_title(ttl, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=.046)
    fig.suptitle("Transfer matrix: training noise determines what is learned")
    fig.tight_layout()
    savefig(fig, args.out_dir / "fig4_transfer_matrix.png")
    return out


# ============================================================
def main():
    p = argparse.ArgumentParser(description="spike coding regime vs noise")
    p.add_argument("--sigmas", type=str,
                   default="0.005,0.02,0.05,0.1,0.2,0.4,0.8")
    p.add_argument("--speeds", type=str, default="1.0")
    p.add_argument("--signal", choices=("sine", "rw"), default="sine")
    p.add_argument("--steps", type=int, default=1024, help="時系列長 T")
    p.add_argument("--trials", type=int, default=24, help="試行数 R")
    p.add_argument("--window", type=int, default=16,
                   help="読み出し窓（--window-mode abs のときのみ使用）")
    p.add_argument("--n-lag", type=int, default=16,
                   help="FIR ラグ span（--window-mode abs のときのみ）")
    p.add_argument("--n-sub", type=int, default=6, help="情報分解のサブビン数")
    p.add_argument("--sub-len", type=int, default=3)
    p.add_argument("--stride", type=int, default=6)
    # --- 項目1: 全解析窓を tau_x の倍数で指定する ---
    p.add_argument("--window-mode", choices=("tau", "abs"), default="tau",
                   help="tau=信号相関時間で正規化（既定）/ abs=旧来の絶対ステップ")
    p.add_argument("--window-frac", type=float, default=0.5,
                   help="読み出し移動平均の幅 / tau_x")
    p.add_argument("--span-frac", type=float, default=1.0,
                   help="FIR・rate デコーダの時間スパン / tau_x")
    p.add_argument("--info-frac", type=float, default=0.25,
                   help="情報分解の窓幅 / tau_x。窓内で刺激が準静的である必要が"
                        "あるので 1.0 では長すぎる（§5.5 課題2）")
    p.add_argument("--lag-taps", type=int, default=12,
                   help="FIR のタップ数（span によらず固定 = 表現力を揃える）")
    p.add_argument("--jitter-fracs", type=str,
                   default="0.02,0.05,0.1,0.2,0.4,0.8,1.5",
                   help="ジッタ幅 / tau_x")
    p.add_argument("--rel-fracs", type=str, default="0.05,0.1,0.2,0.35,0.6",
                   help="試行間再現性の平滑化幅 / tau_x。最小値は全速度で 1 "
                        "ステップを超えるよう取る（§5.5 課題3 の床対策）")
    p.add_argument("--n-lvl", type=int, default=6, help="刺激の離散化段数")
    p.add_argument("--sigma-trains", type=str,
                   default="0.02,0.05,0.1,0.2,0.4,0.8",
                   help="matrix モードの訓練ノイズ列")
    p.add_argument("--light-metrics", action="store_true",
                   help="ジッタ掃引を省いて高速化（転移行列など点数が多い実験用）")
    p.add_argument("--probe-mode", choices=("test", "train", "both", "matrix"),
                   default="test",
                   help="test=凍結エンコーダで sigma 掃引 (実験A) / "
                        "train=各 sigma で新規学習 (実験B)")
    p.add_argument("--sigma-train", type=float, default=0.3,
                   help="test モードでエンコーダを学習するノイズ強度")
    p.add_argument("--jitters", type=str, default="1,2,4,8,16,32")
    p.add_argument("--verbose", action="store_true")
    fncl.add_common_args(p, epochs=400, hidden_dim=16, seeds="0")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/spike_coding_regime")
    if args.quick:
        args.steps, args.trials, args.epochs = 384, 8, 120
        args.sigmas = "0.01,0.1,0.8"
    args.rel_widths = [1.0, 2.0, 4.0, 8.0, 16.0]
    args.jitter_list = [int(v) for v in args.jitters.split(",")]
    args.jitter_fracs = [float(v) for v in args.jitter_fracs.split(",")]
    args.rel_fracs = [float(v) for v in args.rel_fracs.split(",")]
    device = torch.device(args.device)

    sigmas = [float(v) for v in args.sigmas.split(",")]
    speeds = [float(v) for v in args.speeds.split(",")]
    modes = ["test", "train"] if args.probe_mode == "both" else [args.probe_mode]
    args.sigma_train_list = [float(v) for v in args.sigma_trains.split(",")]
    args.sigma_test_list = sigmas

    rows = []
    for mode in modes:
        for speed in speeds:
            for seed in args.seed_list:
                if mode == "matrix":
                    print(f"\n  ===== [6: transfer matrix] speed={speed:g} "
                          f"seed={seed} =====")
                    rows += run_matrix(speed, args, device, seed)
                elif mode == "test":
                    print(f"\n  ===== [A: frozen encoder] speed={speed:g} "
                          f"seed={seed} (trained at sigma="
                          f"{args.sigma_train:g}) =====")
                    rows += run_test_sweep(speed, args, device, seed)
                else:
                    for sigma in sigmas:
                        print(f"\n  ===== [B: train sweep] sigma={sigma:g} "
                              f"speed={speed:g} seed={seed} =====")
                        r = run_train_sweep(sigma, speed, args, device, seed)
                        rows.append(r)
                        print(f"    task MSE={r['task_mse']:.4f} "
                              f"Gamma={r['gamma']:.2f} "
                              f"rel(fine)={r['rel_fine']:.3f} "
                              f"timing_frac={r['timing_frac']:.3f} "
                              f"J1/2={r['j_half']:.1f} CV={r['cv']:.2f} "
                              f"sig_share={r['signal_share']:.2f}")

    # ---- 表 ----
    lines = [f"**符号化様式 vs ノイズ強度** (H={args.hidden_dim}, T={args.steps}, "
             f"R={args.trials}, h={args.crossing_h}, window={args.window}, "
             f"signal={args.signal}, epochs={args.epochs}, "
             f"sigma_train={args.sigma_train:g}, "
             f"window_mode={args.window_mode})", "",
             "| mode | sigma | speed | tau | Gamma | refit MSE | rate | "
             "rel(fine) | rel(coarse) | timing frac | J(1/2)/tau | rate dec | "
             "FIR dec | CV | sig share |",
             "|---" * 15 + "|"]
    for r in rows:
        flag = " (dead)" if r.get("dead") else ""
        lines.append(
            f"| {r['mode']}{flag} | {r['sigma']:g} | {r['speed']:g} "
            f"| {r['tau']:.0f} | {r['gamma']:.2f} | {r['task_mse_refit']:.3f} "
            f"| {r['rate']:.3f} | {r['rel_fine']:.3f} "
            f"| {r['rel_coarse']:.3f} | {r['timing_frac']:.3f} "
            f"| {r['j_half_norm']:.3f} | {r['mse_rate']:.3f} "
            f"| {r['mse_fir']:.3f} | {r['cv']:.2f} | {r['signal_share']:.2f} |")
    if any(r["mode"] == "matrix" for r in rows):
        lines += [""] + matrix_tables(rows, args)
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_coding_regime.md", table)
    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args), "rows": rows})
    make_figures(rows, args)


if __name__ == "__main__":
    main()
