"""
fncl_phase1_fxp.py — FPGA 実装 Phase 1「固定小数点ゴールデンモデル」

Phase 0 で凍結した構成 (cov_jac + sgdm(1-2^-4) + lr 2^-10 + cosine ROM +
jac_track, B=1 提示ごと更新, 一様ノイズ) を、RTL にそのまま写せる形で再現する。

  Stage A (--stage stream): ストリーミング統計 (running sum) + 同一パス slope
      の float 版。Phase 0 の B1 結果と同水準かを確認し、mirror EMA の
      シフト定数 (2^-k) を凍結する。
  Stage B (--stage lfsr):   一様ノイズを LFSR 由来に置換 (float 演算のまま)。
      配線 (phase: ユニット毎 LFSR / leap: 1 本を時分割) とビット幅 (16/24)
      を比較し、系列間相関が cov_weight を壊さないかを確認する。
  Stage C (--stage fxp):    全データパスを int64 上の固定小数点で実装し、
      ビット幅 (重み Fw / 前活性 Fd / ノイズ bits) を掃引。必要語長を実測する。

アルゴリズム上の変更点 (Phase 0 float 版との差分; いずれも Stage A で検証):
  - KDE slope をノイズ再抽選の再パスでなく同一 forward の xor カウントから取る
    (追加パス 1 本の削減)。
  - mirror EMA 率を 1-0.9^(1/128) ≈ 8.2e-4 から 2^-k (シフト 1 回) に丸める。
  - T = 2^6 なので Cov/Var は整数比: W_hat = (T*Σdz - Σd*Σz) / (T*Σz² - (Σz)²)。
    除算はユニットあたり 1 回 (分母は列共有)。

判定: LFSR + 固定小数点の B=1 学習が Phase 0 batch_ref (0.00092) の x2 以内。

実行例:
  python fncl_phase1_fxp.py --quick                    # 動作確認
  python fncl_phase1_fxp.py --stage stream,lfsr        # Stage A+B
  python fncl_phase1_fxp.py --stage fxp                # Stage C (A/B の後で)
"""
import argparse
import math
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 保存専用 (fncl_driver が pyplot を import する前に設定)
import matplotlib.pyplot as plt  # noqa: E402

import fncl_driver as fncl  # noqa: E402
from fncl_driver import (add_common_args, finalize_args, make_task,  # noqa: E402
                         model_factory, config_dict, write_text, save_json,
                         savefig)

# Phase 0 実験②の batch_ref (uniform, H=64, T=64, budget=192k, seeds 0-2)
BATCH_REF = 0.00092

# Phase 2 語長契約: fxp_min_14_10_8 (Fw14/Fd10/Nn8) の実測語長 +1 bit 以上。
# RTL はこの幅の飽和演算で実装する (--stage fxpsat がこの契約を検証する)。
SAT_WIDTHS = {"W": 18, "d": 16, "ys": 16, "credit": 14, "slope": 13,
              "v": 18, "M_store": 20, "Num": 24}


def sat(x, bits: int):
    """語長契約への飽和 (符号込み bits ビット)."""
    lim = 1 << (bits - 1)
    return torch.clamp(x, -lim, lim - 1)


# ============================================================
# LFSR (Galois): 倍加法で任意長の系列を一括生成する
# ============================================================
LFSR_TAPS = {16: 0xB400, 24: 0xE10000}   # x^16+x^14+x^13+x^11+1 / x^24+x^23+x^22+x^17+1


def _lfsr_step(s: np.ndarray, taps: int) -> np.ndarray:
    """Galois LFSR を 1 ステップ進める (要素ごと・ベクトル化)."""
    lsb = s & 1
    return (s >> 1) ^ (lsb * taps)


def lfsr_sequence(nbits: int, length: int, seed: int = 1) -> torch.Tensor:
    """状態系列 seq[0:length] (int64, 値域 [1, 2^nbits))。

    seq[t+K] = M^K(seq[t]) が要素ごとのビット XOR で書けることを使い、
    「先頭 64 個を逐次生成 -> 以降は配列全体に M^len を適用して倍々に延長」
    で O(length * nbits / 64) 語演算に落とす (24 bit の全周期 16.7M も数秒)。
    """
    taps = LFSR_TAPS[nbits]
    n0 = min(64, length)
    seq = np.empty(n0, dtype=np.int64)
    s = np.int64(seed & ((1 << nbits) - 1)) or np.int64(1)
    for i in range(n0):
        seq[i] = s
        s = _lfsr_step(s, taps)
    # 状態更新は GF(2) 上の線形写像 M。seq[t+K] = M^K(seq[t]) を
    # 「M^K の列マスクを配列全体にビット XOR で適用」して倍々に延長する。
    while len(seq) < length:
        K = len(seq)
        colmask = _advance_masks(nbits, K, taps)
        nxt = np.zeros_like(seq)
        for b in range(nbits):
            nxt ^= ((seq >> b) & 1) * colmask[b]
        seq = np.concatenate([seq, nxt])
    return torch.from_numpy(seq[:length].copy())


_ADV_CACHE = {}


def _advance_masks(nbits: int, K: int, taps: int) -> np.ndarray:
    """M^K の列マスク: colmask[b] = M^K(e_b)。二乗法で構成しキャッシュする."""
    key = (nbits, K, taps)
    if key in _ADV_CACHE:
        return _ADV_CACHE[key]
    # M^1 の列マスク
    base = np.array([_lfsr_step(np.int64(1 << b), taps) for b in range(nbits)],
                    dtype=np.int64)

    def compose(A, B):
        """(A∘B)(e_b) = A(B(e_b)) を列マスクで."""
        out = np.zeros(nbits, dtype=np.int64)
        for b in range(nbits):
            v = B[b]
            r = np.int64(0)
            for bb in range(nbits):
                if (v >> bb) & 1:
                    r ^= A[bb]
            out[b] = r
        return out

    result = None
    P = base
    k = K
    while k:
        if k & 1:
            result = P.copy() if result is None else compose(P, result)
        P = compose(P, P)
        k >>= 1
    _ADV_CACHE[key] = result
    return result


class NoiseSource:
    """一様ノイズ源。kind: rand | phase16 | leap16 | phase24 | leap24。

    draw(layer, n) -> [n, T, H] の float 値 (~Uniform(-1, 1))。
    draw_code(layer, n, k) -> [n, T, H] の k ビット一様整数 (固定小数点用)。
    fork() は独立カウンタの複製 (評価パス用; 学習ストリームを進めない)。
    """

    def __init__(self, kind: str, T: int, H: int, n_layers: int, seed: int):
        self.kind, self.T, self.H, self.L = kind, T, H, n_layers
        if kind == "rand":
            self.gen = torch.Generator().manual_seed(seed * 7919 + 13)
            return
        nbits = int(kind[-2:])
        self.nbits = nbits
        self.P = (1 << nbits) - 1
        self.seq = _SEQ_CACHE.setdefault(nbits,
                                         lfsr_sequence(nbits, self.P))
        self.cnt = [0] * n_layers
        if kind.startswith("phase"):
            stride = int(self.P * 0.6180339887) | 1
            base = (seed * 104729) % self.P
            self.off = [((base + (l * H + torch.arange(H)) * stride) % self.P)
                        for l in range(n_layers)]
        else:                                   # leap: 1 本を時分割
            self.off = [torch.tensor([(seed * 104729 + l * (self.P // n_layers))
                                      % self.P]) for l in range(n_layers)]

    def fork(self, tag: int = 1):
        src = NoiseSource.__new__(NoiseSource)
        src.__dict__.update(self.__dict__)
        if self.kind == "rand":
            src.gen = torch.Generator().manual_seed(
                int(torch.randint(0, 2**31, (1,), generator=self.gen)) + tag)
        else:
            src.cnt = [c + (self.P // 2) + tag * 65537 for c in self.cnt]
        return src

    def _states(self, layer: int, n: int) -> torch.Tensor:
        T, H = self.T, self.H
        if self.kind.startswith("phase"):
            t_idx = self.cnt[layer] + torch.arange(n * T).view(n, T, 1)
            idx = (self.off[layer].view(1, 1, H) + t_idx) % self.P
            self.cnt[layer] += n * T
        else:
            flat = self.cnt[layer] + torch.arange(n * T * H).view(n, T, H)
            idx = (self.off[layer].view(1, 1, 1) + flat) % self.P
            self.cnt[layer] += n * T * H
        return self.seq[idx]

    def draw(self, layer: int, n: int = 1) -> torch.Tensor:
        if self.kind == "rand":
            return torch.rand(n, self.T, self.H, generator=self.gen,
                              dtype=torch.float64) * 2.0 - 1.0
        s = self._states(layer, n)
        return (2.0 * s.to(torch.float64) + 1.0 - 2.0 ** self.nbits) / 2.0 ** self.nbits

    def draw_code(self, layer: int, n: int, k: int) -> torch.Tensor:
        """k ビット一様整数 [0, 2^k) (LFSR 状態の上位 k ビット)."""
        if self.kind == "rand":
            u = torch.rand(n, self.T, self.H, generator=self.gen,
                           dtype=torch.float64)
            return (u * (1 << k)).long().clamp_(0, (1 << k) - 1)
        return self._states(layer, n) >> (self.nbits - k)


_SEQ_CACHE = {}


# ============================================================
# 交差活性 (ストリーミング形): 2 値化 -> 巡回 XOR -> code {0,1,2}
# ============================================================
def crossing_code(dn: torch.Tensor, h):
    """dn [..., T, H] -> (code [..., T, H] ∈ {0,1,2}, cdiff [..., H]).

    code = xor1 + xor2 (z = code/2)。cdiff = Σ_t (xor2 - xor1) は同一パスの
    KDE slope 分子 (HW では forward と同時にカウントできる)。
    """
    b1 = dn > h
    b2 = dn > -h
    x1 = (torch.roll(b1, -1, dims=-2) ^ b1).long()
    x2 = (torch.roll(b2, -1, dims=-2) ^ b2).long()
    return x1 + x2, (x2 - x1).sum(dim=-2)


def imatmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """整数行列積 (int64)。値域が 2^53 に収まる前提で float64 経由 (正確)."""
    return (a.to(torch.float64) @ b.to(torch.float64)).long()


def rshift(x, s: int):
    """丸め付き算術右シフト (s<=0 は左シフト)。HW: 加算 1 + シフト."""
    if s <= 0:
        return x << (-s)
    return (x + (1 << (s - 1))) >> s


def rdiv(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """丸め付き整数除算 (den > 0)。HW: ユニットあたり 1 除算."""
    return torch.div(2 * num + den, 2 * den, rounding_mode="floor")


# ============================================================
# float ストリーミング版 (Stage A / B)
# ============================================================
def extract_params(net):
    p = {}
    for i, name in enumerate(["0", "1", "out"]):
        p[f"W{name}"] = net.fcs[i].weight.detach().double().clone()
        p[f"b{name}"] = net.fcs[i].bias.detach().double().clone()
    return p


def predict_float(p, x_q, src, h, T, passes=8):
    """8-pass 予測 (float ストリーミング datapath)."""
    N = x_q.shape[0]
    y = torch.zeros(N, dtype=torch.float64)
    for _ in range(passes):
        d0 = x_q @ p["W0"].T + p["b0"]                       # [N, H]
        code0, _ = crossing_code(d0.unsqueeze(1) + src.draw(0, N), h)
        z0 = code0.double() / 2.0
        d1 = z0 @ p["W1"].T + p["b1"]
        code1, _ = crossing_code(d1 + src.draw(1, N), h)
        z1 = code1.double() / 2.0
        y += (z1 @ p["Wout"].T + p["bout"]).mean(dim=1).squeeze(-1)
    return (y / passes).numpy()


def train_stream_float(p, x_all, t_all, args, cfg, src):
    """B=1 オンライン学習 (float, running-sum 統計, 同一パス slope)."""
    T, H = args.num_samples, args.hidden_dim
    h = args.crossing_h
    steps = args.budget
    lr0 = args.lr / x_all.shape[0]                           # 線形則 lr*B/128
    ema_a = (1.0 - 0.9 ** (1.0 / x_all.shape[0]) if cfg["ema"] == "exact"
             else 2.0 ** (-cfg["ema"]))
    mu = args.momentum
    N_all = x_all.shape[0]
    eval_every = max(1, steps // 50)
    eps_c = 4.0 * T * T * 1e-6                               # cov_weight の eps 相当

    M1, Mout = None, None
    vel = {k: torch.zeros_like(v) for k, v in p.items()}
    curve = []
    pos, direction = 0, 1
    for step in range(steps):
        lr_t = lr0 * 0.5 * (1.0 + math.cos(math.pi * step / steps))
        if cfg["order"] == "iid":
            i = np.random.randint(N_all)
        else:
            i = pos
            pos += direction
            if pos in (0, N_all - 1):
                direction = -direction
        x, t = x_all[i], float(t_all[i])

        # ---- forward (T サンプルを流しながら running sum を積む) ----
        d0 = (p["W0"] @ x + p["b0"])                          # [H]
        code0, cdiff0 = crossing_code(d0.unsqueeze(0) + src.draw(0)[0], h)
        slope0 = cdiff0.double() / (2.0 * h * T)              # [H]
        d1 = (code0.double() / 2.0) @ p["W1"].T + p["b1"]     # [T, H]
        code1, cdiff1 = crossing_code(d1 + src.draw(1)[0], h)
        slope1 = cdiff1.double() / (2.0 * h * T)
        ys = ((code1.double() / 2.0) @ p["Wout"].T + p["bout"]).squeeze(-1)  # [T]
        y = float(ys.mean())

        # ---- mirror 測定 (整数比の float 版) + EMA ----
        Sz0, Szz0 = code0.sum(0).double(), (code0 * code0).sum(0).double()
        Sz1, Szz1 = code1.sum(0).double(), (code1 * code1).sum(0).double()
        Sd1 = d1.sum(0)
        Sdz1 = d1.T @ code0.double()
        num1 = T * Sdz1 - torch.outer(Sd1, Sz0)
        den1 = T * Szz0 - Sz0 * Sz0
        W1_hat = 2.0 * num1 / (den1 + eps_c)
        Sy = float(ys.sum())
        Syz = ys @ code1.double()
        num_o = T * Syz - Sy * Sz1
        den_o = T * Szz1 - Sz1 * Sz1
        Wo_hat = (2.0 * num_o / (den_o + eps_c)).unsqueeze(0)  # [1, H]
        if M1 is None:
            M1, Mout = W1_hat.clone(), Wo_hat.clone()
        else:
            M1 += ema_a * (W1_hat - M1)
            Mout += ema_a * (Wo_hat - Mout)

        # ---- 再帰 credit -> 勾配 -> sgdm 更新 + KP 追跡 ----
        e = 2.0 * (y - t)
        a1 = e * Mout.squeeze(0)                              # [H]
        dd1 = a1 * slope1
        a0 = M1.T @ dd1
        a0s = a0 * slope0
        zbar0 = Sz0 / (2.0 * T)
        zbar1 = Sz1 / (2.0 * T)
        grads = {"W1": torch.outer(dd1, zbar0), "b1": dd1,
                 "W0": torch.outer(a0s, x), "b0": a0s,
                 "Wout": (e * zbar1).unsqueeze(0),
                 "bout": torch.tensor([e], dtype=torch.float64)}
        for k, g in grads.items():
            vel[k] = mu * vel[k] + g
            step_k = lr_t * vel[k]
            p[k] -= step_k
            if k == "W1":
                M1 -= step_k
            elif k == "Wout":
                Mout -= step_k

        if step % eval_every == 0 or step == steps - 1:
            pred = predict_float(p, x_all, src.fork(step), h, T)
            mse = float(np.mean((pred - t_all.numpy().ravel()) ** 2))
            curve.append((step, mse))

    mirror_r = {"out": _corr(Mout, p["Wout"]), "W1": _corr(M1, p["W1"])}
    return curve, mirror_r


def _corr(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


# ============================================================
# 固定小数点版 (Stage C): int64 上の正確なビット演算
# ============================================================
class Widths:
    """データパス各量の最大絶対値を追跡し、必要語長を報告する."""

    def __init__(self):
        self.m = {}

    def track(self, name, x):
        v = int(x.abs().max()) if torch.is_tensor(x) else abs(int(x))
        if v > self.m.get(name, 0):
            self.m[name] = v

    def report(self, fracs: dict) -> str:
        out = []
        for name, mx in self.m.items():
            bits = max(1, mx).bit_length() + 1                # 符号込み
            out.append(f"{name}:{bits}b(f{fracs.get(name, '?')})")
        return " ".join(out)


def quant_params(p, Fw):
    return {k: torch.round(v * (1 << Fw)).long() for k, v in p.items()}


def _forward_fxp(pq, xq_n, src, cfg, n):
    """[n] 入力の 1 パス forward。返り値は code0, code1, d1q, ysq, yq など."""
    Fw, Fd, Nn = cfg["Fw"], cfg["Fd"], cfg["Nn"]
    T = cfg["T"]
    hq = cfg["hq"]
    S = (lambda x, k: sat(x, SAT_WIDTHS[k])) if cfg.get("sat") else (lambda x, k: x)
    d0 = rshift(imatmul(pq["W0"], xq_n.T), Fw).T + rshift(pq["b0"], Fw - Fd)  # [n,H] (Fd)
    d0 = S(d0, "d")
    nz0 = (2 * src.draw_code(0, n, Nn) + 1 - (1 << Nn)) << (Fd - Nn)
    code0, cdiff0 = crossing_code(d0.unsqueeze(1) + nz0, hq)
    d1 = rshift(imatmul(code0, pq["W1"].T), 1 + Fw - Fd) \
        + rshift(pq["b1"], Fw - Fd)                                       # [n,T,H]
    d1 = S(d1, "d")
    nz1 = (2 * src.draw_code(1, n, Nn) + 1 - (1 << Nn)) << (Fd - Nn)
    code1, cdiff1 = crossing_code(d1 + nz1, hq)
    ys = rshift(imatmul(code1, pq["Wout"].T), 1 + Fw - Fd).squeeze(-1) \
        + rshift(pq["bout"], Fw - Fd)                                     # [n,T]
    ys = S(ys, "ys")
    y = rshift(ys.sum(dim=-1), int(math.log2(T)))                         # [n] (Fd)
    return d0, code0, cdiff0, d1, code1, cdiff1, ys, y


def predict_fxp(pq, xq_all, src, cfg, passes=8):
    acc = torch.zeros(xq_all.shape[0], dtype=torch.int64)
    for _ in range(passes):
        acc += _forward_fxp(pq, xq_all, src, cfg, xq_all.shape[0])[-1]
    return (acc.double() / passes / (1 << cfg["Fd"])).numpy()


def fxp_prepare(p, args, cfg):
    """固定小数点学習の状態辞書と完全化した cfg を作る (RTL テストベンチ兼用)."""
    T = args.num_samples
    Fd = cfg["Fd"]
    hq = int(round(args.crossing_h * (1 << Fd)))
    # slope = rdiv(cdiff << 2Fd, 2*hq*T): 丸め付き整数除算のみで HW 再現可能
    cfg = dict(cfg, T=T, hq=hq, logT=int(math.log2(T)),
               den_slope=2 * hq * T)
    state = {"pq": quant_params(p, cfg["Fw"]),
             "vel": {k: torch.zeros_like(v)
                     for k, v in quant_params(p, cfg["Fw"]).items()},
             "M1s": None, "Mos": None}
    return state, cfg


def fxp_step(state, xq, tq: int, rom_q: int, src, cfg, wd: Widths = None):
    """1 提示ぶんの学習ステップ (forward -> mirror -> credit -> 更新 + KP)。

    state (pq/vel/M1s/Mos) を in-place に進め、RTL 突き合わせ用の中間量
    トレースを返す。train_stream_fxp はこれを budget 回呼ぶだけ。
    """
    Fw, Fd, Ge, Gg = cfg["Fw"], cfg["Fd"], cfg["Ge"], cfg["Gg"]
    Fg = Fd + Gg
    k_ema, lr_shift, logT = cfg["k_ema"], cfg["lr_shift"], cfg["logT"]
    pq, vel = state["pq"], state["vel"]

    d0, code0, cdiff0, d1, code1, cdiff1, ys, y = \
        _forward_fxp(pq, xq, src, cfg, 1)
    d0, code0, cdiff0 = d0[0], code0[0], cdiff0[0]
    d1, code1, cdiff1, ys = d1[0], code1[0], cdiff1[0], ys[0]
    y = int(y[0])
    S = (lambda x, k: sat(x, SAT_WIDTHS[k])) if cfg.get("sat") else (lambda x, k: x)
    dsl = torch.tensor(cfg["den_slope"])
    slope0 = S(rdiv(cdiff0 << (2 * Fd), dsl), "slope")        # [H] (Fd)
    slope1 = S(rdiv(cdiff1 << (2 * Fd), dsl), "slope")

    # ---- mirror: 整数比 (T*Σdz - Σd*Σz) / (T*Σz² - (Σz)²) ----
    Sz0, Szz0 = code0.sum(0), (code0 * code0).sum(0)
    Sz1, Szz1 = code1.sum(0), (code1 * code1).sum(0)
    Sd1 = d1.sum(0)
    Sdz1 = imatmul(d1.T, code0)
    num1 = S((Sdz1 << logT) - Sd1.unsqueeze(1) * Sz0.unsqueeze(0), "Num")
    den1 = ((Szz0 << logT) - Sz0 * Sz0).clamp_(min=1)
    W1_hat = S(rdiv(num1 << (Fw + 1 - Fd), den1), "W")        # [H,H] (Fw)
    Sy = int(ys.sum())
    Syz = imatmul(ys.unsqueeze(0), code1).squeeze(0)
    num_o = S((Syz << logT) - Sy * Sz1, "Num")
    den_o = ((Szz1 << logT) - Sz1 * Sz1).clamp_(min=1)
    Wo_hat = S(rdiv(num_o << (Fw + 1 - Fd), den_o), "W").unsqueeze(0)
    if state["M1s"] is None:
        state["M1s"], state["Mos"] = W1_hat << Ge, Wo_hat << Ge
    else:
        state["M1s"] = S(state["M1s"] + rshift((W1_hat << Ge) - state["M1s"],
                                               k_ema), "M_store")
        state["Mos"] = S(state["Mos"] + rshift((Wo_hat << Ge) - state["Mos"],
                                               k_ema), "M_store")
    M1, Mo = rshift(state["M1s"], Ge), rshift(state["Mos"], Ge)  # 使用値 (Fw)

    # ---- 再帰 credit (全て Fd) ----
    e = 2 * (y - tq)
    a1 = S(rshift(e * Mo.squeeze(0), Fw), "credit")
    dd1 = S(rshift(a1 * slope1, Fd), "credit")
    a0 = S(rshift(imatmul(M1.T, dd1.unsqueeze(1)).squeeze(1), Fw), "credit")
    a0s = S(rshift(a0 * slope0, Fd), "credit")

    # ---- 勾配 (Fg = Fd+Gg) -> sgdm -> 更新 + KP ----
    grads = {"W1": rshift(dd1.unsqueeze(1) * Sz0.unsqueeze(0), 1 + logT - Gg),
             "b1": dd1 << Gg,
             "W0": rshift(a0s.unsqueeze(1) * xq[0].unsqueeze(0), Fd - Gg),
             "b0": a0s << Gg,
             "Wout": rshift((e * Sz1), 1 + logT - Gg).unsqueeze(0),
             "bout": torch.tensor([e << Gg])}
    for k, g in grads.items():
        vel[k] = S(vel[k] - rshift(vel[k], 4) + g, "v")       # mu = 1 - 2^-4
        step_k = rshift(vel[k] * rom_q, Fg + 8 + lr_shift - Fw)
        pq[k] = S(pq[k] - step_k, "W")
        if k == "W1":
            state["M1s"] = S(state["M1s"] - (step_k << Ge), "M_store")
        elif k == "Wout":
            state["Mos"] = S(state["Mos"] - (step_k << Ge), "M_store")

    if wd is not None:                                        # 語長の実測
        wd.track("W", pq["W1"])
        wd.track("Wout", pq["Wout"])
        wd.track("d", d1)
        wd.track("ys", ys)
        wd.track("Num", num1)
        wd.track("NumOut", num_o)
        wd.track("M_store", state["M1s"])
        wd.track("v", vel["W1"])
        wd.track("credit", a0)
    return {"d0": d0, "code0": code0, "code1": code1, "d1": d1, "ys": ys,
            "y": y, "slope0": slope0, "slope1": slope1, "num1": num1,
            "den1": den1, "W1_hat": W1_hat, "Wo_hat": Wo_hat, "e": e,
            "a1": a1, "dd1": dd1, "a0": a0, "Sz0": Sz0, "Sz1": Sz1}


def train_stream_fxp(p, x_all, t_all, args, cfg, src):
    """B=1 オンライン学習の固定小数点ゴールデンモデル。

    形式: 重み/ミラー Fw、前活性・credit・slope Fd、EMA 蓄積 Fw+Ge、
    勾配/速度 Fd+Gg。lr = 2^-lr_shift x (9bit cosine ROM/256)。
    除算は mirror の (Num << (Fw+1-Fd)) / Den のみ (丸め付き整数除算)。
    ループ本体は fxp_step (RTL の bit-exact リファレンス)。
    """
    steps = args.budget
    rom = torch.round(256.0 * 0.5 * (1.0 + torch.cos(
        math.pi * (torch.arange(1024, dtype=torch.float64) + 0.5) / 1024))).long()
    N_all = x_all.shape[0]
    eval_every = max(1, steps // 50)

    state, cfg = fxp_prepare(p, args, cfg)
    xq_all = torch.round(x_all * (1 << Fd_of(cfg))).long()
    tq_all = torch.round(t_all * (1 << Fd_of(cfg))).long()
    wd = Widths()
    curve = []
    pos, direction = 0, 1
    for step in range(steps):
        rom_q = int(rom[(step * 1024) // steps])              # 9bit ROM 値
        if cfg["order"] == "iid":
            i = np.random.randint(N_all)
        else:
            i = pos
            pos += direction
            if pos in (0, N_all - 1):
                direction = -direction
        fxp_step(state, xq_all[i:i + 1], int(tq_all[i]), rom_q, src, cfg,
                 wd=(wd if step % 256 == 0 else None))
        if step % eval_every == 0 or step == steps - 1:
            pred = predict_fxp(state["pq"], xq_all, src.fork(step), cfg)
            mse = float(np.mean((pred - t_all.numpy().ravel()) ** 2))
            curve.append((step, mse))

    Fw, Fd, Ge, Gg = cfg["Fw"], cfg["Fd"], cfg["Ge"], cfg["Gg"]
    mirror_r = {"out": _corr(rshift(state["Mos"], Ge), state["pq"]["Wout"]),
                "W1": _corr(rshift(state["M1s"], Ge), state["pq"]["W1"])}
    fracs = {"W": Fw, "Wout": Fw, "d": Fd, "ys": Fd, "Num": Fd, "NumOut": Fd,
             "M_store": Fw + Ge, "v": Fd + Gg, "credit": Fd}
    return curve, mirror_r, wd.report(fracs)


def Fd_of(cfg) -> int:
    return cfg["Fd"]


# ============================================================
# 実験ドライバ
# ============================================================
def run_config(name, cfg, args, x_all, t_all, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    fresh = model_factory("uniform", args, torch.device("cpu"))
    p = extract_params(fresh())
    n_layers = 2
    src = NoiseSource(cfg["noise"], args.num_samples, args.hidden_dim,
                      n_layers, seed)
    t0 = time.time()
    if cfg["arith"] == "float":
        curve, mirror_r = train_stream_float(p, x_all, t_all, args, cfg, src)
        widths = ""
    else:
        curve, mirror_r, widths = train_stream_fxp(p, x_all, t_all, args, cfg, src)
    final = float(np.mean([m for _, m in curve[-3:]]))        # 終端 3 点平均
    last = curve[-1][1]
    print(f"[seed {seed}] {name:16s} final MSE = {last:.5f} "
          f"(tail3 {final:.5f}) mirror_r=" +
          ",".join(f"{k}:{v:.3f}" for k, v in mirror_r.items()) +
          f"  ({time.time() - t0:.0f}s)" + (f"\n    widths: {widths}" if widths else ""),
          flush=True)
    return {"curve": curve, "mirror_r": mirror_r, "final": last,
            "tail3": final, "widths": widths}


def stage_table(title, results, args, note=""):
    lines = [f"**{title}** (H={args.hidden_dim}, T={args.num_samples}, "
             f"budget={args.budget}, seeds={args.seed_list}; "
             f"Phase 0 batch_ref = {BATCH_REF})", "",
             "| config | final MSE (mean ± std) | vs batch_ref | mirror r | 備考 |",
             "|---|---|---|---|---|"]
    for name, per_seed in results.items():
        vals = [r["final"] for r in per_seed.values()]
        m, s = float(np.mean(vals)), float(np.std(vals))
        r0 = per_seed[args.seed_list[0]]
        r_txt = ", ".join(f"{k}={np.mean([r['mirror_r'][k] for r in per_seed.values()]):.3f}"
                          for k in r0["mirror_r"])
        lines.append(f"| {name} | {m:.5f} ± {s:.5f} | x{m / BATCH_REF:.2f} "
                     f"| {r_txt} | {r0['widths']} |")
    if note:
        lines += ["", note]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="Phase 1: fixed-point golden model")
    add_common_args(p)
    p.set_defaults(lr=0.125)
    p.add_argument("--stage", type=str, default="stream,lfsr,fxp")
    p.add_argument("--momentum", type=float, default=0.9375)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--order", choices=("traj", "iid"), default="traj")
    p.add_argument("--ema-shift", type=int, default=10,
                   help="Stage B/C の mirror EMA シフト定数 k (2^-k)")
    p.add_argument("--fxp-noise", type=str, default="phase24",
                   help="Stage C のノイズ源 (Stage B の結果で選ぶ)")
    args = finalize_args(p.parse_args(), default_out="out/fncl_phase1_fxp")
    if args.budget is None:
        args.budget = 128 * args.epochs
    if args.quick:
        args.budget = min(args.budget, 4000)
    stages = [s.strip() for s in args.stage.split(",") if s.strip()]
    device = torch.device("cpu")
    _, _, x_all, t_all = make_task(device)
    x_all, t_all = x_all.double(), t_all.double()
    print(f"Phase 1 stages={stages} budget={args.budget} "
          f"order={args.order} lr0={args.lr}/128", flush=True)

    def sweep(stage_name, cfgs, seeds=None, note=""):
        seeds = seeds or args.seed_list
        results = {}
        for name, cfg in cfgs.items():
            cfg.setdefault("order", args.order)
            results[name] = {s: run_config(name, cfg, args, x_all, t_all, s)
                             for s in seeds}
        table = stage_table(f"Phase 1 {stage_name}", results, args, note)
        print("\n" + table)
        write_text(args.out_dir / f"table_{stage_name}.md", table)
        save_json(args.out_dir / f"results_{stage_name}.json",
                  {"config": config_dict(args),
                   "results": {n: {s: {k: v for k, v in r.items() if k != "curve"}
                                   for s, r in per.items()}
                               for n, per in results.items()},
                   "curves": {n: per[seeds[0]]["curve"]
                              for n, per in results.items()}})
        fig = plt.figure(figsize=(7.0, 4.5))
        for n, per in results.items():
            st, vals = zip(*per[seeds[0]]["curve"])
            plt.plot(st, vals, label=n, lw=1.2)
        plt.yscale("log")
        plt.xlabel("presentations")
        plt.ylabel("eval MSE (log)")
        plt.title(f"Phase 1 {stage_name}")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        fig.tight_layout()
        savefig(fig, args.out_dir / f"fig_{stage_name}.png")
        return results

    if "stream" in stages:
        cfgs = {"stream_exact": {"arith": "float", "noise": "rand", "ema": "exact"},
                "stream_exact_iid": {"arith": "float", "noise": "rand",
                                     "ema": "exact", "order": "iid"},
                "stream_k10": {"arith": "float", "noise": "rand", "ema": 10},
                "stream_k11": {"arith": "float", "noise": "rand", "ema": 11}}
        sweep("stream", cfgs,
              note="判定: Phase 0 B1 (x1.00-1.09) と同水準なら A 合格。"
                   "k10/k11 が exact と並ぶなら EMA はシフト 1 回で凍結。")

    if "lfsr" in stages:
        k = args.ema_shift
        cfgs = {f"lfsr_{v}": {"arith": "float", "noise": v, "ema": k}
                for v in ("phase16", "leap16", "phase24", "leap24")}
        sweep("lfsr", cfgs,
              note=f"EMA は 2^-{k} 固定。rand 基準は table_stream.md の "
                   "stream_k10 行と比較する。")

    if "fxp" in stages:
        base = {"arith": "fxp", "noise": args.fxp_noise, "Fw": 16, "Fd": 12,
                "Nn": 12, "Ge": 4, "Gg": 4, "k_ema": args.ema_shift,
                "lr_shift": 10}
        cfgs = {"fxp_base": dict(base),
                "fxp_rand": dict(base, noise="rand"),
                "fxp_fw14": dict(base, Fw=14),
                "fxp_fw12": dict(base, Fw=12),
                "fxp_fd10": dict(base, Fd=10, Nn=10),
                "fxp_fd8": dict(base, Fd=8, Nn=8),
                "fxp_nn8": dict(base, Nn=8),
                "fxp_nn6": dict(base, Nn=6)}
        variant_seeds = args.seed_list[:1]
        results = sweep("fxp", {"fxp_base": cfgs.pop("fxp_base")},
                        note="(base のみ全 seed; 下の掃引は seed 単発)")
        sweep("fxp_sweep", cfgs, seeds=variant_seeds,
              note="判定: x2 以内の最小ビット幅構成を Phase 2 の語長仕様とする。"
                   f"base = Fw16/Fd12/Nn12/EMA 2^-{args.ema_shift}/lr 2^-10。")
        _ = results

    if "fxpsat" in stages:
        # Phase 2: 語長契約 (SAT_WIDTHS) の飽和演算で凍結構成を再検証する
        base = {"arith": "fxp", "noise": args.fxp_noise, "Fw": 14, "Fd": 10,
                "Nn": 8, "Ge": 4, "Gg": 4, "k_ema": args.ema_shift,
                "lr_shift": 10, "sat": True}
        sweep("fxpsat", {"fxp_sat_14_10_8": base},
              note=f"語長契約 = {SAT_WIDTHS} での飽和演算。判定: x2 以内なら "
                   "RTL はこの契約で実装できる。")

    if "fxpmin" in stages:
        # 単軸掃引で生き残った縮小軸を「組み合わせて」最終検証する (全 seed)
        base = {"arith": "fxp", "noise": args.fxp_noise, "Ge": 4, "Gg": 4,
                "k_ema": args.ema_shift, "lr_shift": 10}
        cfgs = {"fxp_min_14_10_8": dict(base, Fw=14, Fd=10, Nn=8),
                "fxp_min_14_8_6": dict(base, Fw=14, Fd=8, Nn=6),
                "fxp_min_16_10_8": dict(base, Fw=16, Fd=10, Nn=8)}
        sweep("fxpmin", cfgs,
              note="判定: x2 以内なら Phase 2 の語長仕様として凍結。")


if __name__ == "__main__":
    main()
