"""cov_jac 学習コア (L=H 構成, docs/idea_fpga.md §5 の ROW/COL アレイ)。

1 提示 (go パルス) ごとに forward -> mirror -> credit -> 更新 + KP を実行し、
ゴールデンモデル tmp/fncl_phase1_fxp.py の fxp_step と bit-exact に一致する。

構成 (§5.1):
  ROW レーン j = 第 2 層ユニット: W1 行 / v_W1 行 / b1 / Wout / Mout /
      LFSR1 / 交差 1 / readout 統計 (Σys·z1, Σd1)
  COL レーン i = 第 1 層ユニット: M1 列 / vM 複製 (KP 用) / Σdz 列 /
      LFSR0 / 交差 0 / 除算器 / credit MAC
  ブロードキャストバス: code0_i / d1_j / Σd1_j / dd1_j / Σz0_i を配る。

フェーズ (FSM): START -> {IN0 -> BC_CODE -> D1LAT -> IN1 -> BC_D1} x T
  (交差の巡回項は WRAP0/WRAP1 で処理) -> YE -> slope 除算 -> out-mirror 除算
  -> M1 mirror (j ループ, 除算 + EMA) -> credit (a1, dd1, a0) -> 更新
  (ベクトル一括 / W1 i ループ / KP 複製 j ループ) -> DONE。

このファイルは機能検証用のリファレンス RTL (v0):
  - レーンローカル記憶は レジスタ配列 (分散 RAM 相当) で表現
  - 除算は直列 RoundDiv (レーンあたり 1 基)
  - パイプライン重畳 (§5.2 の SAMPLE 2 位相重ね) は未実装
"""
from amaranth.hdl import Array, Elaboratable, Module, Mux, Signal, signed

from .crossing import CrossingUnit
from .divider import RoundDiv
from .lfsr import GaloisLfsr
from .rounding import rshift_round, saturate

# 語長契約 (tmp/fncl_phase1_fxp.SAT_WIDTHS と一致させること)
W_W, W_D, W_YS, W_CR, W_SL, W_V, W_MS, W_NUM = 18, 16, 16, 14, 13, 18, 20, 24


def gated(val, code):
    """val * code (code ∈ {0,1,2}): シフトと選択のみ."""
    return Mux(code[1], val << 1, Mux(code[0], val, 0))


class FnclCore(Elaboratable):
    def __init__(self, H, T, params, Fw=14, Fd=10, Nn=8, Ge=4, Gg=4,
                 k_ema=10, lr_shift=10, hq=205):
        assert T & (T - 1) == 0
        self.H, self.T = H, T
        self.Fw, self.Fd, self.Nn, self.Ge, self.Gg = Fw, Fd, Nn, Ge, Gg
        self.k_ema, self.hq = k_ema, hq
        self.logT = T.bit_length() - 1
        self.sh_mac = 1 + Fw - Fd                 # code·W の総和 -> Fd
        self.sh_b = Fw - Fd                       # バイアス -> Fd
        self.sh_num = Fw + 1 - Fd                 # mirror 商 -> Fw
        self.sh_g1 = 1 + self.logT - Gg           # dd1·Σz -> Fg
        self.sh_g0 = Fd - Gg                      # a0s·x -> Fg
        self.sh_step = (Fd + Gg) + 8 + lr_shift - Fw
        self.den_slope = 2 * hq * T

        self.go = Signal()
        self.done = Signal()
        self.xq = Signal(signed(14))
        self.tq = Signal(signed(14))
        self.romq = Signal(9)
        self.y = Signal(signed(17))
        self.first = Signal(init=1)

        P = params

        def sig(w, v, name):
            return Signal(signed(w), init=int(v), name=name)

        # ---- ROW レーン (j) ----
        self.w1 = [[sig(W_W, P["W1"][j][i], f"w1_{j}_{i}") for i in range(H)]
                   for j in range(H)]
        self.v_w1 = [[sig(W_V, 0, f"vw1_{j}_{i}") for i in range(H)]
                     for j in range(H)]
        self.b1 = [sig(W_W, P["b1"][j], f"b1_{j}") for j in range(H)]
        self.v_b1 = [sig(W_V, 0, f"vb1_{j}") for j in range(H)]
        self.wout = [sig(W_W, P["Wout"][j], f"wout_{j}") for j in range(H)]
        self.v_wout = [sig(W_V, 0, f"vwout_{j}") for j in range(H)]
        self.mos = [sig(W_MS, 0, f"mos_{j}") for j in range(H)]
        self.lfsr1 = [GaloisLfsr(24) for _ in range(H)]
        self.cross1 = [CrossingUnit(W_D + 2, T, hq) for _ in range(H)]
        self.div_r = [RoundDiv(30, 15) for _ in range(H)]

        # ---- COL レーン (i) ----
        self.w0 = [sig(W_W, P["W0"][i], f"w0_{i}") for i in range(H)]
        self.v_w0 = [sig(W_V, 0, f"vw0_{i}") for i in range(H)]
        self.b0 = [sig(W_W, P["b0"][i], f"b0_{i}") for i in range(H)]
        self.v_b0 = [sig(W_V, 0, f"vb0_{i}") for i in range(H)]
        self.m1s = [[sig(W_MS, 0, f"m1s_{i}_{j}") for j in range(H)]
                    for i in range(H)]            # [COL i][行 j] = M1s[j][i]
        self.vmr = [[sig(W_V, 0, f"vmr_{i}_{j}") for j in range(H)]
                    for i in range(H)]            # v_W1 の複製 (KP 用)
        self.acc_dz = [[Signal(signed(26), name=f"adz_{i}_{j}")
                        for j in range(H)] for i in range(H)]
        self.lfsr0 = [GaloisLfsr(24) for _ in range(H)]
        self.cross0 = [CrossingUnit(W_D + 2, T, hq) for _ in range(H)]
        self.div_c = [RoundDiv(30, 15) for _ in range(H)]

        # ---- グローバル ----
        self.bout = sig(W_W, P["bout"], "bout")
        self.v_bout = sig(W_V, 0, "v_bout")

    def elaborate(self, platform):
        m = Module()
        H, T = self.H, self.T
        Fw, Fd, Nn, Ge = self.Fw, self.Fd, self.Nn, self.Ge
        for n, subs in [("lfsr0", self.lfsr0), ("lfsr1", self.lfsr1),
                        ("cross0", self.cross0), ("cross1", self.cross1),
                        ("divc", self.div_c), ("divr", self.div_r)]:
            for k, s in enumerate(subs):
                m.submodules[f"{n}_{k}"] = s

        # ---- 提示ごとの一時レジスタ ----
        d0 = [Signal(signed(W_D), name=f"d0_{i}") for i in range(H)]
        d1 = [Signal(signed(W_D), name=f"d1_{j}") for j in range(H)]
        code0_l = [Signal(2, name=f"c0l_{i}") for i in range(H)]
        d1acc = [Signal(signed(26), name=f"d1a_{j}") for j in range(H)]
        sd1 = [Signal(signed(24), name=f"sd1_{j}") for j in range(H)]
        syz = [Signal(signed(25), name=f"syz_{j}") for j in range(H)]
        ysum = Signal(signed(24))
        e = Signal(signed(19))
        den1 = [Signal(15, name=f"den1_{i}") for i in range(H)]
        den_o = [Signal(15, name=f"deno_{j}") for j in range(H)]
        slope0 = [Signal(signed(W_SL), name=f"sl0_{i}") for i in range(H)]
        slope1 = [Signal(signed(W_SL), name=f"sl1_{j}") for j in range(H)]
        wo_hat = [Signal(signed(W_W), name=f"woh_{j}") for j in range(H)]
        a1 = [Signal(signed(W_CR), name=f"a1_{j}") for j in range(H)]
        dd1 = [Signal(signed(W_CR), name=f"dd1_{j}") for j in range(H)]
        a0acc = [Signal(signed(38), name=f"a0a_{i}") for i in range(H)]
        a0 = [Signal(signed(W_CR), name=f"a0_{i}") for i in range(H)]
        a0s = [Signal(signed(W_CR), name=f"a0s_{i}") for i in range(H)]
        self.dbg = {"d0": d0, "d1": d1, "ysum": ysum, "e": e,
                    "code0_l": code0_l,
                    "slope0": slope0, "slope1": slope1, "sd1": sd1,
                    "syz": syz, "a1": a1, "dd1": dd1, "a0": a0, "a0s": a0s,
                    "acc_dz": self.acc_dz, "wo_hat": wo_hat}
        kc = Signal(range(T + 1))                 # cross0 に入れたサンプル数
        pk = Signal(range(T))                     # 処理中の code 添字
        bc = Signal(range(H))                     # ブロードキャストカウンタ
        jc = Signal(range(H + 1))                 # mirror の行ループ

        # ---- ノイズ整形: nd = (2n + 1 - 2^Nn) << (Fd - Nn) ----
        def noise(lfsr):
            n = lfsr.state[24 - Nn:24]
            return ((n << 1) + 1 - (1 << Nn)).as_signed() << (Fd - Nn)

        # ---- 交差・LFSR の常時配線 ----
        fsm_in0 = Signal()
        fsm_in1 = Signal()
        self.dbg["in0"], self.dbg["in1"] = fsm_in0, fsm_in1
        for i in range(H):
            m.d.comb += [
                self.cross0[i].dn.eq(d0[i] + noise(self.lfsr0[i])),
                self.cross0[i].in_valid.eq(fsm_in0),
                self.cross0[i].wrap_en.eq(0),     # WRAP0 でのみ上書きで許可
                self.lfsr0[i].en.eq(fsm_in0),
            ]
        for j in range(H):
            m.d.comb += [
                self.cross1[j].dn.eq(d1[j] + noise(self.lfsr1[j])),
                self.cross1[j].in_valid.eq(fsm_in1),
                self.cross1[j].wrap_en.eq(0),     # WRAP1 でのみ上書きで許可
                self.lfsr1[j].en.eq(fsm_in1),
            ]

        # ---- code0 のラッチ (IN0 / WRAP0 の emit サイクル) ----
        with m.If(self.cross0[0].code_valid):
            m.d.sync += [code0_l[i].eq(self.cross0[i].code) for i in range(H)]

        # ---- readout (cross1 の emit サイクルに常時): 加算木 + 統計 ----
        mac1 = sum(gated(self.wout[j], self.cross1[j].code) for j in range(H))
        ys_v = Signal(signed(W_YS))
        m.d.comb += ys_v.eq(saturate(
            rshift_round(mac1, self.sh_mac)
            + rshift_round(self.bout, self.sh_b), W_YS))
        with m.If(self.cross1[0].code_valid):
            m.d.sync += ysum.eq(ysum + ys_v)
            m.d.sync += [syz[j].eq(syz[j] + gated(ys_v, self.cross1[j].code))
                         for j in range(H)]

        # ---- ブロードキャスト値 ----
        code_bc = Array(code0_l)[bc]
        d1_bc = Array(d1)[bc]
        dd1_bc = Array(dd1)[bc]
        sz0_bc = Array([c.sum_code for c in self.cross0])[bc]
        sd1_bc = Array(sd1)[jc]

        # ---- sgdm (momentum = 1 - 2^-4, lr = ROM x 2^-lr_shift) ----
        def sgdm(vel, g):
            vn = saturate(vel - rshift_round(vel, 4) + g, W_V)
            return vn, rshift_round(vn * self.romq, self.sh_step)

        with m.FSM():
            with m.State("IDLE"):
                with m.If(self.go):
                    m.d.sync += [ysum.eq(0), kc.eq(0)]
                    m.d.sync += [s.eq(0) for s in sd1 + syz]
                    m.next = "START"
            with m.State("START"):                # 交差リセット + d0
                m.d.comb += [c.start.eq(1) for c in self.cross0 + self.cross1]
                m.d.sync += [d0[i].eq(saturate(
                    rshift_round(self.w0[i] * self.xq, Fw)
                    + rshift_round(self.b0[i], self.sh_b), W_D))
                    for i in range(H)]
                m.next = "IN0"
            with m.State("IN0"):                  # cross0 へサンプル投入
                m.d.comb += fsm_in0.eq(1)
                m.d.sync += kc.eq(kc + 1)
                with m.If(kc == 0):
                    m.next = "IN0"
                with m.Else():
                    m.d.sync += [pk.eq(kc - 1), bc.eq(0)]
                    m.next = "BC_CODE"
            with m.State("WRAP0"):                # cross0 巡回項 -> code0(T-1)
                m.d.comb += [c.wrap_en.eq(1) for c in self.cross0]
                m.d.sync += [pk.eq(T - 1), bc.eq(0)]
                m.next = "BC_CODE"
            with m.State("BC_CODE"):              # d1_j += W1[j][bc] (ゲート加算)
                for j in range(H):
                    v = gated(Array(self.w1[j])[bc], code_bc)
                    m.d.sync += d1acc[j].eq(Mux(bc == 0, v, d1acc[j] + v))
                with m.If(bc == H - 1):
                    m.next = "D1LAT"
                with m.Else():
                    m.d.sync += bc.eq(bc + 1)
            with m.State("D1LAT"):
                m.d.sync += [d1[j].eq(saturate(
                    rshift_round(d1acc[j], self.sh_mac)
                    + rshift_round(self.b1[j], self.sh_b), W_D))
                    for j in range(H)]
                m.next = "IN1"
            with m.State("IN1"):                  # cross1 へ投入 + Σd1
                m.d.comb += fsm_in1.eq(1)
                m.d.sync += [sd1[j].eq(sd1[j] + d1[j]) for j in range(H)]
                m.d.sync += bc.eq(0)
                m.next = "BC_D1"
            with m.State("BC_D1"):                # Σdz[i][pk 行? -> 列累算]
                for i in range(H):
                    v = gated(d1_bc, code0_l[i])
                    tgt = Array(self.acc_dz[i])[bc]
                    m.d.sync += tgt.eq(Mux(pk == 0, v, tgt + v))
                with m.If(bc == H - 1):
                    with m.If(pk == T - 1):
                        m.next = "WRAP1"
                    with m.Elif(kc == T):
                        m.next = "WRAP0"
                    with m.Else():
                        m.next = "IN0"
                with m.Else():
                    m.d.sync += bc.eq(bc + 1)
            with m.State("WRAP1"):                # cross1 巡回項 -> readout
                m.d.comb += [c.wrap_en.eq(1) for c in self.cross1]
                m.next = "YE"
            with m.State("YE"):                   # y, 分母, 誤差の準備
                m.d.sync += self.y.eq(rshift_round(ysum, self.logT))
                for i in range(H):
                    dv = ((self.cross0[i].sum_code2 << self.logT)
                          - self.cross0[i].sum_code * self.cross0[i].sum_code)
                    m.d.sync += den1[i].eq(Mux(dv < 1, 1, dv))
                for j in range(H):
                    dv = ((self.cross1[j].sum_code2 << self.logT)
                          - self.cross1[j].sum_code * self.cross1[j].sum_code)
                    m.d.sync += den_o[j].eq(Mux(dv < 1, 1, dv))
                m.next = "SLST"
            with m.State("SLST"):                 # slope = rdiv(cdiff<<2Fd, 2hT)
                m.d.sync += e.eq((self.y - self.tq) << 1)
                for i in range(H):
                    m.d.comb += [self.div_c[i].start.eq(1),
                                 self.div_c[i].num.eq(
                                     self.cross0[i].cdiff << (2 * Fd)),
                                 self.div_c[i].den.eq(self.den_slope)]
                for j in range(H):
                    m.d.comb += [self.div_r[j].start.eq(1),
                                 self.div_r[j].num.eq(
                                     self.cross1[j].cdiff << (2 * Fd)),
                                 self.div_r[j].den.eq(self.den_slope)]
                m.next = "SLWT"
            with m.State("SLWT"):
                with m.If(self.div_c[0].done):
                    m.d.sync += [slope0[i].eq(saturate(self.div_c[i].q, W_SL))
                                 for i in range(H)]
                    m.d.sync += [slope1[j].eq(saturate(self.div_r[j].q, W_SL))
                                 for j in range(H)]
                    m.next = "MOST"
            with m.State("MOST"):                 # out mirror: W_hat -> EMA
                for j in range(H):
                    num_o = saturate((syz[j] << self.logT)
                                     - ysum * self.cross1[j].sum_code, W_NUM)
                    m.d.comb += [self.div_r[j].start.eq(1),
                                 self.div_r[j].num.eq(num_o << self.sh_num),
                                 self.div_r[j].den.eq(den_o[j])]
                m.next = "MOWT"
            with m.State("MOWT"):
                with m.If(self.div_r[0].done):
                    for j in range(H):
                        wh = saturate(self.div_r[j].q, W_W)
                        m.d.sync += wo_hat[j].eq(wh)
                        m.d.sync += self.mos[j].eq(Mux(
                            self.first, wh << Ge,
                            saturate(self.mos[j] + rshift_round(
                                (wh << Ge) - self.mos[j], self.k_ema), W_MS)))
                    m.d.sync += jc.eq(0)
                    m.next = "M1ST"
            with m.State("M1ST"):                 # M1 mirror: 行 jc の除算
                for i in range(H):
                    num1 = saturate(
                        (Array(self.acc_dz[i])[jc] << self.logT)
                        - sd1_bc * self.cross0[i].sum_code, W_NUM)
                    m.d.comb += [self.div_c[i].start.eq(1),
                                 self.div_c[i].num.eq(num1 << self.sh_num),
                                 self.div_c[i].den.eq(den1[i])]
                m.next = "M1WT"
            with m.State("M1WT"):
                with m.If(self.div_c[0].done):
                    for i in range(H):
                        wh = saturate(self.div_c[i].q, W_W)
                        tgt = Array(self.m1s[i])[jc]
                        m.d.sync += tgt.eq(Mux(
                            self.first, wh << Ge,
                            saturate(tgt + rshift_round(
                                (wh << Ge) - tgt, self.k_ema), W_MS)))
                    with m.If(jc == H - 1):
                        m.next = "CR1"
                    with m.Else():
                        m.d.sync += jc.eq(jc + 1)
                        m.next = "M1ST"
            with m.State("CR1"):                  # a1 = e·Mo >> Fw
                m.d.sync += [a1[j].eq(saturate(rshift_round(
                    e * rshift_round(self.mos[j], Ge), Fw), W_CR))
                    for j in range(H)]
                m.next = "CR2"
            with m.State("CR2"):                  # dd1 = a1·slope1 >> Fd
                m.d.sync += [dd1[j].eq(saturate(rshift_round(
                    a1[j] * slope1[j], Fd), W_CR)) for j in range(H)]
                m.d.sync += bc.eq(0)
                m.next = "A0L"
            with m.State("A0L"):                  # a0 = Σ_j M1u[j][i]·dd1_j
                for i in range(H):
                    prod = rshift_round(Array(self.m1s[i])[bc], Ge) * dd1_bc
                    m.d.sync += a0acc[i].eq(Mux(bc == 0, prod,
                                                a0acc[i] + prod))
                with m.If(bc == H - 1):
                    m.next = "A0F"
                with m.Else():
                    m.d.sync += bc.eq(bc + 1)
            with m.State("A0F"):
                m.d.sync += [a0[i].eq(saturate(rshift_round(a0acc[i], Fw),
                                               W_CR)) for i in range(H)]
                m.next = "A0S"
            with m.State("A0S"):                  # a0s = a0·slope0 >> Fd
                m.d.sync += [a0s[i].eq(saturate(rshift_round(
                    a0[i] * slope0[i], Fd), W_CR)) for i in range(H)]
                m.next = "UPDV"
            with m.State("UPDV"):                 # ベクトルパラメータ一括更新
                for j in range(H):
                    vn, st = sgdm(self.v_b1[j], dd1[j] << self.Gg)
                    m.d.sync += [self.v_b1[j].eq(vn),
                                 self.b1[j].eq(saturate(self.b1[j] - st, W_W))]
                    g = rshift_round(e * self.cross1[j].sum_code, self.sh_g1)
                    vn, st = sgdm(self.v_wout[j], g)
                    m.d.sync += [
                        self.v_wout[j].eq(vn),
                        self.wout[j].eq(saturate(self.wout[j] - st, W_W)),
                        self.mos[j].eq(saturate(self.mos[j] - (st << Ge),
                                                W_MS))]
                for i in range(H):
                    g = rshift_round(a0s[i] * self.xq, self.sh_g0)
                    vn, st = sgdm(self.v_w0[i], g)
                    m.d.sync += [self.v_w0[i].eq(vn),
                                 self.w0[i].eq(saturate(self.w0[i] - st, W_W))]
                    vn, st = sgdm(self.v_b0[i], a0s[i] << self.Gg)
                    m.d.sync += [self.v_b0[i].eq(vn),
                                 self.b0[i].eq(saturate(self.b0[i] - st, W_W))]
                vn, st = sgdm(self.v_bout, e << self.Gg)
                m.d.sync += [self.v_bout.eq(vn),
                             self.bout.eq(saturate(self.bout - st, W_W)),
                             bc.eq(0)]
                m.next = "UPW1"
            with m.State("UPW1"):                 # W1 更新 (i = bc ループ)
                for j in range(H):
                    g = rshift_round(dd1[j] * sz0_bc, self.sh_g1)
                    vn, st = sgdm(Array(self.v_w1[j])[bc], g)
                    tgt = Array(self.w1[j])[bc]
                    m.d.sync += [Array(self.v_w1[j])[bc].eq(vn),
                                 tgt.eq(saturate(tgt - st, W_W))]
                with m.If(bc == H - 1):
                    m.d.sync += bc.eq(0)
                    m.next = "UPKP"
                with m.Else():
                    m.d.sync += bc.eq(bc + 1)
            with m.State("UPKP"):                 # KP: COL 側 sgdm 複製 (j = bc)
                for i in range(H):
                    g = rshift_round(dd1_bc * self.cross0[i].sum_code,
                                     self.sh_g1)
                    vn, st = sgdm(Array(self.vmr[i])[bc], g)
                    tgt = Array(self.m1s[i])[bc]
                    m.d.sync += [Array(self.vmr[i])[bc].eq(vn),
                                 tgt.eq(saturate(tgt - (st << Ge), W_MS))]
                with m.If(bc == H - 1):
                    m.next = "DONE"
                with m.Else():
                    m.d.sync += bc.eq(bc + 1)
            with m.State("DONE"):
                m.d.comb += self.done.eq(1)
                m.d.sync += self.first.eq(0)
                m.next = "IDLE"
        return m
