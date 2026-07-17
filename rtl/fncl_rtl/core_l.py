"""cov_jac 学習コア (L パラメトリック + Memory 版, Phase 3 Step 3)。

core.py (FnclCore, L=H・レジスタ配列) の後継。違いは 2 点:

  1. **L パラメトリック**: ROW/COL 各 L レーン。レーン l は行/列
     j = l*R + rc (rc = 0..R-1, R = H/L) を受け持ち、H^2 配列を舐める
     ループは (bc, rc) の 2 重カウンタになる。
  2. **大配列は Memory**: W1 / v_W1 (ROW レーン) と M1s / vM / Σdz
     (COL レーン) はレーンローカルの同期読み出し Memory (深さ R*H)。
     読み出しレイテンシ 1 を ph (2 相) で吸収する v1 実装
     (パイプライン重畳は Step 4 の最適化)。

交差ユニット・LFSR はユニット毎に並列のまま (コンパレータ+カウンタで安価;
折り畳みは Phase 4)。H 長ベクトル (d1, credit, 統計) もレジスタのまま。

ゴールデンモデル tmp/fncl_phase1_fxp.py の fxp_step と bit-exact
(rtl/tests/test_core_l.py で L=8/4/2 と長期学習を検証)。
"""
from amaranth.hdl import Array, Elaboratable, Module, Mux, Signal, signed
from amaranth.lib.memory import Memory

from .crossing import CrossingUnit
from .divider import RoundDiv
from .lfsr import GaloisLfsr
from .rounding import rshift_round, saturate

W_W, W_D, W_YS, W_CR, W_SL, W_V, W_MS, W_NUM = 18, 16, 16, 14, 13, 18, 20, 24


def gated(val, code):
    return Mux(code[1], val << 1, Mux(code[0], val, 0))


class FnclCoreL(Elaboratable):
    def __init__(self, H, T, L, params, Fw=14, Fd=10, Nn=8, Ge=4, Gg=4,
                 k_ema=10, lr_shift=10, hq=205):
        assert T & (T - 1) == 0 and H % L == 0
        self.H, self.T, self.L = H, T, L
        self.R = R = H // L
        self.Fw, self.Fd, self.Nn, self.Ge, self.Gg = Fw, Fd, Nn, Ge, Gg
        self.k_ema, self.hq = k_ema, hq
        self.logT = T.bit_length() - 1
        self.sh_mac = 1 + Fw - Fd
        self.sh_b = Fw - Fd
        self.sh_num = Fw + 1 - Fd
        self.sh_g1 = 1 + self.logT - Gg
        self.sh_g0 = Fd - Gg
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

        # ---- レーンローカル Memory (深さ R*H, addr = rc*H + 相手添字) ----
        def mems(name, width, init_rows=None):
            out = []
            for l in range(self.L):
                if init_rows is None:
                    init = None
                else:
                    init = [int(v) for rc in range(R)
                            for v in init_rows[l * R + rc]]
                out.append(Memory(shape=signed(width), depth=R * H,
                                  init=init or []))
            return out

        self.mem_w1 = mems("w1", W_W, P["W1"])
        self.mem_vw1 = mems("vw1", W_V)
        self.mem_m1s = mems("m1s", W_MS)      # COL lane l: [rc][j] = M1s[j][i]
        self.mem_vmr = mems("vmr", W_V)
        self.mem_adz = mems("adz", 26)

        # ---- レジスタ (ユニット毎ベクトル + スカラ) ----
        self.b0 = [sig(W_W, P["b0"][i], f"b0_{i}") for i in range(H)]
        self.v_b0 = [sig(W_V, 0, f"vb0_{i}") for i in range(H)]
        self.w0 = [sig(W_W, P["W0"][i], f"w0_{i}") for i in range(H)]
        self.v_w0 = [sig(W_V, 0, f"vw0_{i}") for i in range(H)]
        self.b1 = [sig(W_W, P["b1"][j], f"b1_{j}") for j in range(H)]
        self.v_b1 = [sig(W_V, 0, f"vb1_{j}") for j in range(H)]
        self.wout = [sig(W_W, P["Wout"][j], f"wout_{j}") for j in range(H)]
        self.v_wout = [sig(W_V, 0, f"vwout_{j}") for j in range(H)]
        self.mos = [sig(W_MS, 0, f"mos_{j}") for j in range(H)]
        self.bout = sig(W_W, P["bout"], "bout")
        self.v_bout = sig(W_V, 0, "v_bout")

        self.lfsr0 = [GaloisLfsr(24) for _ in range(H)]
        self.lfsr1 = [GaloisLfsr(24) for _ in range(H)]
        self.cross0 = [CrossingUnit(W_D + 2, T, hq) for _ in range(H)]
        self.cross1 = [CrossingUnit(W_D + 2, T, hq) for _ in range(H)]
        self.div_c = [RoundDiv(30, 15) for _ in range(self.L)]
        self.div_r = [RoundDiv(30, 15) for _ in range(self.L)]

        # テストベンチ用の覗き穴 (addr は未駆動 = tb から設定できる)
        self.dbg_rp = {name: [mem.read_port(domain="sync") for mem in mems]
                       for name, mems in [("w1", self.mem_w1),
                                          ("vw1", self.mem_vw1),
                                          ("m1s", self.mem_m1s),
                                          ("vmr", self.mem_vmr)]}

    def elaborate(self, platform):
        m = Module()
        H, T, L, R = self.H, self.T, self.L, self.R
        Fw, Fd, Nn, Ge = self.Fw, self.Fd, self.Nn, self.Ge
        for name, subs in [("lfsr0", self.lfsr0), ("lfsr1", self.lfsr1),
                           ("cross0", self.cross0), ("cross1", self.cross1),
                           ("divc", self.div_c), ("divr", self.div_r),
                           ("m_w1", self.mem_w1), ("m_vw1", self.mem_vw1),
                           ("m_m1s", self.mem_m1s), ("m_vmr", self.mem_vmr),
                           ("m_adz", self.mem_adz)]:
            for k, s in enumerate(subs):
                m.submodules[f"{name}_{k}"] = s

        # ---- Memory ポート ----
        def ports(memlist):
            rps = [mem.read_port(domain="sync") for mem in memlist]
            wps = [mem.write_port() for mem in memlist]
            for rp in rps:
                m.d.comb += rp.en.eq(1)
            return rps, wps

        rp_w1, wp_w1 = ports(self.mem_w1)
        rp_vw1, wp_vw1 = ports(self.mem_vw1)
        rp_m1s, wp_m1s = ports(self.mem_m1s)
        rp_vmr, wp_vmr = ports(self.mem_vmr)
        rp_adz, wp_adz = ports(self.mem_adz)

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
        a1 = [Signal(signed(W_CR), name=f"a1_{j}") for j in range(H)]
        dd1 = [Signal(signed(W_CR), name=f"dd1_{j}") for j in range(H)]
        a0acc = [Signal(signed(38), name=f"a0a_{i}") for i in range(H)]
        a0 = [Signal(signed(W_CR), name=f"a0_{i}") for i in range(H)]
        a0s = [Signal(signed(W_CR), name=f"a0s_{i}") for i in range(H)]
        kc = Signal(range(T + 1))
        pk = Signal(range(T))
        bc = Signal(range(H))
        rc = Signal(range(R + 1))
        jc = Signal(range(H + 1))
        ph = Signal()

        def lane(vec, l):
            """レーン l が受け持つスライスの動的 rc 選択."""
            return Array(vec[l * R:(l + 1) * R])[rc]

        addr = Signal(range(R * H))
        m.d.comb += addr.eq(rc * H + bc)
        addr_j = Signal(range(R * H))
        m.d.comb += addr_j.eq(rc * H + jc)

        fsm_in0 = Signal()
        fsm_in1 = Signal()
        for i in range(H):
            nz = ((self.lfsr0[i].state[24 - Nn:24] << 1) + 1
                  - (1 << Nn)).as_signed() << (Fd - Nn)
            m.d.comb += [self.cross0[i].dn.eq(d0[i] + nz),
                         self.cross0[i].in_valid.eq(fsm_in0),
                         self.cross0[i].wrap_en.eq(0),
                         self.lfsr0[i].en.eq(fsm_in0)]
        for j in range(H):
            nz = ((self.lfsr1[j].state[24 - Nn:24] << 1) + 1
                  - (1 << Nn)).as_signed() << (Fd - Nn)
            m.d.comb += [self.cross1[j].dn.eq(d1[j] + nz),
                         self.cross1[j].in_valid.eq(fsm_in1),
                         self.cross1[j].wrap_en.eq(0),
                         self.lfsr1[j].en.eq(fsm_in1)]

        with m.If(self.cross0[0].code_valid):
            m.d.sync += [code0_l[i].eq(self.cross0[i].code) for i in range(H)]

        mac1 = sum(gated(self.wout[j], self.cross1[j].code) for j in range(H))
        ys_v = Signal(signed(W_YS))
        m.d.comb += ys_v.eq(saturate(
            rshift_round(mac1, self.sh_mac)
            + rshift_round(self.bout, self.sh_b), W_YS))
        with m.If(self.cross1[0].code_valid):
            m.d.sync += ysum.eq(ysum + ys_v)
            m.d.sync += [syz[j].eq(syz[j] + gated(ys_v, self.cross1[j].code))
                         for j in range(H)]

        code_bc = Array(code0_l)[bc]
        d1_bc = Array(d1)[bc]
        dd1_bc = Array(dd1)[bc]
        sz0_bc = Array([c.sum_code for c in self.cross0])[bc]
        sd1_j = Array(sd1)[jc]

        def sgdm(vel_val, g):
            vn = saturate(vel_val - rshift_round(vel_val, 4) + g, W_V)
            return vn, rshift_round(vn * self.romq, self.sh_step)

        def bump(next_state, cnt=bc, limit=H):
            """(cnt, rc) の 2 重カウンタを進める (ph=1 で呼ぶ)."""
            with m.If((rc == R - 1) & (cnt == limit - 1)):
                m.d.sync += [rc.eq(0), cnt.eq(0)]
                m.next = next_state
            with m.Elif(rc == R - 1):
                m.d.sync += [rc.eq(0), cnt.eq(cnt + 1)]
            with m.Else():
                m.d.sync += rc.eq(rc + 1)

        with m.FSM():
            with m.State("IDLE"):
                with m.If(self.go):
                    m.d.sync += [ysum.eq(0), kc.eq(0), bc.eq(0), rc.eq(0),
                                 ph.eq(0)]
                    m.d.sync += [s.eq(0) for s in sd1 + syz]
                    m.next = "START"
            with m.State("START"):
                m.d.comb += [c.start.eq(1) for c in self.cross0 + self.cross1]
                m.d.sync += [d0[i].eq(saturate(
                    rshift_round(self.w0[i] * self.xq, Fw)
                    + rshift_round(self.b0[i], self.sh_b), W_D))
                    for i in range(H)]
                m.next = "IN0"
            with m.State("IN0"):
                m.d.comb += fsm_in0.eq(1)
                m.d.sync += kc.eq(kc + 1)
                with m.If(kc == 0):
                    m.next = "IN0"
                with m.Else():
                    m.d.sync += [pk.eq(kc - 1), bc.eq(0), rc.eq(0), ph.eq(0)]
                    m.next = "BC_CODE"
            with m.State("WRAP0"):
                m.d.comb += [c.wrap_en.eq(1) for c in self.cross0]
                m.d.sync += [pk.eq(T - 1), bc.eq(0), rc.eq(0), ph.eq(0)]
                m.next = "BC_CODE"
            with m.State("BC_CODE"):                  # d1acc += W1·code (2 相)
                for l in range(L):
                    m.d.comb += rp_w1[l].addr.eq(addr)
                m.d.sync += ph.eq(~ph)
                with m.If(ph):
                    for l in range(L):
                        v = gated(rp_w1[l].data, code_bc)
                        tgt = lane(d1acc, l)
                        m.d.sync += tgt.eq(Mux(bc == 0, v, tgt + v))
                    bump("D1LAT")
            with m.State("D1LAT"):
                m.d.sync += [d1[j].eq(saturate(
                    rshift_round(d1acc[j], self.sh_mac)
                    + rshift_round(self.b1[j], self.sh_b), W_D))
                    for j in range(H)]
                m.next = "IN1"
            with m.State("IN1"):
                m.d.comb += fsm_in1.eq(1)
                m.d.sync += [sd1[j].eq(sd1[j] + d1[j]) for j in range(H)]
                m.d.sync += [bc.eq(0), rc.eq(0), ph.eq(0)]
                m.next = "BC_D1"
            with m.State("BC_D1"):                    # Σdz rmw (2 相)
                for l in range(L):
                    m.d.comb += [rp_adz[l].addr.eq(addr),
                                 wp_adz[l].addr.eq(addr)]
                m.d.sync += ph.eq(~ph)
                with m.If(ph):
                    for l in range(L):
                        v = gated(d1_bc, lane(code0_l, l))
                        m.d.comb += [wp_adz[l].en.eq(1),
                                     wp_adz[l].data.eq(
                                         Mux(pk == 0, v, rp_adz[l].data + v))]
                    with m.If((rc == R - 1) & (bc == H - 1)):
                        m.d.sync += [rc.eq(0), bc.eq(0)]
                        with m.If(pk == T - 1):
                            m.next = "WRAP1"
                        with m.Elif(kc == T):
                            m.next = "WRAP0"
                        with m.Else():
                            m.next = "IN0"
                    with m.Elif(rc == R - 1):
                        m.d.sync += [rc.eq(0), bc.eq(bc + 1)]
                    with m.Else():
                        m.d.sync += rc.eq(rc + 1)
            with m.State("WRAP1"):
                m.d.comb += [c.wrap_en.eq(1) for c in self.cross1]
                m.next = "YE"
            with m.State("YE"):
                m.d.sync += self.y.eq(rshift_round(ysum, self.logT))
                for i in range(H):
                    dv = ((self.cross0[i].sum_code2 << self.logT)
                          - self.cross0[i].sum_code * self.cross0[i].sum_code)
                    m.d.sync += den1[i].eq(Mux(dv < 1, 1, dv))
                for j in range(H):
                    dv = ((self.cross1[j].sum_code2 << self.logT)
                          - self.cross1[j].sum_code * self.cross1[j].sum_code)
                    m.d.sync += den_o[j].eq(Mux(dv < 1, 1, dv))
                m.d.sync += rc.eq(0)
                m.next = "SLST"
            with m.State("SLST"):                     # slope (rc ループ, L 並列)
                m.d.sync += e.eq((self.y - self.tq) << 1)
                for l in range(L):
                    cd0 = Array([c.cdiff for c in
                                 self.cross0[l * R:(l + 1) * R]])[rc]
                    cd1 = Array([c.cdiff for c in
                                 self.cross1[l * R:(l + 1) * R]])[rc]
                    m.d.comb += [self.div_c[l].start.eq(1),
                                 self.div_c[l].num.eq(cd0 << (2 * Fd)),
                                 self.div_c[l].den.eq(self.den_slope),
                                 self.div_r[l].start.eq(1),
                                 self.div_r[l].num.eq(cd1 << (2 * Fd)),
                                 self.div_r[l].den.eq(self.den_slope)]
                m.next = "SLWT"
            with m.State("SLWT"):
                with m.If(self.div_c[0].done):
                    for l in range(L):
                        m.d.sync += [
                            lane(slope0, l).eq(saturate(self.div_c[l].q, W_SL)),
                            lane(slope1, l).eq(saturate(self.div_r[l].q, W_SL))]
                    with m.If(rc == R - 1):
                        m.d.sync += rc.eq(0)
                        m.next = "MOST"
                    with m.Else():
                        m.d.sync += rc.eq(rc + 1)
                        m.next = "SLST"
            with m.State("MOST"):                     # out mirror (rc ループ)
                for l in range(L):
                    num_o = saturate(
                        (lane(syz, l) << self.logT)
                        - ysum * Array([c.sum_code for c in
                                        self.cross1[l * R:(l + 1) * R]])[rc],
                        W_NUM)
                    m.d.comb += [self.div_r[l].start.eq(1),
                                 self.div_r[l].num.eq(num_o << self.sh_num),
                                 self.div_r[l].den.eq(lane(den_o, l))]
                m.next = "MOWT"
            with m.State("MOWT"):
                with m.If(self.div_r[0].done):
                    for l in range(L):
                        wh = saturate(self.div_r[l].q, W_W)
                        tgt = lane(self.mos, l)
                        m.d.sync += tgt.eq(Mux(
                            self.first, wh << Ge,
                            saturate(tgt + rshift_round(
                                (wh << Ge) - tgt, self.k_ema), W_MS)))
                    with m.If(rc == R - 1):
                        m.d.sync += [rc.eq(0), jc.eq(0), ph.eq(0)]
                        m.next = "M1RD"
                    with m.Else():
                        m.d.sync += rc.eq(rc + 1)
                        m.next = "MOST"
            with m.State("M1RD"):                     # Σdz[rc][jc] 読み出し
                for l in range(L):
                    m.d.comb += rp_adz[l].addr.eq(addr_j)
                m.d.sync += ph.eq(~ph)
                with m.If(ph):
                    m.next = "M1ST"
            with m.State("M1ST"):                     # num1 -> 除算 (L 並列)
                for l in range(L):
                    num1 = saturate(
                        (rp_adz[l].data << self.logT)
                        - sd1_j * Array([c.sum_code for c in
                                         self.cross0[l * R:(l + 1) * R]])[rc],
                        W_NUM)
                    m.d.comb += [rp_adz[l].addr.eq(addr_j),
                                 self.div_c[l].start.eq(1),
                                 self.div_c[l].num.eq(num1 << self.sh_num),
                                 self.div_c[l].den.eq(lane(den1, l))]
                m.next = "M1WT"
            with m.State("M1WT"):                     # 除算待ち + m1s prefetch
                for l in range(L):
                    m.d.comb += rp_m1s[l].addr.eq(addr_j)
                with m.If(self.div_c[0].done):
                    for l in range(L):
                        wh = saturate(self.div_c[l].q, W_W)
                        old = rp_m1s[l].data
                        m.d.comb += [wp_m1s[l].en.eq(1),
                                     wp_m1s[l].addr.eq(addr_j),
                                     wp_m1s[l].data.eq(Mux(
                                         self.first, wh << Ge,
                                         saturate(old + rshift_round(
                                             (wh << Ge) - old, self.k_ema),
                                             W_MS)))]
                    m.d.sync += ph.eq(0)
                    with m.If((rc == R - 1) & (jc == H - 1)):
                        m.d.sync += [rc.eq(0), jc.eq(0)]
                        m.next = "CR1"
                    with m.Elif(rc == R - 1):
                        m.d.sync += [rc.eq(0), jc.eq(jc + 1)]
                        m.next = "M1RD"
                    with m.Else():
                        m.d.sync += rc.eq(rc + 1)
                        m.next = "M1RD"
            with m.State("CR1"):
                m.d.sync += [a1[j].eq(saturate(rshift_round(
                    e * rshift_round(self.mos[j], Ge), Fw), W_CR))
                    for j in range(H)]
                m.next = "CR2"
            with m.State("CR2"):
                m.d.sync += [dd1[j].eq(saturate(rshift_round(
                    a1[j] * slope1[j], Fd), W_CR)) for j in range(H)]
                m.d.sync += [bc.eq(0), rc.eq(0), ph.eq(0)]
                m.next = "A0L"
            with m.State("A0L"):                      # a0 = Σ M1u·dd1 (2 相)
                for l in range(L):
                    m.d.comb += rp_m1s[l].addr.eq(addr)
                m.d.sync += ph.eq(~ph)
                with m.If(ph):
                    for l in range(L):
                        prod = rshift_round(rp_m1s[l].data, Ge) * dd1_bc
                        tgt = lane(a0acc, l)
                        m.d.sync += tgt.eq(Mux(bc == 0, prod, tgt + prod))
                    bump("A0F")
            with m.State("A0F"):
                m.d.sync += [a0[i].eq(saturate(rshift_round(a0acc[i], Fw),
                                               W_CR)) for i in range(H)]
                m.next = "A0S"
            with m.State("A0S"):
                m.d.sync += [a0s[i].eq(saturate(rshift_round(
                    a0[i] * slope0[i], Fd), W_CR)) for i in range(H)]
                m.next = "UPDV"
            with m.State("UPDV"):
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
                             bc.eq(0), rc.eq(0), ph.eq(0)]
                m.next = "UPW1"
            with m.State("UPW1"):                     # W1/v 更新 (2 相 rmw)
                for l in range(L):
                    m.d.comb += [rp_w1[l].addr.eq(addr),
                                 rp_vw1[l].addr.eq(addr)]
                m.d.sync += ph.eq(~ph)
                with m.If(ph):
                    for l in range(L):
                        g = rshift_round(lane(dd1, l) * sz0_bc, self.sh_g1)
                        vn, st = sgdm(rp_vw1[l].data, g)
                        m.d.comb += [
                            wp_vw1[l].en.eq(1), wp_vw1[l].addr.eq(addr),
                            wp_vw1[l].data.eq(vn),
                            wp_w1[l].en.eq(1), wp_w1[l].addr.eq(addr),
                            wp_w1[l].data.eq(saturate(rp_w1[l].data - st,
                                                      W_W))]
                    bump("UPKP")
            with m.State("UPKP"):                     # KP: vM 複製 (2 相 rmw)
                for l in range(L):
                    m.d.comb += [rp_vmr[l].addr.eq(addr),
                                 rp_m1s[l].addr.eq(addr)]
                m.d.sync += ph.eq(~ph)
                with m.If(ph):
                    for l in range(L):
                        sz0_i = Array([c.sum_code for c in
                                       self.cross0[l * R:(l + 1) * R]])[rc]
                        g = rshift_round(dd1_bc * sz0_i, self.sh_g1)
                        vn, st = sgdm(rp_vmr[l].data, g)
                        m.d.comb += [
                            wp_vmr[l].en.eq(1), wp_vmr[l].addr.eq(addr),
                            wp_vmr[l].data.eq(vn),
                            wp_m1s[l].en.eq(1), wp_m1s[l].addr.eq(addr),
                            wp_m1s[l].data.eq(saturate(
                                rp_m1s[l].data - (st << Ge), W_MS))]
                    bump("DONE")
            with m.State("DONE"):
                m.d.comb += self.done.eq(1)
                m.d.sync += self.first.eq(0)
                m.next = "IDLE"
        return m
