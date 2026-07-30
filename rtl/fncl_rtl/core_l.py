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

Step 4 (合成試行後の改修) で除算器を集約: v0 の直列 RoundDiv 2L 基を
パイプライン RoundDivPipe 2 本 (div_c = COL 側: slope0 + 隠れ mirror,
div_r = ROW 側: slope1 + out mirror) に置き換え、SL/MO/M1 フェーズは
1 要素/サイクルのストリーミング発行。除算前の Num 計算 (乗算+飽和) も
フェーズあたり 1 実体に落ちる。

ゴールデンモデル tmp/fncl_phase1_fxp.py の fxp_step と bit-exact
(rtl/tests/test_core_l.py で L=8/4/2 と長期学習を検証)。
"""
from amaranth.hdl import Array, Elaboratable, Module, Mux, Signal, signed
from amaranth.lib.memory import Memory

from .crossing import CrossingEngine
from .divider import RoundDivPipe
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
        # 並列参照されるもの (b1: D1LAT, wout: mac1) はレジスタのまま。
        # ストリーミング参照のみの H 長ベクトルは Memory (分散 RAM) 化して
        # 読み出しマルチプレクサと書き込みデコーダを消す。
        self.b1 = [sig(W_W, P["b1"][j], f"b1_{j}") for j in range(H)]
        self.wout = [sig(W_W, P["Wout"][j], f"wout_{j}") for j in range(H)]
        self.bout = sig(W_W, P["bout"], "bout")
        self.mem_w0 = Memory(shape=signed(W_W), depth=H,
                             init=[int(v) for v in P["W0"]])
        self.mem_b0 = Memory(shape=signed(W_W), depth=H,
                             init=[int(v) for v in P["b0"]])
        self.mem_mos = Memory(shape=signed(W_MS), depth=H, init=[])
        self.mem_a0s = Memory(shape=signed(W_CR), depth=H, init=[])
        # 速度 5 群を 1 本に: addr = kind*H + unit (kind: 0=b1 1=wout 2=w0
        # 3=b0), addr 4H = bout。UPDV ストリームの添字と一致する。
        self.mem_vall = Memory(shape=signed(W_V), depth=4 * H + 1, init=[])
        # 提示内テンポラリの H 長ベクトル (書き込み・読み出しともストリーム)
        self.mem_sl0 = Memory(shape=signed(W_SL), depth=H, init=[])
        self.mem_sl1 = Memory(shape=signed(W_SL), depth=H, init=[])
        self.mem_den1 = Memory(shape=15, depth=H, init=[])
        self.mem_deno = Memory(shape=15, depth=H, init=[])
        self.mem_syz = Memory(shape=signed(25), depth=H, init=[])
        self.mem_sd1 = Memory(shape=signed(24), depth=H, init=[])

        self.lfsr0 = [GaloisLfsr(24) for _ in range(H)]
        self.lfsr1 = [GaloisLfsr(24) for _ in range(H)]
        self.eng0 = CrossingEngine(H, W_D + 2, T, hq)
        self.eng1 = CrossingEngine(H, W_D + 2, T, hq)
        self.div_c = RoundDivPipe(30, 15)
        self.div_r = RoundDivPipe(30, 15)

        # テストベンチ用の覗き穴 (addr は未駆動 = tb から設定できる)
        self.dbg_rp = {name: [mem.read_port(domain="sync") for mem in mems]
                       for name, mems in [("w1", self.mem_w1),
                                          ("vw1", self.mem_vw1),
                                          ("m1s", self.mem_m1s),
                                          ("vmr", self.mem_vmr),
                                          ("w0", [self.mem_w0]),
                                          ("b0", [self.mem_b0]),
                                          ("mos", [self.mem_mos]),
                                          ("vall", [self.mem_vall])]}

    def elaborate(self, platform):
        m = Module()
        H, T, L, R = self.H, self.T, self.L, self.R
        Fw, Fd, Nn, Ge = self.Fw, self.Fd, self.Nn, self.Ge
        for name, subs in [("lfsr0", self.lfsr0), ("lfsr1", self.lfsr1),
                           ("eng0", [self.eng0]), ("eng1", [self.eng1]),
                           ("divc", [self.div_c]), ("divr", [self.div_r]),
                           ("m_w1", self.mem_w1), ("m_vw1", self.mem_vw1),
                           ("m_m1s", self.mem_m1s), ("m_vmr", self.mem_vmr),
                           ("m_adz", self.mem_adz),
                           ("m_w0", [self.mem_w0]), ("m_b0", [self.mem_b0]),
                           ("m_mos", [self.mem_mos]),
                           ("m_a0s", [self.mem_a0s]),
                           ("m_vall", [self.mem_vall]),
                           ("m_sl0", [self.mem_sl0]),
                           ("m_sl1", [self.mem_sl1]),
                           ("m_den1", [self.mem_den1]),
                           ("m_deno", [self.mem_deno]),
                           ("m_syz", [self.mem_syz]),
                           ("m_sd1", [self.mem_sd1])]:
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
        rp_w0, wp_w0 = ports([self.mem_w0])
        rp_b0, wp_b0 = ports([self.mem_b0])
        rp_mos, wp_mos = ports([self.mem_mos])
        rp_a0s, wp_a0s = ports([self.mem_a0s])
        rp_va, wp_va = ports([self.mem_vall])
        rp_sl0, wp_sl0 = ports([self.mem_sl0])
        rp_sl1, wp_sl1 = ports([self.mem_sl1])
        rp_den1, wp_den1 = ports([self.mem_den1])
        rp_deno, wp_deno = ports([self.mem_deno])
        rp_syz, wp_syz = ports([self.mem_syz])
        rp_sd1, wp_sd1 = ports([self.mem_sd1])

        # ---- 提示ごとの一時レジスタ ----
        d0 = [Signal(signed(W_D), name=f"d0_{i}") for i in range(H)]
        d1 = [Signal(signed(W_D), name=f"d1_{j}") for j in range(H)]
        code0_l = [Signal(2, name=f"c0l_{i}") for i in range(H)]
        code1_l = [Signal(2, name=f"c1l_{j}") for j in range(H)]
        szr0 = [Signal((2 * T).bit_length(), name=f"szr0_{i}")
                for i in range(H)]                  # Σcode0 の複製 (UPKP 用)
        d1acc = [Signal(signed(26), name=f"d1a_{j}") for j in range(H)]
        ysum = Signal(signed(24))
        ys_acc = Signal(signed(26))                 # 出力 MAC の直列累算
        ysv = Signal(signed(W_YS))                  # 確定 ys (SYZS 用)
        wrp = Signal()                              # 巡回パス処理中
        e = Signal(signed(19))
        sc20d = Signal((4 * T).bit_length())        # DEN: Σcode² プリフェッチ
        sc21d = Signal((4 * T).bit_length())
        dd1 = [Signal(signed(W_CR), name=f"dd1_{j}") for j in range(H)]
        a0acc = [Signal(signed(38), name=f"a0a_{i}") for i in range(H)]
        kc = Signal(range(T + 1))
        pk = Signal(range(T))
        bc = Signal(range(H))
        rc = Signal(range(R + 1))
        jc = Signal(range(H + 1))
        ph = Signal()
        # ---- 除算ストリーミング用 (SL/MO: ic 発行 oc 回収, M1: 3 重) ----
        ic = Signal(range(H + 1))
        oc = Signal(range(H))
        lc = Signal(range(L))                       # M1 発行前段 (jc, rc, lc)
        vi = Signal()                               # M1 発行段 (前段 +1cyc)
        li = Signal(range(L))
        jq = Signal(range(H))                       # M1 出力前段 (pre_valid)
        rq = Signal(range(R))
        lq = Signal(range(L))
        wlane = Signal(range(L))                    # M1 書き戻し (出力段)
        waddr = Signal(range(R * H))
        m1fin = Signal()
        # ---- ユニット毎ベクトル演算のストリーミング用 (START/DEN/CR/A0X/UPDV)
        # 演算器をフェーズあたり 1 実体に落とすための共有パイプラインレジスタ
        kd = Signal(range(6))                       # UPDV の対象種別カウンタ
        ic0 = Signal(range(H))                      # RAM 読み出しの前段 (+1cyc)
        act0 = Signal()
        kd0 = Signal(3)
        b0d = Signal(signed(W_W))                   # START: b0 プリフェッチ
        vold0 = Signal(signed(W_V))                 # UPDV: 速度プリフェッチ
        i1 = Signal(range(H))
        i2 = Signal(range(H))
        i3 = Signal(range(H))
        v1 = Signal()
        v2 = Signal()
        v3 = Signal()
        k1 = Signal(3)
        k2 = Signal(3)
        k3 = Signal(3)
        st_p = Signal(signed(36))                   # 積ステージ 1
        st_p2 = Signal(signed(36))                  # 積ステージ 2
        # M1/MO の発行段分割 (積・RAM 読み値を 1 段ラッチしてから除算器へ)
        isp = Signal(signed(34))                    # Σ×sum の積
        isa = Signal(signed(26))                    # Σdz / syz 読み値
        isd = Signal(15)                            # 分母読み値
        isv = Signal()
        # M1/MO の EMA 書き戻し段分割 (商の飽和・旧値ラッチ → 翌 cyc 書き込み)
        mwh = Signal(signed(W_W))                   # sat(q)
        mold = Signal(signed(W_MS))                 # EMA 旧値
        mwv = Signal()                              # 書き戻し有効
        mwa = Signal(range(R * H))                  # 書き戻しアドレス
        mwl = Signal(range(L))                      # 書き戻しレーン (M1)
        m1fin2 = Signal()
        mofin = Signal()
        a0d = Signal(signed(W_CR))                  # A0X: a0 段ラッチ
        sl0d = Signal(signed(W_SL))                 # A0X: slope0 読み値
        bcd = Signal(range(H))                      # A0L: bc 遅延 (dd1 用)
        ar1 = Signal(range(R))                      # A0L: rc 遅延チェーン
        ar2 = Signal(range(R))
        ar3 = Signal(range(R))
        ab1 = Signal()                              # A0L: bc==0 フラグ遅延
        ab2 = Signal()
        ab3 = Signal()
        st_g = Signal(signed(24))                   # UPDV: 乗算不要種別の g
        st_vn = Signal(signed(W_V))                 # UPDV: 更新後速度
        st_pq = Signal(signed(28))                  # UPDV: vn×romq
        # ---- レーン共有 ALU (UPW1 / UPKP / A0L で状態間共有, 2 DSP/レーン)
        ga = [Signal(signed(17), name=f"ga{l}") for l in range(L)]
        gb = Signal(signed(15))
        gprod = [Signal(signed(32), name=f"gp{l}") for l in range(L)]
        qprod = [Signal(signed(28), name=f"qp{l}") for l in range(L)]
        rga = [Signal(signed(17), name=f"rga{l}") for l in range(L)]
        rgb = Signal(signed(15))
        pg = [Signal(signed(32), name=f"pg{l}") for l in range(L)]
        vd = [Signal(signed(W_V), name=f"vd{l}") for l in range(L)]
        vn2 = [Signal(signed(W_V), name=f"vn2{l}") for l in range(L)]
        pq = [Signal(signed(28), name=f"pq{l}") for l in range(L)]
        a1d = Signal(range(R * H))                  # 更新 RMW パイプの addr 遅延
        a2d = Signal(range(R * H))
        a3d = Signal(range(R * H))
        a4d = Signal(range(R * H))
        uv1 = Signal()
        uv2 = Signal()
        uv3 = Signal()
        uv4 = Signal()
        updone = Signal()

        def lane(vec, l):
            """レーン l が受け持つスライスの動的 rc 選択."""
            return Array(vec[l * R:(l + 1) * R])[rc]

        addr = Signal(range(R * H))
        m.d.comb += addr.eq(rc * H + bc)
        addr_j = Signal(range(R * H))
        m.d.comb += addr_j.eq(rc * H + jc)

        # 交差エンジンの dn (E1 の添字でユニット毎レジスタを mux) と LFSR 歩進。
        # ノイズは LFSR 上位 Nn bit だけ mux してから共通の整形を 1 回行う。
        def noise(lfsrs, idx):
            nzb = Array([s.state[24 - Nn:24] for s in lfsrs])[idx]
            return ((nzb << 1) + 1 - (1 << Nn)).as_signed() << (Fd - Nn)

        m.d.comb += [
            self.eng0.dn.eq(Array(d0)[self.eng0.idx_e1]
                            + noise(self.lfsr0, self.eng0.idx_e1)),
            self.eng1.dn.eq(Array(d1)[self.eng1.idx_e1]
                            + noise(self.lfsr1, self.eng1.idx_e1))]
        for i in range(H):
            m.d.comb += [
                self.lfsr0[i].en.eq(self.eng0.adv & (self.eng0.idx_e1 == i)),
                self.lfsr1[i].en.eq(self.eng1.adv & (self.eng1.idx_e1 == i))]

        # code はエンジンの添字順出力をシフトで格納 (デコーダ不要)。
        # あわせて layer1 は出力 MAC を直列累算する (加算木の置き換え)。
        with m.If(self.eng0.code_valid):
            m.d.sync += [code0_l[H - 1].eq(self.eng0.code)] + [
                code0_l[i].eq(code0_l[i + 1]) for i in range(H - 1)]
        with m.If(self.eng1.code_valid):
            m.d.sync += [code1_l[H - 1].eq(self.eng1.code)] + [
                code1_l[j].eq(code1_l[j + 1]) for j in range(H - 1)]
            m.d.sync += ys_acc.eq(
                ys_acc + gated(Array(self.wout)[self.eng1.idx_e1],
                               self.eng1.code))

        code_bc = Array(code0_l)[bc]
        d1_bc = Array(d1)[bc]
        dd1_bc = Array(dd1)[bc]

        # レーン共有 ALU の実体 (単一インスタンス化のため comb Signal に固定):
        # gprod = g の積, qprod = vn×romq, vnc = sgdm 速度更新, stc = ステップ
        vnc = [Signal(signed(W_V), name=f"vnc{l}") for l in range(L)]
        stc = [Signal(signed(20), name=f"stc{l}") for l in range(L)]
        for l in range(L):
            m.d.comb += [
                gprod[l].eq(ga[l] * gb),
                qprod[l].eq(vn2[l] * self.romq),
                vnc[l].eq(saturate(vd[l] - rshift_round(vd[l], 4)
                                   + rshift_round(pg[l], self.sh_g1), W_V)),
                stc[l].eq(rshift_round(pq[l], self.sh_step))]

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
                                 ph.eq(0), ic.eq(0), v1.eq(0), act0.eq(0),
                                 wrp.eq(0)]
                    m.next = "START"
            with m.State("START"):                    # d0 (乗算 1 実体で 1/cyc)
                m.d.comb += [rp_w0[0].addr.eq(ic), rp_b0[0].addr.eq(ic)]
                m.d.sync += [ic0.eq(ic), act0.eq(ic < H)]
                with m.If(ic < H):
                    m.d.sync += ic.eq(ic + 1)
                m.d.sync += [st_p.eq(rp_w0[0].data * self.xq),
                             b0d.eq(rp_b0[0].data),
                             i1.eq(ic0), v1.eq(act0)]
                with m.If(v1):
                    m.d.sync += Array(d0)[i1].eq(saturate(
                        rshift_round(st_p, Fw)
                        + rshift_round(b0d, self.sh_b), W_D))
                    with m.If(i1 == H - 1):
                        m.d.sync += ic.eq(0)
                        m.next = "IN0S"
            with m.State("IN0S"):                     # layer0 交差 (1 ユニット/cyc)
                m.d.comb += [self.eng0.idx.eq(ic),
                             self.eng0.in_valid.eq(ic < H),
                             self.eng0.first_smp.eq(kc == 0)]
                with m.If(ic < H):
                    m.d.sync += ic.eq(ic + 1)
                with m.If(ic == H):                   # 最終要素の E1 完了
                    m.d.sync += [kc.eq(kc + 1), ic.eq(0)]
                    with m.If(kc == 0):
                        m.next = "IN0S"
                    with m.Else():
                        m.d.sync += [pk.eq(kc - 1), bc.eq(0), rc.eq(0),
                                     ph.eq(0)]
                        m.next = "BC_CODE"
            with m.State("WRAP0S"):                   # layer0 巡回 + szr0 複製
                m.d.comb += [self.eng0.idx.eq(ic),
                             self.eng0.in_valid.eq(ic < H),
                             self.eng0.wrap.eq(1)]
                with m.If(self.eng0.code_valid):
                    m.d.sync += [szr0[H - 1].eq(self.eng0.sum_out)] + [
                        szr0[i].eq(szr0[i + 1]) for i in range(H - 1)]
                with m.If(ic < H):
                    m.d.sync += ic.eq(ic + 1)
                with m.If(ic == H):
                    m.d.sync += [pk.eq(T - 1), bc.eq(0), rc.eq(0), ph.eq(0),
                                 ic.eq(0)]
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
                m.d.sync += [ic.eq(0), ys_acc.eq(0)]
                m.next = "IN1S"
            with m.State("IN1S"):                     # layer1 交差 + ys 直列 MAC
                m.d.comb += [self.eng1.idx.eq(ic),
                             self.eng1.in_valid.eq(ic < H),
                             self.eng1.first_smp.eq(pk == 0)]
                with m.If(ic < H):
                    m.d.sync += ic.eq(ic + 1)
                with m.If(ic == H):
                    m.d.sync += ic.eq(0)
                    m.next = "YSC"
            with m.State("WRAP1S"):                   # layer1 巡回
                m.d.comb += [self.eng1.idx.eq(ic),
                             self.eng1.in_valid.eq(ic < H),
                             self.eng1.wrap.eq(1)]
                with m.If(ic < H):
                    m.d.sync += ic.eq(ic + 1)
                with m.If(ic == H):
                    m.d.sync += ic.eq(0)
                    m.next = "YSC"
            with m.State("YSC"):                      # ys 確定 + ysum 累算
                ys_v = saturate(rshift_round(ys_acc, self.sh_mac)
                                + rshift_round(self.bout, self.sh_b), W_YS)
                m.d.sync += [ysv.eq(ys_v), ic.eq(0), act0.eq(0)]
                with m.If(pk != 0):
                    m.d.sync += ysum.eq(ysum + ys_v)
                m.next = "SYZS"
            with m.State("SYZS"):                     # syz/sd1 の RMW ストリーム
                m.d.comb += [rp_syz[0].addr.eq(ic), rp_sd1[0].addr.eq(ic)]
                m.d.sync += [ic0.eq(ic), act0.eq(ic < H)]
                with m.If(ic < H):
                    m.d.sync += ic.eq(ic + 1)
                with m.If(act0):
                    with m.If(~wrp):                  # sd1 += d1 (巡回パス以外)
                        m.d.comb += [
                            wp_sd1[0].en.eq(1), wp_sd1[0].addr.eq(ic0),
                            wp_sd1[0].data.eq(Mux(
                                pk == 0, Array(d1)[ic0],
                                rp_sd1[0].data + Array(d1)[ic0]))]
                    with m.If((pk != 0) | wrp):       # syz += ys·z (code あり)
                        v_g = gated(ysv, Array(code1_l)[ic0])
                        m.d.comb += [
                            wp_syz[0].en.eq(1), wp_syz[0].addr.eq(ic0),
                            wp_syz[0].data.eq(Mux(
                                (pk == 1) & ~wrp, v_g, rp_syz[0].data + v_g))]
                    with m.If(ic0 == H - 1):
                        with m.If(wrp):
                            m.next = "YE"
                        with m.Else():
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
                        m.d.sync += [rc.eq(0), bc.eq(0), ic.eq(0)]
                        with m.If(pk == T - 1):
                            m.d.sync += [ys_acc.eq(0), wrp.eq(1)]
                            m.next = "WRAP1S"
                        with m.Elif(kc == T):
                            m.next = "WRAP0S"
                        with m.Else():
                            m.next = "IN0S"
                    with m.Elif(rc == R - 1):
                        m.d.sync += [rc.eq(0), bc.eq(bc + 1)]
                    with m.Else():
                        m.d.sync += rc.eq(rc + 1)
            with m.State("YE"):
                m.d.sync += [self.y.eq(rshift_round(ysum, self.logT)),
                             ic.eq(0), v1.eq(0), act0.eq(0), wrp.eq(0)]
                m.next = "DEN"
            with m.State("DEN"):                      # 分散 (乗算 2 実体で 1/cyc)
                m.d.comb += [self.eng0.stat_addr.eq(ic),
                             self.eng1.stat_addr.eq(ic)]
                m.d.sync += [ic0.eq(ic), act0.eq(ic < H)]
                with m.If(ic < H):
                    m.d.sync += ic.eq(ic + 1)
                m.d.sync += [st_p.eq(self.eng0.stat_sum * self.eng0.stat_sum),
                             st_p2.eq(self.eng1.stat_sum
                                      * self.eng1.stat_sum),
                             sc20d.eq(self.eng0.stat_sum2),
                             sc21d.eq(self.eng1.stat_sum2),
                             i1.eq(ic0), v1.eq(act0)]
                with m.If(v1):
                    dv0 = (sc20d << self.logT) - st_p
                    dv1 = (sc21d << self.logT) - st_p2
                    m.d.comb += [
                        wp_den1[0].en.eq(1), wp_den1[0].addr.eq(i1),
                        wp_den1[0].data.eq(Mux(dv0 < 1, 1, dv0)),
                        wp_deno[0].en.eq(1), wp_deno[0].addr.eq(i1),
                        wp_deno[0].data.eq(Mux(dv1 < 1, 1, dv1))]
                    with m.If(i1 == H - 1):
                        m.d.sync += [rc.eq(0), ic.eq(0), oc.eq(0),
                                     act0.eq(0)]
                        m.next = "SL"
            with m.State("SL"):                       # slope (両除算器に 1/cyc)
                m.d.sync += e.eq((self.y - self.tq) << 1)
                m.d.comb += [self.eng0.stat_addr.eq(ic),
                             self.eng1.stat_addr.eq(ic)]
                m.d.sync += [ic0.eq(ic), act0.eq(ic < H)]
                with m.If(ic < H):
                    m.d.sync += ic.eq(ic + 1)
                with m.If(act0):
                    m.d.comb += [self.div_c.in_valid.eq(1),
                                 self.div_c.num.eq(
                                     self.eng0.stat_cdiff << (2 * Fd)),
                                 self.div_c.den.eq(self.den_slope),
                                 self.div_r.in_valid.eq(1),
                                 self.div_r.num.eq(
                                     self.eng1.stat_cdiff << (2 * Fd)),
                                 self.div_r.den.eq(self.den_slope)]
                with m.If(self.div_c.out_valid):
                    m.d.comb += [
                        wp_sl0[0].en.eq(1), wp_sl0[0].addr.eq(oc),
                        wp_sl0[0].data.eq(saturate(self.div_c.q, W_SL)),
                        wp_sl1[0].en.eq(1), wp_sl1[0].addr.eq(oc),
                        wp_sl1[0].data.eq(saturate(self.div_r.q, W_SL))]
                    with m.If(oc == H - 1):
                        m.d.sync += [ic.eq(0), oc.eq(0)]
                        m.next = "MO"
                    with m.Else():
                        m.d.sync += oc.eq(oc + 1)
            with m.State("MO"):                       # out mirror (div_r に 1/cyc)
                m.d.comb += [rp_deno[0].addr.eq(ic),  # 発行前段
                             rp_syz[0].addr.eq(ic),
                             self.eng1.stat_addr.eq(ic)]
                m.d.sync += [ic0.eq(ic), act0.eq(ic < H)]
                with m.If(ic < H):
                    m.d.sync += ic.eq(ic + 1)
                # S1: 積と RAM 読み値をラッチ / S2: num 確定 → 除算器
                m.d.sync += [isp.eq(ysum * self.eng1.stat_sum),
                             isa.eq(rp_syz[0].data),
                             isd.eq(rp_deno[0].data), isv.eq(act0)]
                num_o = saturate((isa << self.logT) - isp, W_NUM)
                m.d.comb += [self.div_r.in_valid.eq(isv),
                             self.div_r.num.eq(num_o << self.sh_num),
                             self.div_r.den.eq(isd)]
                # 出力 1cyc 前に旧値読み (連続出力中は次要素 oc+1 を先行駆動)
                m.d.comb += rp_mos[0].addr.eq(
                    Mux(self.div_r.out_valid, oc + 1, oc))
                m.d.sync += mwv.eq(self.div_r.out_valid)
                with m.If(self.div_r.out_valid):    # W1: 商の飽和と旧値ラッチ
                    m.d.sync += [mwh.eq(saturate(self.div_r.q, W_W)),
                                 mold.eq(rp_mos[0].data), mwa.eq(oc)]
                    with m.If(oc == H - 1):
                        m.d.sync += mofin.eq(1)
                    with m.Else():
                        m.d.sync += oc.eq(oc + 1)
                with m.If(mwv):                     # W2: EMA 書き込み
                    m.d.comb += [wp_mos[0].en.eq(1), wp_mos[0].addr.eq(mwa),
                                 wp_mos[0].data.eq(Mux(
                                     self.first, mwh << Ge,
                                     saturate(mold + rshift_round(
                                         (mwh << Ge) - mold, self.k_ema),
                                         W_MS)))]
                with m.If(mofin & mwv):
                    m.d.sync += [jc.eq(0), rc.eq(0), lc.eq(0), vi.eq(0),
                                 jq.eq(0), rq.eq(0), lq.eq(0), m1fin.eq(0),
                                 mofin.eq(0), mwv.eq(0)]
                    m.next = "M1"
            with m.State("M1"):                       # 隠れ mirror (div_c に 1/cyc)
                # 前段: (jc, rc, lc) が Σdz / den1 / sd1 / Σcode0 読みを先行駆動
                for l in range(L):
                    m.d.comb += rp_adz[l].addr.eq(addr_j)
                m.d.comb += [rp_den1[0].addr.eq(lc * R + rc),
                             rp_sd1[0].addr.eq(jc),
                             self.eng0.stat_addr.eq(lc * R + rc)]
                m.d.sync += [vi.eq(jc < H), li.eq(lc)]
                with m.If(jc < H):
                    with m.If(lc == L - 1):
                        m.d.sync += lc.eq(0)
                        with m.If(rc == R - 1):
                            m.d.sync += [rc.eq(0), jc.eq(jc + 1)]
                        with m.Else():
                            m.d.sync += rc.eq(rc + 1)
                    with m.Else():
                        m.d.sync += lc.eq(lc + 1)
                # S1: 積と RAM 読み値をラッチ (乗算はフェーズ全体で 1 実体)
                m.d.sync += [
                    isp.eq(rp_sd1[0].data * self.eng0.stat_sum),
                    isa.eq(Array([rp.data for rp in rp_adz])[li]),
                    isd.eq(rp_den1[0].data), isv.eq(vi)]
                # S2: num1 確定 → 除算器
                num1 = saturate((isa << self.logT) - isp, W_NUM)
                m.d.comb += [self.div_c.in_valid.eq(isv),
                             self.div_c.num.eq(num1 << self.sh_num),
                             self.div_c.den.eq(isd)]
                # 出力前段: pre_valid で M1s 旧値をプリフェッチ
                for l in range(L):
                    m.d.comb += rp_m1s[l].addr.eq(rq * H + jq)
                with m.If(self.div_c.pre_valid):
                    m.d.sync += [wlane.eq(lq), waddr.eq(rq * H + jq)]
                    with m.If((jq == H - 1) & (rq == R - 1) & (lq == L - 1)):
                        m.d.sync += m1fin.eq(1)
                    with m.Elif(lq == L - 1):
                        m.d.sync += lq.eq(0)
                        with m.If(rq == R - 1):
                            m.d.sync += [rq.eq(0), jq.eq(jq + 1)]
                        with m.Else():
                            m.d.sync += rq.eq(rq + 1)
                    with m.Else():
                        m.d.sync += lq.eq(lq + 1)
                # 出力段 W1: 商の飽和と旧値・宛先のラッチ
                m.d.sync += mwv.eq(self.div_c.out_valid)
                with m.If(self.div_c.out_valid):
                    m.d.sync += [
                        mwh.eq(saturate(self.div_c.q, W_W)),
                        mold.eq(Array([rp.data for rp in rp_m1s])[wlane]),
                        mwa.eq(waddr), mwl.eq(wlane)]
                    with m.If(m1fin):
                        m.d.sync += m1fin2.eq(1)
                # 出力段 W2: EMA 書き込み
                with m.If(mwv):
                    for l in range(L):
                        m.d.comb += [wp_m1s[l].en.eq(mwl == l),
                                     wp_m1s[l].addr.eq(mwa),
                                     wp_m1s[l].data.eq(Mux(
                                         self.first, mwh << Ge,
                                         saturate(mold + rshift_round(
                                             (mwh << Ge) - mold, self.k_ema),
                                             W_MS)))]
                with m.If(m1fin2 & mwv):
                    m.d.sync += [jc.eq(0), rc.eq(0), ic.eq(0),
                                 v1.eq(0), v2.eq(0), act0.eq(0),
                                 m1fin2.eq(0), mwv.eq(0)]
                    m.next = "CR"
            with m.State("CR"):                       # credit (乗算 2 実体 3 段)
                m.d.comb += [rp_mos[0].addr.eq(ic), rp_sl1[0].addr.eq(ic0)]
                m.d.sync += [ic0.eq(ic), act0.eq(ic < H)]
                with m.If(ic < H):
                    m.d.sync += ic.eq(ic + 1)
                m.d.sync += [st_p.eq(e * rshift_round(rp_mos[0].data, Ge)),
                             i1.eq(ic0), v1.eq(act0)]
                a1v = saturate(rshift_round(st_p, Fw), W_CR)
                m.d.sync += [st_p2.eq(a1v * rp_sl1[0].data),
                             i2.eq(i1), v2.eq(v1)]
                with m.If(v2):
                    m.d.sync += Array(dd1)[i2].eq(saturate(
                        rshift_round(st_p2, Fd), W_CR))
                    with m.If(i2 == H - 1):
                        m.d.sync += [bc.eq(0), rc.eq(0), ph.eq(0),
                                     updone.eq(0)]
                        m.next = "A0L"
            with m.State("A0L"):                      # a0 = Σ M1u·dd1 (4 段パイプ)
                for l in range(L):
                    m.d.comb += [rp_m1s[l].addr.eq(addr),
                                 ga[l].eq(rga[l])]
                m.d.comb += gb.eq(rgb)
                # S1: カウンタ遅延 / S2: オペランドラッチ / S3: 積 / S4: 累算
                m.d.sync += [bcd.eq(bc), ar1.eq(rc), ab1.eq(bc == 0),
                             v1.eq(~updone)]
                for l in range(L):
                    m.d.sync += [rga[l].eq(rshift_round(rp_m1s[l].data, Ge)),
                                 pg[l].eq(gprod[l])]
                m.d.sync += [rgb.eq(Array(dd1)[bcd]),
                             ar2.eq(ar1), ab2.eq(ab1), v2.eq(v1),
                             ar3.eq(ar2), ab3.eq(ab2), v3.eq(v2)]
                with m.If(v3):
                    for l in range(L):
                        tgt = Array(a0acc[l * R:(l + 1) * R])[ar3]
                        m.d.sync += tgt.eq(Mux(ab3, pg[l], tgt + pg[l]))
                with m.If(~updone):
                    with m.If((rc == R - 1) & (bc == H - 1)):
                        m.d.sync += updone.eq(1)
                    with m.Elif(rc == R - 1):
                        m.d.sync += [rc.eq(0), bc.eq(bc + 1)]
                    with m.Else():
                        m.d.sync += rc.eq(rc + 1)
                with m.If(updone & ~v1 & ~v2 & ~v3):
                    m.d.sync += [bc.eq(0), rc.eq(0), updone.eq(0),
                                 ic.eq(0), act0.eq(0)]
                    m.next = "A0X"
            with m.State("A0X"):                      # a0s (乗算 1 実体で 1/cyc)
                m.d.comb += rp_sl0[0].addr.eq(ic)     # 発行前段
                m.d.sync += [ic0.eq(ic), act0.eq(ic < H)]
                with m.If(ic < H):
                    m.d.sync += ic.eq(ic + 1)
                # S1: a0acc の mux/丸めと slope0 読み値をラッチ (乗算と分離)
                m.d.sync += [
                    a0d.eq(saturate(rshift_round(Array(a0acc)[ic0], Fw),
                                    W_CR)),
                    sl0d.eq(rp_sl0[0].data), i1.eq(ic0), v1.eq(act0)]
                # S2: 積
                m.d.sync += [st_p.eq(a0d * sl0d), i2.eq(i1), v2.eq(v1)]
                with m.If(v2):
                    m.d.comb += [wp_a0s[0].en.eq(1), wp_a0s[0].addr.eq(i2),
                                 wp_a0s[0].data.eq(saturate(
                                     rshift_round(st_p, Fd), W_CR))]
                    with m.If(i2 == H - 1):
                        m.d.sync += [ic.eq(0), kd.eq(0),
                                     v1.eq(0), v2.eq(0), v3.eq(0)]
                        m.next = "UPDV"
            with m.State("UPDV"):
                # kd: 0=b1(g=dd1<<Gg) 1=wout(g=e·sz1>>) 2=w0(g=a0s·xq>>)
                #     3=b0(g=a0s<<Gg)  4=bout(g=e<<Gg)。5 段パイプライン
                # 前段: v_all / a0s / Σcode1 の読み出し発行
                m.d.comb += [rp_va[0].addr.eq(kd * H + ic),
                             rp_a0s[0].addr.eq(ic),
                             self.eng1.stat_addr.eq(ic)]
                m.d.sync += [kd0.eq(kd), ic0.eq(ic), act0.eq(kd <= 4)]
                with m.If(kd < 4):
                    with m.If(ic == H - 1):
                        m.d.sync += [ic.eq(0), kd.eq(kd + 1)]
                    with m.Else():
                        m.d.sync += ic.eq(ic + 1)
                with m.Elif(kd == 4):
                    m.d.sync += kd.eq(5)
                # S0: g の積 / パススルー
                sz1u = self.eng1.stat_sum
                a0s_v = rp_a0s[0].data
                g0 = Mux(kd0 == 0, Array(dd1)[ic0] << self.Gg,
                         Mux(kd0 == 3, a0s_v << self.Gg, e << self.Gg))
                m.d.sync += [
                    st_p.eq(Mux(kd0 == 1, e, a0s_v)
                            * Mux(kd0 == 1, sz1u, self.xq)),
                    st_g.eq(g0), vold0.eq(rp_va[0].data),
                    k1.eq(kd0), i1.eq(ic0), v1.eq(act0)]
                # S1: vn = v - (v>>4) + g, v_all 書き戻し
                g = Mux(k1 == 1, rshift_round(st_p, self.sh_g1),
                        Mux(k1 == 2, rshift_round(st_p, self.sh_g0), st_g))
                vn_s = saturate(vold0 - rshift_round(vold0, 4) + g, W_V)
                m.d.sync += [st_vn.eq(vn_s), k2.eq(k1), i2.eq(i1), v2.eq(v1)]
                with m.If(v1):
                    m.d.comb += [wp_va[0].en.eq(1),
                                 wp_va[0].addr.eq(k1 * H + i1),
                                 wp_va[0].data.eq(vn_s)]
                # S2: vn × lr(ROM) + 重み旧値のプリフェッチ
                m.d.sync += [st_pq.eq(st_vn * self.romq),
                             k3.eq(k2), i3.eq(i2), v3.eq(v2)]
                m.d.comb += [rp_w0[0].addr.eq(i2), rp_b0[0].addr.eq(i2),
                             rp_mos[0].addr.eq(i2)]
                # S3: st を引いて重み書き戻し (wout は mos の KP も)
                st = rshift_round(st_pq, self.sh_step)
                with m.If(v3):
                    with m.Switch(k3):
                        with m.Case(0):
                            tgt = Array(self.b1)[i3]
                            m.d.sync += tgt.eq(saturate(tgt - st, W_W))
                        with m.Case(1):
                            tgt = Array(self.wout)[i3]
                            m.d.sync += tgt.eq(saturate(tgt - st, W_W))
                            m.d.comb += [
                                wp_mos[0].en.eq(1), wp_mos[0].addr.eq(i3),
                                wp_mos[0].data.eq(saturate(
                                    rp_mos[0].data - (st << Ge), W_MS))]
                        with m.Case(2):
                            m.d.comb += [
                                wp_w0[0].en.eq(1), wp_w0[0].addr.eq(i3),
                                wp_w0[0].data.eq(saturate(
                                    rp_w0[0].data - st, W_W))]
                        with m.Case(3):
                            m.d.comb += [
                                wp_b0[0].en.eq(1), wp_b0[0].addr.eq(i3),
                                wp_b0[0].data.eq(saturate(
                                    rp_b0[0].data - st, W_W))]
                        with m.Case(4):
                            m.d.sync += self.bout.eq(
                                saturate(self.bout - st, W_W))
                    with m.If(k3 == 4):
                        m.d.sync += [bc.eq(0), rc.eq(0), ph.eq(0),
                                     updone.eq(0), uv1.eq(0), uv2.eq(0),
                                     uv3.eq(0), uv4.eq(0)]
                        m.next = "UPW1"
            with m.State("UPW1"):                     # W1/v 更新 (4 段 RMW パイプ)
                for l in range(L):
                    m.d.comb += [rp_vw1[l].addr.eq(addr),   # S0 読み出し発行
                                 rp_w1[l].addr.eq(a3d),     # S3 W1 プリフェッチ
                                 ga[l].eq(rga[l])]
                    m.d.sync += [rga[l].eq(lane(dd1, l)),   # S0 オペランドラッチ
                                 pg[l].eq(gprod[l]),        # S1 g 積
                                 vd[l].eq(rp_vw1[l].data),
                                 vn2[l].eq(vnc[l]),         # S2 速度更新
                                 pq[l].eq(qprod[l])]        # S3 vn×romq
                # Σcode0[bc] は統計ポートを S0 で駆動し S1 の積に直結
                m.d.comb += [self.eng0.stat_addr.eq(bc),
                             gb.eq(self.eng0.stat_sum)]
                m.d.sync += [a1d.eq(addr), a2d.eq(a1d), a3d.eq(a2d),
                             a4d.eq(a3d), uv1.eq(~updone), uv2.eq(uv1),
                             uv3.eq(uv2), uv4.eq(uv3)]
                with m.If(uv2):                             # S2 v 書き戻し
                    for l in range(L):
                        m.d.comb += [wp_vw1[l].en.eq(1),
                                     wp_vw1[l].addr.eq(a2d),
                                     wp_vw1[l].data.eq(vnc[l])]
                with m.If(uv4):                             # S4 W1 書き戻し
                    for l in range(L):
                        m.d.comb += [wp_w1[l].en.eq(1),
                                     wp_w1[l].addr.eq(a4d),
                                     wp_w1[l].data.eq(saturate(
                                         rp_w1[l].data - stc[l], W_W))]
                with m.If(~updone):
                    with m.If((rc == R - 1) & (bc == H - 1)):
                        m.d.sync += updone.eq(1)
                    with m.Elif(rc == R - 1):
                        m.d.sync += [rc.eq(0), bc.eq(bc + 1)]
                    with m.Else():
                        m.d.sync += rc.eq(rc + 1)
                with m.If(updone & ~uv1 & ~uv2 & ~uv3 & ~uv4):
                    m.d.sync += [bc.eq(0), rc.eq(0), updone.eq(0)]
                    m.next = "UPKP"
            with m.State("UPKP"):                     # KP: vM 複製 (4 段 RMW パイプ)
                for l in range(L):
                    sz0_i = Array(szr0[l * R:(l + 1) * R])[rc]
                    m.d.comb += [rp_vmr[l].addr.eq(addr),
                                 rp_m1s[l].addr.eq(a3d),
                                 ga[l].eq(rga[l])]
                    m.d.sync += [rga[l].eq(sz0_i),
                                 pg[l].eq(gprod[l]),
                                 vd[l].eq(rp_vmr[l].data),
                                 vn2[l].eq(vnc[l]),
                                 pq[l].eq(qprod[l])]
                m.d.comb += gb.eq(rgb)
                m.d.sync += [rgb.eq(dd1_bc),
                             a1d.eq(addr), a2d.eq(a1d), a3d.eq(a2d),
                             a4d.eq(a3d), uv1.eq(~updone), uv2.eq(uv1),
                             uv3.eq(uv2), uv4.eq(uv3)]
                with m.If(uv2):                             # S2 vM 書き戻し
                    for l in range(L):
                        m.d.comb += [wp_vmr[l].en.eq(1),
                                     wp_vmr[l].addr.eq(a2d),
                                     wp_vmr[l].data.eq(vnc[l])]
                with m.If(uv4):                             # S4 M1s KP 書き戻し
                    for l in range(L):
                        m.d.comb += [wp_m1s[l].en.eq(1),
                                     wp_m1s[l].addr.eq(a4d),
                                     wp_m1s[l].data.eq(saturate(
                                         rp_m1s[l].data - (stc[l] << Ge),
                                         W_MS))]
                with m.If(~updone):
                    with m.If((rc == R - 1) & (bc == H - 1)):
                        m.d.sync += updone.eq(1)
                    with m.Elif(rc == R - 1):
                        m.d.sync += [rc.eq(0), bc.eq(bc + 1)]
                    with m.Else():
                        m.d.sync += rc.eq(rc + 1)
                with m.If(updone & ~uv1 & ~uv2 & ~uv3 & ~uv4):
                    m.d.sync += [bc.eq(0), rc.eq(0), updone.eq(0)]
                    m.next = "DONE"
            with m.State("DONE"):
                m.d.comb += self.done.eq(1)
                m.d.sync += self.first.eq(0)
                m.next = "IDLE"
        return m
