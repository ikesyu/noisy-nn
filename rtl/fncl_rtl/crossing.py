"""交差活性 (2 値化 -> 巡回 XOR -> code {0,1,2} + 統計)。

ゴールデンモデル crossing_code(dn, h) と bit-exact:
    b1[t] = dn[t] > h,  b2[t] = dn[t] > -h
    code[t] = (b1[t+1] ^ b1[t]) + (b2[t+1] ^ b2[t])   (t = T-1 は b[0] と巡回)
    cdiff   = Σ_t (x2[t] - x1[t])                     (同一パス KDE slope 分子)
    sum_code = Σ code,  sum_code2 = Σ code²           (mirror の分母統計)

2 実装:
  - CrossingUnit:   ユニット毎並列 (Step 1-4 の構成)。start/in_valid/wrap_en。
  - CrossingEngine: H ユニットを 1 ユニット/cyc で時分割する Step 5 の置き換え。
    ユニット毎状態 {prev, first, cdiff, Σcode, Σcode²} を深さ H の Memory に
    持ち、E0 (状態読み出し) → E1 (比較・XOR・累算・書き戻し) の 2 段。
    統計はサンプル 0 (first_smp) の書き込みが蓄積 0 から始めるので RAM の
    クリアは不要。サンプル計数・巡回判定は呼び出し側 FSM が行う。
"""
from amaranth.hdl import Cat, Elaboratable, Module, Mux, Signal, signed
from amaranth.lib.memory import Memory


class CrossingUnit(Elaboratable):
    def __init__(self, w_d: int, T: int, h: int):
        assert h > 0
        self.T, self.h = T, h
        self.start = Signal()                       # 窓の開始 (統計リセット)
        self.in_valid = Signal()
        self.wrap_en = Signal(init=1)               # 巡回項の出力許可 (保留可)
        self.dn = Signal(signed(w_d))               # 前活性 + ノイズ (Fd)
        self.code_valid = Signal()
        self.code = Signal(2)                       # {0, 1, 2} (z = code/2)
        self.done = Signal()                        # 巡回項の出力サイクル
        self.cdiff = Signal(signed(T.bit_length() + 1))
        self.sum_code = Signal(range(2 * T + 1))
        self.sum_code2 = Signal(range(4 * T + 1))

    def elaborate(self, platform):
        m = Module()
        T = self.T
        b1 = Signal()
        b2 = Signal()
        m.d.comb += [b1.eq(self.dn > self.h), b2.eq(self.dn > -self.h)]

        k = Signal(range(T + 1))                    # 受け取ったサンプル数
        prev1, prev2 = Signal(), Signal()
        first1, first2 = Signal(), Signal()
        wrap = Signal()                             # 巡回項を出す番

        x1 = Signal()
        x2 = Signal()
        with m.If(wrap & self.wrap_en):             # code[T-1] = b[0] ^ b[T-1]
            m.d.comb += [x1.eq(first1 ^ prev1), x2.eq(first2 ^ prev2)]
            m.d.comb += [self.code_valid.eq(1), self.done.eq(1)]
            m.d.sync += wrap.eq(0)
        with m.Elif(~wrap & self.in_valid):
            m.d.comb += [x1.eq(b1 ^ prev1), x2.eq(b2 ^ prev2)]
            m.d.comb += self.code_valid.eq(k != 0)  # code[k-1]
            m.d.sync += [prev1.eq(b1), prev2.eq(b2)]
            with m.If(k == 0):
                m.d.sync += [first1.eq(b1), first2.eq(b2)]
            with m.If(k == T - 1):
                m.d.sync += [wrap.eq(1), k.eq(0)]
            with m.Else():
                m.d.sync += k.eq(k + 1)
        m.d.comb += self.code.eq(x1 + x2)

        with m.If(self.start):
            m.d.sync += [k.eq(0), wrap.eq(0), self.cdiff.eq(0),
                         self.sum_code.eq(0), self.sum_code2.eq(0)]
        with m.Elif(self.code_valid):
            m.d.sync += [
                self.cdiff.eq(self.cdiff + x2 - x1),
                self.sum_code.eq(self.sum_code + self.code),
                # code² は {0,1,4}: code==2 のとき 4
                self.sum_code2.eq(self.sum_code2 + (x1 & x2) * 4
                                  + (x1 ^ x2)),
            ]
        return m


class CrossingEngine(Elaboratable):
    """H ユニット時分割の交差活性エンジン (1 ユニット/cyc, 2 段)。

    E0: idx / in_valid / first_smp / wrap を駆動 (毎サイクル発行可)。
    E1: idx_e1 のユニットの dn を外部が組み合わせで与える (d + ノイズの
        mux)。code / code_valid / sum_out (確定 Σcode) が出る。
    stat_addr → stat_cdiff / stat_sum / stat_sum2: 窓終了後の統計読み出し
    (同期 1 サイクル, 専用ポート)。
    """

    def __init__(self, H, w_d: int, T: int, h: int):
        assert h > 0
        self.H, self.T, self.h = H, T, h
        self.wc = T.bit_length() + 1
        self.ws = (2 * T).bit_length()
        self.ws2 = (4 * T).bit_length()
        self.idx = Signal(range(H))
        self.in_valid = Signal()
        self.first_smp = Signal()                   # サンプル 0 (状態初期化)
        self.wrap = Signal()                        # 巡回パス (dn 不要)
        self.dn = Signal(signed(w_d))
        self.idx_e1 = Signal(range(H))
        self.adv = Signal()                         # E1 で実サンプル消費中
        self.code_valid = Signal()
        self.code = Signal(2)
        self.sum_out = Signal(self.ws)              # E1 確定の Σcode
        self.stat_addr = Signal(range(H))
        self.stat_cdiff = Signal(signed(self.wc))
        self.stat_sum = Signal(self.ws)
        self.stat_sum2 = Signal(self.ws2)
        self.mem = Memory(shape=4 + self.wc + self.ws + self.ws2, depth=H,
                          init=[])

    def elaborate(self, platform):
        m = Module()
        m.submodules.mem = self.mem
        wc, ws, ws2 = self.wc, self.ws, self.ws2
        rp = self.mem.read_port(domain="sync")
        sp = self.mem.read_port(domain="sync")
        wp = self.mem.write_port()
        m.d.comb += [rp.en.eq(1), rp.addr.eq(self.idx),
                     sp.en.eq(1), sp.addr.eq(self.stat_addr)]

        v1 = Signal()
        f1 = Signal()
        w1 = Signal()
        m.d.sync += [self.idx_e1.eq(self.idx), v1.eq(self.in_valid),
                     f1.eq(self.first_smp), w1.eq(self.wrap)]

        def fields(word):
            o = 4
            cdiff = word[o:o + wc].as_signed()
            ssum = word[o + wc:o + wc + ws]
            ssum2 = word[o + wc + ws:o + wc + ws + ws2]
            return word[0], word[1], word[2], word[3], cdiff, ssum, ssum2

        prev1, prev2, first1, first2, cdiff, ssum, ssum2 = fields(rp.data)
        b1 = Signal()
        b2 = Signal()
        m.d.comb += [b1.eq(self.dn > self.h), b2.eq(self.dn > -self.h)]
        x1 = Signal()
        x2 = Signal()
        m.d.comb += [
            x1.eq(Mux(w1, first1 ^ prev1, b1 ^ prev1)),
            x2.eq(Mux(w1, first2 ^ prev2, b2 ^ prev2)),
            self.code.eq(x1 + x2),
            self.code_valid.eq(v1 & (w1 | ~f1)),
            self.adv.eq(v1 & ~w1)]
        cd0 = Mux(f1, 0, cdiff)
        s0 = Mux(f1, 0, ssum)
        s20 = Mux(f1, 0, ssum2)
        cdn = Signal(signed(wc))
        ssn = Signal(ws)
        ss2n = Signal(ws2)
        m.d.comb += [
            cdn.eq(Mux(self.code_valid, cd0 + x2 - x1, cd0)),
            ssn.eq(Mux(self.code_valid, s0 + self.code, s0)),
            ss2n.eq(Mux(self.code_valid,
                        s20 + (x1 & x2) * 4 + (x1 ^ x2), s20)),
            self.sum_out.eq(ssn)]
        m.d.comb += [
            wp.en.eq(v1), wp.addr.eq(self.idx_e1),
            wp.data.eq(Cat(Mux(w1, prev1, b1), Mux(w1, prev2, b2),
                           Mux(f1, b1, first1), Mux(f1, b2, first2),
                           cdn, ssn, ss2n))]
        _, _, _, _, scd, ssm, ssm2 = fields(sp.data)
        m.d.comb += [self.stat_cdiff.eq(scd), self.stat_sum.eq(ssm),
                     self.stat_sum2.eq(ssm2)]
        return m
