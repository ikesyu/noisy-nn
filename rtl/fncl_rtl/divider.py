"""丸め付き符号付き整数除算器 (mirror 用, 直列 1 bit/サイクル)。

ゴールデンモデル rdiv(num, den) = floor((2*num + den) / (2*den)) と bit-exact
(den > 0)。負の被除数は floor 除算なので
    N2 = 2*num + den,  D2 = 2*den
    N2 >= 0: q =  N2 // D2
    N2 <  0: q = -ceil(-N2 / D2) = -((-N2 + D2 - 1) // D2)
と符号を外してから復元付き直列除算する。レイテンシ w_num+2 サイクル。
パイプライン化 (1 結果/サイクル) は Phase 3 後半で置き換え可能なように
start/done ハンドシェイクにしてある。
"""
from amaranth.hdl import Elaboratable, Module, Mux, Signal, signed


class RoundDiv(Elaboratable):
    def __init__(self, w_num: int, w_den: int):
        self.w_num, self.w_den = w_num, w_den
        self.start = Signal()
        self.num = Signal(signed(w_num))
        self.den = Signal(w_den)                    # > 0
        self.busy = Signal()
        self.done = Signal()                        # 1 サイクルパルス
        self.q = Signal(signed(w_num + 1))

    def elaborate(self, platform):
        m = Module()
        WA = self.w_num + 2                         # |N2| の最大幅
        n2 = Signal(signed(WA + 1))
        m.d.comb += n2.eq((self.num << 1) + self.den)
        neg = Signal()
        a_in = Signal(WA + 1)
        m.d.comb += a_in.eq(Mux(n2 < 0, -n2 + (self.den << 1) - 1, n2))

        den_r = Signal(self.w_den)                  # start で den をラッチ
        d2 = Signal(self.w_den + 1)
        m.d.comb += d2.eq(den_r << 1)
        a = Signal(WA + 1)                          # 被除数 (シフトレジスタ)
        r = Signal(self.w_den + 2)                  # 部分剰余
        quo = Signal(WA + 1)
        cnt = Signal(range(WA + 2))
        sub = Signal(self.w_den + 3)
        m.d.comb += sub.eq(((r << 1) | a[-1]) - d2)

        with m.If(self.start):
            m.d.sync += [a.eq(a_in), neg.eq(n2 < 0), r.eq(0), quo.eq(0),
                         den_r.eq(self.den),
                         cnt.eq(WA + 1), self.busy.eq(1), self.done.eq(0)]
        with m.Elif(self.busy):
            with m.If(sub[-1]):                     # 借り: 引けない
                m.d.sync += [r.eq((r << 1) | a[-1]), quo.eq(quo << 1)]
            with m.Else():
                m.d.sync += [r.eq(sub), quo.eq((quo << 1) | 1)]
            m.d.sync += [a.eq(a << 1), cnt.eq(cnt - 1)]
            with m.If(cnt == 1):
                m.d.sync += [self.busy.eq(0), self.done.eq(1)]
        with m.Else():
            m.d.sync += self.done.eq(0)
        m.d.comb += self.q.eq(Mux(neg, -quo, quo))
        return m
