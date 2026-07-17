"""交差活性ユニット (ストリーミング): 2 値化 -> 巡回 XOR -> code {0,1,2} + 統計。

ゴールデンモデル crossing_code(dn, h) と bit-exact:
    b1[t] = dn[t] > h,  b2[t] = dn[t] > -h
    code[t] = (b1[t+1] ^ b1[t]) + (b2[t+1] ^ b2[t])   (t = T-1 は b[0] と巡回)
    cdiff   = Σ_t (x2[t] - x1[t])                     (同一パス KDE slope 分子)
    sum_code = Σ code,  sum_code2 = Σ code²           (mirror の分母統計)

タイミング: start パルスで窓を開始し、in_valid とともに dn を T 個流す。
サンプル k (k>=1) の入力サイクルに code[k-1] を組み合わせで出力し、
最終サンプルの次のサイクルに巡回項 code[T-1] を出して done を立てる。
出力順は添字順 (code[0], ..., code[T-1]) なので、下流 (W1 MAC / Σdz) は
ゴールデンモデルの t と同じ対応で消費できる。
"""
from amaranth.hdl import Elaboratable, Module, Signal, signed


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
