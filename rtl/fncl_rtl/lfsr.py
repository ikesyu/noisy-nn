"""24 bit Galois LFSR (ユニット毎ノイズ源)。

ゴールデンモデル (fncl_phase1_fxp.lfsr_sequence / _lfsr_step) と同一の遷移:
    s' = (s >> 1) ^ (s[0] ? taps : 0)
状態そのものが一様乱数値。固定小数点ノイズは上位 Nn ビット n を取り
    nd = (2n + 1 - 2^Nn) << (Fd - Nn)
で d ドメインへ整列する (この整形は crossing 側/呼び出し側で行う)。
"""
from amaranth.hdl import Elaboratable, Module, Mux, Signal

TAPS = {16: 0xB400, 24: 0xE10000}


class GaloisLfsr(Elaboratable):
    def __init__(self, nbits: int = 24, taps: int = None):
        self.nbits = nbits
        self.taps = TAPS[nbits] if taps is None else taps
        self.load = Signal()               # seed を読み込む (en より優先)
        self.seed = Signal(nbits)
        self.en = Signal()                 # 1 ステップ進める
        self.state = Signal(nbits)         # 現在値 = 乱数出力

    def elaborate(self, platform):
        m = Module()
        nxt = (self.state >> 1) ^ Mux(self.state[0], self.taps, 0)
        with m.If(self.load):
            m.d.sync += self.state.eq(self.seed)
        with m.Elif(self.en):
            m.d.sync += self.state.eq(nxt)
        return m
