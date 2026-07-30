"""丸め付き符号付き整数除算器 (mirror 用)。

ゴールデンモデル rdiv(num, den) = floor((2*num + den) / (2*den)) と bit-exact
(den > 0)。負の被除数は floor 除算なので
    N2 = 2*num + den,  D2 = 2*den
    N2 >= 0: q =  N2 // D2
    N2 <  0: q = -ceil(-N2 / D2) = -((-N2 + D2 - 1) // D2)
と符号を外してから復元付き除算する。2 実装:

  - RoundDiv:     直列 1 bit/サイクル (start/done, レイテンシ w_num+3)。
                  Step 2/3 の v0 構成用。
  - RoundDivPipe: 1 反復 = 1 ステージのパイプライン (in_valid/out_valid,
                  1 結果/サイクル, レイテンシ w_num+4)。Step 4 で 2L 基の
                  直列除算器をアレイ毎 1 本に集約するための実装。
                  被除数レジスタは反復毎に 1 bit 縮み、商は 1 bit 伸びる
                  (合計幅一定) ので FF は直列版 L 基分より小さい。
"""
from amaranth.hdl import Cat, Elaboratable, Module, Mux, Signal, signed


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


class RoundDivPipe(Elaboratable):
    """rdiv と bit-exact のパイプライン除算器 (1 結果/サイクル)。

    in_valid とともに num/den を与えると w_num+4 サイクル後に out_valid と
    q が出る。発行は毎サイクル可能 (ストール無し)。pre_valid は out_valid の
    1 サイクル前 (書き戻し先メモリの同期読み出しプリフェッチ用)。
    """

    def __init__(self, w_num: int, w_den: int):
        self.w_num, self.w_den = w_num, w_den
        self.in_valid = Signal()
        self.num = Signal(signed(w_num))
        self.den = Signal(w_den)                    # > 0
        self.out_valid = Signal()
        self.pre_valid = Signal()
        self.q = Signal(signed(w_num + 1))
        self.LATENCY = w_num + 5                    # in_valid -> out_valid

    def elaborate(self, platform):
        m = Module()
        WA = self.w_num + 2
        S = WA + 1                                  # 反復回数 (直列版と同一)
        # 入力段レジスタ: 発行側の mux/乗算パスから a_in 前処理を切り離す
        num_r = Signal(signed(self.w_num))
        den_i = Signal(self.w_den)
        vin_r = Signal()
        m.d.sync += [num_r.eq(self.num), den_i.eq(self.den),
                     vin_r.eq(self.in_valid)]
        n2 = Signal(signed(WA + 1))
        m.d.comb += n2.eq((num_r << 1) + den_i)
        a_in = Signal(WA + 1)
        m.d.comb += a_in.eq(Mux(n2 < 0, -n2 + (den_i << 1) - 1, n2))

        # ステージ k (0..S): k 反復済み。a は上位 k bit 消費済み、quo は k bit。
        a_r = [Signal(WA + 1 - k, name=f"a{k}") for k in range(S)]
        r_r = [Signal(self.w_den + 2, name=f"r{k}") for k in range(S + 1)]
        q_r = [Signal(max(k, 1), name=f"q{k}") for k in range(S + 1)]
        d_r = [Signal(self.w_den, name=f"d{k}") for k in range(S)]
        neg_r = [Signal(name=f"ng{k}") for k in range(S + 1)]
        v_r = [Signal(name=f"v{k}") for k in range(S + 1)]

        m.d.sync += [a_r[0].eq(a_in), r_r[0].eq(0), d_r[0].eq(den_i),
                     neg_r[0].eq(n2 < 0), v_r[0].eq(vin_r)]
        for k in range(S):
            sub = Signal(signed(self.w_den + 3), name=f"sub{k}")
            m.d.comb += sub.eq(((r_r[k] << 1) | a_r[k][-1]) - (d_r[k] << 1))
            m.d.sync += [
                r_r[k + 1].eq(Mux(sub[-1], (r_r[k] << 1) | a_r[k][-1], sub)),
                q_r[k + 1].eq(Cat(~sub[-1], q_r[k])[:max(k + 1, 1)]),
                neg_r[k + 1].eq(neg_r[k]), v_r[k + 1].eq(v_r[k])]
            if k + 1 < S:
                m.d.sync += [a_r[k + 1].eq(a_r[k][:-1]),
                             d_r[k + 1].eq(d_r[k])]

        quo = Signal(WA + 1)
        m.d.comb += [quo.eq(q_r[S]),
                     self.q.eq(Mux(neg_r[S], -quo, quo)),
                     self.out_valid.eq(v_r[S]),
                     self.pre_valid.eq(v_r[S - 1])]
        return m
