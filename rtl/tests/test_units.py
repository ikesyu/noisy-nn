"""基本モジュールのゴールデンモデル突き合わせ (bit-exact)。

実行: .venv/bin/python rtl/tests/test_units.py   (リポジトリルートから)

各テストは tmp/fncl_phase1_fxp.py の対応する関数
(lfsr_sequence / rshift / sat / crossing_code / rdiv) と完全一致を確認する。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rtl"))
sys.path.insert(0, str(ROOT / "tmp"))

import numpy as np
import torch
from amaranth.hdl import Elaboratable, Module, Signal, signed
from amaranth.sim import Simulator

from fncl_rtl import CrossingUnit, GaloisLfsr, RoundDiv, rshift_round, saturate
from fncl_rtl.divider import RoundDivPipe
from fncl_phase1_fxp import crossing_code, lfsr_sequence, rdiv, rshift, sat

rng = np.random.default_rng(0)


def run_sim(dut, tb, clock=True):
    sim = Simulator(dut)
    if clock:
        sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()


# ------------------------------------------------------------
def test_lfsr():
    N = 2048
    gold = lfsr_sequence(24, N).tolist()
    dut = GaloisLfsr(24)
    got = []

    async def tb(ctx):
        ctx.set(dut.seed, 1)
        ctx.set(dut.load, 1)
        await ctx.tick()
        ctx.set(dut.load, 0)
        ctx.set(dut.en, 1)
        for _ in range(N):
            got.append(ctx.get(dut.state))
            await ctx.tick()

    run_sim(dut, tb)
    assert got == gold, "LFSR sequence mismatch"
    print(f"test_lfsr        OK ({N} states bit-exact)")


# ------------------------------------------------------------
class _Comb(Elaboratable):
    def __init__(self, fn, w_in=24, w_out=32):
        self.x = Signal(signed(w_in))
        self.y = Signal(signed(w_out))
        self._fn = fn

    def elaborate(self, platform):
        m = Module()
        m.d.comb += self.y.eq(self._fn(self.x))
        return m


def test_rounding_sat():
    xs = [int(v) for v in rng.integers(-(1 << 20), 1 << 20, 500)]
    for s in (1, 4, 10):
        dut = _Comb(lambda v, s=s: rshift_round(v, s))
        got = []

        async def tb(ctx):
            for v in xs:
                ctx.set(dut.x, v)
                got.append(ctx.get(dut.y))

        run_sim(dut, tb, clock=False)
        gold = rshift(torch.tensor(xs, dtype=torch.int64), s).tolist()
        assert got == gold, f"rshift_round mismatch (s={s})"
    for bits in (8, 14, 18):
        dut = _Comb(lambda v, b=bits: saturate(v, b))
        got = []

        async def tb(ctx):
            for v in xs:
                ctx.set(dut.x, v)
                got.append(ctx.get(dut.y))

        run_sim(dut, tb, clock=False)
        gold = sat(torch.tensor(xs, dtype=torch.int64), bits).tolist()
        assert got == gold, f"saturate mismatch (bits={bits})"
    print("test_rounding_sat OK (rshift s=1,4,10 / sat 8,14,18 bit-exact)")


# ------------------------------------------------------------
def test_crossing():
    T, h = 64, 819                       # h = round(0.2 * 2^12)
    dut = CrossingUnit(w_d=17, T=T, h=h)
    for trial in range(20):
        dn = rng.integers(-(1 << 15), 1 << 15, T)
        # ノイズ幅を h 近傍にも寄せて交差が起きやすいケースを混ぜる
        if trial % 2:
            dn = rng.integers(-2 * h, 2 * h, T)
        gold_code, gold_cdiff = crossing_code(
            torch.tensor(dn, dtype=torch.int64).view(T, 1), h)
        gold_code = gold_code.view(-1).tolist()
        result = {}

        async def tb(ctx, dn=dn, result=result):
            ctx.set(dut.start, 1)
            await ctx.tick()
            ctx.set(dut.start, 0)
            codes = []
            for t in range(T):
                ctx.set(dut.in_valid, 1)
                ctx.set(dut.dn, int(dn[t]))
                if ctx.get(dut.code_valid):
                    codes.append(ctx.get(dut.code))
                await ctx.tick()
            ctx.set(dut.in_valid, 0)
            assert ctx.get(dut.code_valid) and ctx.get(dut.done)
            codes.append(ctx.get(dut.code))
            await ctx.tick()
            result["codes"] = codes
            result["cdiff"] = ctx.get(dut.cdiff)
            result["sum"] = ctx.get(dut.sum_code)
            result["sum2"] = ctx.get(dut.sum_code2)

        run_sim(dut, tb)
        assert result["codes"] == gold_code, f"code mismatch (trial {trial})"
        assert result["cdiff"] == int(gold_cdiff), f"cdiff mismatch ({trial})"
        assert result["sum"] == sum(gold_code)
        assert result["sum2"] == sum(c * c for c in gold_code)
    print("test_crossing    OK (20 windows x T=64, code/cdiff/Σ/Σ² bit-exact)")


# ------------------------------------------------------------
def test_divider():
    W_NUM, W_DEN = 30, 15
    dut = RoundDiv(W_NUM, W_DEN)
    cases = [(int(n), int(d)) for n, d in zip(
        rng.integers(-(1 << 28), 1 << 28, 300),
        rng.integers(1, 1 << 14, 300))]
    cases += [(0, 1), (-1, 1), (1, 1), ((1 << 28) - 1, 1), (-(1 << 28), 16383),
              (5, 3), (-5, 3), (7, 2), (-7, 2)]
    got = []

    async def tb(ctx):
        for n, d in cases:
            ctx.set(dut.num, n)
            ctx.set(dut.den, d)
            ctx.set(dut.start, 1)
            await ctx.tick()
            ctx.set(dut.start, 0)
            while not ctx.get(dut.done):
                await ctx.tick()
            got.append(ctx.get(dut.q))
            await ctx.tick()

    run_sim(dut, tb)
    gold = rdiv(torch.tensor([n for n, _ in cases], dtype=torch.int64),
                torch.tensor([d for _, d in cases], dtype=torch.int64)).tolist()
    bad = [(c, a, b) for c, a, b in zip(cases, got, gold) if a != b]
    assert not bad, f"divider mismatch: {bad[:5]}"
    print(f"test_divider     OK ({len(cases)} cases bit-exact, latency "
          f"{W_NUM + 3} cyc)")


# ------------------------------------------------------------
def test_divider_pipe():
    """RoundDivPipe: 連続 (1/cyc) と隙間あり発行の混在で rdiv と bit-exact."""
    W_NUM, W_DEN = 30, 15
    dut = RoundDivPipe(W_NUM, W_DEN)
    cases = [(int(n), int(d)) for n, d in zip(
        rng.integers(-(1 << 28), 1 << 28, 500),
        rng.integers(1, 1 << 14, 500))]
    cases += [(0, 1), (-1, 1), (1, 1), ((1 << 28) - 1, 1), (-(1 << 28), 16383),
              (5, 3), (-5, 3), (7, 2), (-7, 2), (-(1 << 28), 1)]
    gaps = [0] * len(cases)                         # 大半は毎サイクル発行
    for k in rng.integers(0, len(cases), 40):
        gaps[int(k)] = int(rng.integers(1, 5))
    got, pre_seen = [], []

    async def tb(ctx):
        n_issued, wait, prev_pre = 0, 0, 0
        while len(got) < len(cases):
            if wait > 0:                            # 発行間隔を空ける
                wait -= 1
                ctx.set(dut.in_valid, 0)
            elif n_issued < len(cases):
                n, d = cases[n_issued]
                ctx.set(dut.num, n)
                ctx.set(dut.den, d)
                ctx.set(dut.in_valid, 1)
                wait = gaps[n_issued]
                n_issued += 1
            else:
                ctx.set(dut.in_valid, 0)
            await ctx.tick()
            if ctx.get(dut.out_valid):
                got.append(ctx.get(dut.q))
                pre_seen.append(prev_pre)
            prev_pre = ctx.get(dut.pre_valid)

    run_sim(dut, tb)
    gold = rdiv(torch.tensor([n for n, _ in cases], dtype=torch.int64),
                torch.tensor([d for _, d in cases], dtype=torch.int64)).tolist()
    bad = [(c, a, b) for c, a, b in zip(cases, got, gold) if a != b]
    assert not bad, f"divider_pipe mismatch: {bad[:5]}"
    assert all(pre_seen), "pre_valid が out_valid の 1 サイクル前に立っていない"
    print(f"test_divider_pipe OK ({len(cases)} cases bit-exact, "
          f"latency {dut.LATENCY} cyc, 1/cyc + gap 発行)")


# ------------------------------------------------------------
def test_crossing_engine():
    """CrossingEngine: H ユニット時分割で crossing_code と bit-exact."""
    from fncl_rtl.crossing import CrossingEngine
    H, T, h = 4, 64, 819
    dut = CrossingEngine(H, w_d=17, T=T, h=h)
    dn = rng.integers(-2 * h, 2 * h, (H, T))
    gold = [crossing_code(torch.tensor(dn[i], dtype=torch.int64).view(T, 1),
                          h) for i in range(H)]
    got_codes = [[] for _ in range(H)]
    got_stats = {}

    async def tb(ctx):
        pend = [None]                               # E1 待ちの dn

        async def cyc(issue):
            if pend[0] is not None:
                ctx.set(dut.dn, pend[0])
            if issue is None:
                ctx.set(dut.in_valid, 0)
                pend[0] = None
            else:
                i, first, wrap = issue
                ctx.set(dut.idx, i)
                ctx.set(dut.in_valid, 1)
                ctx.set(dut.first_smp, first)
                ctx.set(dut.wrap, wrap)
                pend[0] = None if wrap else int(dn[i][kk])
            if ctx.get(dut.code_valid):             # 前サイクル発行分の E1
                got_codes[ctx.get(dut.idx_e1)].append(ctx.get(dut.code))
            await ctx.tick()

        for kk in range(T):                         # サンプルパス ×T
            for i in range(H):
                await cyc((i, kk == 0, False))
        for i in range(H):                          # 巡回パス
            await cyc((i, False, True))
        await cyc(None)                             # 最終要素の E1 回収
        await cyc(None)
        for i in range(H):                          # 統計読み出し
            ctx.set(dut.stat_addr, i)
            await ctx.tick()
            got_stats[i] = (ctx.get(dut.stat_cdiff), ctx.get(dut.stat_sum),
                            ctx.get(dut.stat_sum2))

    run_sim(dut, tb)
    for i in range(H):
        gold_code = gold[i][0].view(-1).tolist()
        assert got_codes[i] == gold_code, f"engine unit {i} code mismatch"
        assert got_stats[i] == (int(gold[i][1]), sum(gold_code),
                                sum(c * c for c in gold_code)), \
            f"engine unit {i} stats mismatch"
    print(f"test_crossing_engine OK (H={H} x T={T}, code/cdiff/Σ/Σ² "
          f"bit-exact)")


if __name__ == "__main__":
    test_lfsr()
    test_rounding_sat()
    test_crossing()
    test_divider()
    test_divider_pipe()
    test_crossing_engine()
    print("\nall unit tests passed (golden model bit-exact)")
