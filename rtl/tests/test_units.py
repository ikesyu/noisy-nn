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


if __name__ == "__main__":
    test_lfsr()
    test_rounding_sat()
    test_crossing()
    test_divider()
    print("\nall unit tests passed (golden model bit-exact)")
