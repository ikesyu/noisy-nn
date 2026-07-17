"""FnclCoreL (L パラメトリック + Memory 版) の bit-exact 検証。

実行: .venv/bin/python rtl/tests/test_core_l.py [--long]

H=T=8 で L ∈ {8, 4, 2} の各構成を 6 提示ずつゴールデンモデル fxp_step と
突き合わせる (y + 全レジスタ + 全 Memory)。ゴールデンモデルの軌跡は L に
依存しないので 1 回だけ回して全 L で使う。--long は L=4 で 200 提示の
長期学習を回し、50 提示ごとに全状態一致を確認する。
"""
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rtl"))
sys.path.insert(0, str(ROOT / "tmp"))

import numpy as np
import torch
from amaranth.sim import Simulator

from fncl_rtl.core_l import FnclCoreL
from fncl_phase1_fxp import NoiseSource, fxp_prepare, fxp_step

H = T = 8
FW, FD, NN = 14, 10, 8
LONG = "--long" in sys.argv
N_SHORT, N_LONG, CHECK_EVERY = 6, 200, 50


def build_golden():
    torch.manual_seed(42)
    p = {"W0": torch.randn(H, 1, dtype=torch.float64) * 0.8,
         "b0": torch.randn(H, dtype=torch.float64) * 0.5,
         "W1": torch.randn(H, H, dtype=torch.float64) * 0.4,
         "b1": torch.randn(H, dtype=torch.float64) * 0.3,
         "Wout": torch.randn(1, H, dtype=torch.float64) * 0.5,
         "bout": torch.randn(1, dtype=torch.float64) * 0.2}
    args = SimpleNamespace(num_samples=T, crossing_h=0.2)
    cfg0 = {"Fw": FW, "Fd": FD, "Nn": NN, "Ge": 4, "Gg": 4,
            "k_ema": 10, "lr_shift": 10, "sat": True}
    return fxp_prepare(p, args, cfg0)


def golden_run(n_pres):
    state, cfg = build_golden()
    pq0 = {k: v.clone() for k, v in state["pq"].items()}
    src = NoiseSource("phase24", T, H, 2, seed=3)
    rng = np.random.default_rng(7)
    pres, snaps = [], []
    for _ in range(n_pres):
        xq = int(rng.integers(-2 << FD, 2 << FD))
        tq = int(rng.integers(-1 << FD, 1 << FD))
        rom = int(rng.integers(0, 257))
        pres.append((xq, tq, rom))
        trace = fxp_step(state, torch.tensor([[xq]], dtype=torch.int64),
                         tq, rom, src, cfg)
        snaps.append({"y": trace["y"],
                      "pq": deepcopy(state["pq"]),
                      "vel": deepcopy(state["vel"]),
                      "M1s": state["M1s"].clone(),
                      "Mos": state["Mos"].clone()})
    seeds0 = [int(src.seq[int(src.off[0][i])]) for i in range(H)]
    seeds1 = [int(src.seq[int(src.off[1][j])]) for j in range(H)]
    return pq0, cfg, pres, snaps, seeds0, seeds1


async def dump_mem(ctx, rps, R):
    """レーンローカル Memory を [H 行][H 列] 形に読み戻す (ROW 配列向け)."""
    out = [[None] * H for _ in range(H)]
    for l, rp in enumerate(rps):
        for a in range(R * H):
            ctx.set(rp.addr, a)
            await ctx.tick()
            out[l * R + a // H][a % H] = ctx.get(rp.data)
    return out


def run_rtl(L, n_pres, pq0, cfg, pres, snaps, seeds0, seeds1):
    R = H // L
    params = {"W0": pq0["W0"].view(-1).tolist(), "b0": pq0["b0"].tolist(),
              "W1": [row.tolist() for row in pq0["W1"]],
              "b1": pq0["b1"].tolist(),
              "Wout": pq0["Wout"].view(-1).tolist(),
              "bout": int(pq0["bout"][0])}
    core = FnclCoreL(H, T, L, params, Fw=FW, Fd=FD, Nn=NN, hq=cfg["hq"])
    sim = Simulator(core)
    sim.add_clock(1e-6)
    cyc_seen = []

    async def check(ctx, k, snap):
        pq, vel = snap["pq"], snap["vel"]
        for name, sigs, gold in [
                ("W0", core.w0, pq["W0"].view(-1)), ("b0", core.b0, pq["b0"]),
                ("b1", core.b1, pq["b1"]),
                ("Wout", core.wout, pq["Wout"].view(-1)),
                ("v_wout", core.v_wout, vel["Wout"].view(-1)),
                ("v_w0", core.v_w0, vel["W0"].view(-1)),
                ("v_b0", core.v_b0, vel["b0"]),
                ("v_b1", core.v_b1, vel["b1"]),
                ("Mos", core.mos, snap["Mos"].view(-1))]:
            got = [ctx.get(s) for s in sigs]
            want = [int(v) for v in gold]
            assert got == want, f"L={L} p{k} {name}: {got} != {want}"
        assert ctx.get(core.bout) == int(pq["bout"][0])
        assert ctx.get(core.v_bout) == int(vel["bout"][0])
        assert ctx.get(core.y) == snap["y"], f"L={L} p{k} y"
        # Memory: ROW 側は [j][i] 直、COL 側 (m1s/vmr) は転置比較
        w1 = await dump_mem(ctx, core.dbg_rp["w1"], R)
        vw1 = await dump_mem(ctx, core.dbg_rp["vw1"], R)
        m1s = await dump_mem(ctx, core.dbg_rp["m1s"], R)
        vmr = await dump_mem(ctx, core.dbg_rp["vmr"], R)
        for j in range(H):
            for i in range(H):
                assert w1[j][i] == int(pq["W1"][j][i]), f"L={L} p{k} W1"
                assert vw1[j][i] == int(vel["W1"][j][i]), f"L={L} p{k} vW1"
                assert m1s[i][j] == int(snap["M1s"][j][i]), f"L={L} p{k} M1s"
                assert vmr[i][j] == int(vel["W1"][j][i]), f"L={L} p{k} vmr"

    async def tb(ctx):
        for i in range(H):
            ctx.set(core.lfsr0[i].seed, seeds0[i])
            ctx.set(core.lfsr0[i].load, 1)
            ctx.set(core.lfsr1[i].seed, seeds1[i])
            ctx.set(core.lfsr1[i].load, 1)
        await ctx.tick()
        for l in core.lfsr0 + core.lfsr1:
            ctx.set(l.load, 0)
        for k in range(n_pres):
            xq, tq, rom = pres[k]
            ctx.set(core.xq, xq)
            ctx.set(core.tq, tq)
            ctx.set(core.romq, rom)
            ctx.set(core.go, 1)
            await ctx.tick()
            ctx.set(core.go, 0)
            cycles = 0
            while not ctx.get(core.done):
                await ctx.tick()
                cycles += 1
                assert cycles < 100000, f"L={L} p{k}: done が来ない"
            cyc_seen.append(cycles)
            if k == n_pres - 1 or (k + 1) % CHECK_EVERY == 0 or k < 2:
                await check(ctx, k, snaps[k])
            await ctx.tick()

    sim.add_testbench(tb)
    sim.run()
    return cyc_seen[0]


def main():
    n = N_LONG if LONG else N_SHORT
    gold = golden_run(n)
    if LONG:
        cyc = run_rtl(4, n, *gold)
        print(f"L=4 長期: {n} 提示 (チェック {CHECK_EVERY} 提示ごと + 先頭 2) "
              f"全状態 bit-exact / {cyc} cyc/提示")
    else:
        for L in (8, 4, 2):
            cyc = run_rtl(L, n, *gold)
            print(f"L={L}: {n} 提示 bit-exact OK ({cyc} cyc/提示, "
                  f"R={H // L})")
        print("\ntest_core_l OK: L=8/4/2 で fxp_step と全状態 bit-exact 一致")


if __name__ == "__main__":
    main()
