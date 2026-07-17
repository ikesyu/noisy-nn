"""FnclCore (学習コア) とゴールデンモデル fxp_step の 1 提示 bit-exact 照合。

実行: .venv/bin/python rtl/tests/test_core.py   (リポジトリルートから)

H=T=8 のミニ構成・語長は凍結仕様 (Fw14/Fd10/Nn8) のまま。3 提示を連続実行し、
毎提示後に y と全状態 (重み 6 群・速度・M1s/Mos・vM 複製) を突き合わせる。
ノイズは phase24: 各レーンの LFSR をゴールデンモデルと同じ m 系列オフセット
(seq[off]) でシードすれば、以後は 1 サンプル 1 ステップで完全に一致する。
"""
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rtl"))
sys.path.insert(0, str(ROOT / "tmp"))

import torch
from amaranth.sim import Simulator

from fncl_rtl.core import FnclCore
from fncl_phase1_fxp import NoiseSource, fxp_prepare, fxp_step

H = T = 8
FW, FD, NN = 14, 10, 8


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
    state, cfg = fxp_prepare(p, args, cfg0)
    return state, cfg


import numpy as _np

_rng = _np.random.default_rng(7)
PRESENTATIONS = [
    (int(round(0.7 * (1 << FD))), int(round(0.3 * (1 << FD))), 256),
    (int(round(-1.5 * (1 << FD))), int(round(-0.8 * (1 << FD))), 231),
    (300, -500, 128),
] + [(int(_rng.integers(-2 << FD, 2 << FD)),
      int(_rng.integers(-1 << FD, 1 << FD)),
      int(_rng.integers(0, 257))) for _ in range(17)]


def main():
    state, cfg = build_golden()
    pq0 = {k: v.clone() for k, v in state["pq"].items()}
    src = NoiseSource("phase24", T, H, 2, seed=3)
    src2 = NoiseSource("phase24", T, H, 2, seed=3)   # 診断用の複製
    seeds0 = [int(src.seq[int(src.off[0][i])]) for i in range(H)]
    seeds1 = [int(src.seq[int(src.off[1][j])]) for j in range(H)]

    # --- golden: 3 提示を回し、毎提示後の状態を控える ---
    snaps, traces = [], []
    for xq, tq, rom in PRESENTATIONS:
        trace = fxp_step(state, torch.tensor([[xq]], dtype=torch.int64),
                         tq, rom, src, cfg)
        traces.append(trace)
        snaps.append({"y": trace["y"],
                      "pq": deepcopy(state["pq"]),
                      "vel": deepcopy(state["vel"]),
                      "M1s": state["M1s"].clone(),
                      "Mos": state["Mos"].clone()})

    # --- RTL ---
    params = {"W0": pq0["W0"].view(-1).tolist(), "b0": pq0["b0"].tolist(),
              "W1": [row.tolist() for row in pq0["W1"]],
              "b1": pq0["b1"].tolist(),
              "Wout": pq0["Wout"].view(-1).tolist(),
              "bout": int(pq0["bout"][0])}
    core = FnclCore(H, T, params, Fw=FW, Fd=FD, Nn=NN, hq=cfg["hq"])
    sim = Simulator(core)
    sim.add_clock(1e-6)

    def diagnose(ctx, k, trace):
        dbg = core.dbg
        items = [
            ("d0", dbg["d0"], trace["d0"]),
            ("Sz0", [c.sum_code for c in core.cross0], trace["code0"].sum(0)),
            ("Sz1", [c.sum_code for c in core.cross1], trace["code1"].sum(0)),
            ("sd1", dbg["sd1"], trace["d1"].sum(0)),
            ("slope0", dbg["slope0"], trace["slope0"]),
            ("slope1", dbg["slope1"], trace["slope1"]),
            ("syz", dbg["syz"], (trace["ys"].unsqueeze(-1)
                                 * trace["code1"]).sum(0)),
            ("ysum", [dbg["ysum"]], trace["ys"].sum().unsqueeze(0)),
            ("dd1", dbg["dd1"], trace["dd1"]),
            ("a0", dbg["a0"], trace["a0"]),
        ]
        for name, sigs, gold in items:
            got = [ctx.get(s) for s in sigs]
            want = [int(v) for v in gold]
            if got != want:
                print(f"  p{k} MISMATCH {name}:\n    rtl : {got}\n"
                      f"    gold: {want}")
                return
        print(f"  p{k} diagnose: 中間量は一致")

    def check(ctx, k, snap):
        def grid(sigs2d, tensor, transpose=False):
            for a in range(len(sigs2d)):
                for b in range(len(sigs2d[a])):
                    gold = int(tensor[b][a] if transpose else tensor[a][b])
                    got = ctx.get(sigs2d[a][b])
                    assert got == gold, \
                        f"p{k} {tensor_name}: [{a}][{b}] {got} != {gold}"

        def vec(sigs, tensor):
            for a, s in enumerate(sigs):
                assert ctx.get(s) == int(tensor[a]), \
                    f"p{k} {tensor_name}: [{a}] {ctx.get(s)} != {int(tensor[a])}"

        assert ctx.get(core.y) == snap["y"], \
            f"p{k} y: {ctx.get(core.y)} != {snap['y']}"
        pq, vel = snap["pq"], snap["vel"]
        for tensor_name, sigs, gold in [
                ("W0", core.w0, pq["W0"].view(-1)), ("b0", core.b0, pq["b0"]),
                ("b1", core.b1, pq["b1"]),
                ("Wout", core.wout, pq["Wout"].view(-1)),
                ("v_b1", core.v_b1, vel["b1"]),
                ("v_wout", core.v_wout, vel["Wout"].view(-1)),
                ("v_w0", core.v_w0, vel["W0"].view(-1)),
                ("v_b0", core.v_b0, vel["b0"]),
                ("Mos", core.mos, snap["Mos"].view(-1))]:
            vec(sigs, gold)
        for tensor_name, sigs, gold, tr in [
                ("W1", core.w1, pq["W1"], False),
                ("v_W1", core.v_w1, vel["W1"], False),
                ("M1s", core.m1s, snap["M1s"], True),
                ("vmr", core.vmr, vel["W1"], True)]:
            grid(sigs, gold, transpose=tr)
        tensor_name = "bout"
        assert ctx.get(core.bout) == int(pq["bout"][0])
        assert ctx.get(core.v_bout) == int(vel["bout"][0])

    async def tb(ctx):
        for i in range(H):
            ctx.set(core.lfsr0[i].seed, seeds0[i])
            ctx.set(core.lfsr0[i].load, 1)
            ctx.set(core.lfsr1[i].seed, seeds1[i])
            ctx.set(core.lfsr1[i].load, 1)
        await ctx.tick()
        for l in core.lfsr0 + core.lfsr1:
            ctx.set(l.load, 0)
        for k, (xq, tq, rom) in enumerate(PRESENTATIONS):
            ctx.set(core.xq, xq)
            ctx.set(core.tq, tq)
            ctx.set(core.romq, rom)
            ctx.set(core.go, 1)
            await ctx.tick()
            ctx.set(core.go, 0)
            cycles = 0
            d1_seq, lf1_seq, c0_seq, c0l_seq = [], [], [], []
            while not ctx.get(core.done):
                if ctx.get(core.dbg["in1"]):
                    d1_seq.append([ctx.get(s) for s in core.dbg["d1"]])
                    lf1_seq.append([ctx.get(l.state) for l in core.lfsr1])
                    c0l_seq.append([ctx.get(s) for s in core.dbg["code0_l"]])
                if ctx.get(core.cross0[0].code_valid):
                    c0_seq.append([ctx.get(c.code) for c in core.cross0])
                await ctx.tick()
                cycles += 1
                assert cycles < 20000, f"p{k}: done が来ない"
            if k == 0:
                gold_c0 = [[int(v) for v in row] for row in traces[0]["code0"]]
                for t, (got, want) in enumerate(zip(c0_seq, gold_c0)):
                    if got != want:
                        print(f"  p0 code0[t={t}]\n    rtl : {got}\n"
                              f"    gold: {want}")
                        break
                else:
                    print(f"  p0 code0 stream: 一致 ({len(c0_seq)} emissions)")
                for t, (got, want) in enumerate(zip(c0l_seq, gold_c0)):
                    if got != want:
                        print(f"  p0 code0_latch[t={t}]\n    rtl : {got}\n"
                              f"    gold: {want}")
                        break
                else:
                    print("  p0 code0 latch stream: 一致")
                gold_d1 = [[int(v) for v in row] for row in traces[0]["d1"]]
                for t, (got, want) in enumerate(zip(d1_seq, gold_d1)):
                    if got != want:
                        print(f"  p0 d1[t={t}]\n    rtl : {got}\n"
                              f"    gold: {want}")
                        break
                else:
                    print("  p0 d1 stream: 一致")
                gold_lf = [[int(src2.seq[(int(src2.off[1][j]) + t)
                                         % src2.P]) for j in range(H)]
                           for t in range(T)]
                for t, (got, want) in enumerate(zip(lf1_seq, gold_lf)):
                    if got != want:
                        print(f"  p0 lfsr1[t={t}]\n    rtl : {got}\n"
                              f"    gold: {want}")
                        break
                else:
                    print("  p0 lfsr1 stream: 一致")
            diagnose(ctx, k, traces[k])
            check(ctx, k, snaps[k])
            print(f"presentation {k}: bit-exact OK ({cycles} cyc, "
                  f"y={snaps[k]['y']})")
            await ctx.tick()

    sim.add_testbench(tb)
    sim.run()
    print(f"\ntest_core OK: {len(PRESENTATIONS)} 提示 x 全状態 "
          f"(重み/速度/M1s/Mos/vM 複製) が fxp_step と bit-exact 一致")


if __name__ == "__main__":
    main()
