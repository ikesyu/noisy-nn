"""FnclCoreL を Verilog に書き出す（合成試行用）.

実行: .venv/bin/python rtl/export_core_l.py --H 8 --L 4 [--T 64] [--out rtl/out/core_l_h8_l4.v]
重み初期値は資源量に影響しないため 0 で埋める（学習実験には test_core_l.py の経路を使う）.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from amaranth.back import verilog

from fncl_rtl.core_l import FnclCoreL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    out = a.out or f"rtl/out/core_l_h{a.H}_l{a.L}.v"
    H = a.H
    params = {"W0": [0] * H, "b0": [0] * H, "W1": [[0] * H for _ in range(H)],
              "b1": [0] * H, "Wout": [0] * H, "bout": 0}
    core = FnclCoreL(H, a.T, a.L, params)
    ports = [core.go, core.done, core.xq, core.tq, core.romq, core.y]
    Path(out).write_text(verilog.convert(core, name="top", ports=ports))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
