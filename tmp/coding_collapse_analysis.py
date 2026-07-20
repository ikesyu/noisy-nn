"""
coding_collapse_analysis.py — Gamma collapse の良さを定量化する

spike_coding_regime.py の結果 (results.json) を読み、「異なる信号速度の曲線が
Gamma の一軸にどれだけ乗るか」をスカラーで測る。目視の代わりに数値で
「解析窓の tau 正規化 (項目1) が collapse を直したか」を判定するために使う。

collapse quality の定義
-----------------------
各指標について、速度ごとの曲線を共通の log-Gamma グリッド上へ線形補間し、
グリッド点ごとの速度間のばらつき (std) を、その指標の全体レンジで割る:

    C = mean_over_grid( std_over_speeds( metric ) ) / range(metric)

C が小さいほど「一本の曲線に乗っている」= Gamma が制御パラメータとして
機能している。速度が 1 つしかない指標や、重なる Gamma 域が無い場合は nan。

seed 方向は先に平均する (seed 間のばらつきは別途 seed_spread として報告)。

使い方
------
    python tmp/coding_collapse_analysis.py out/coding_tau out/coding_abs
"""
import json
import sys
from pathlib import Path

import numpy as np

METRICS = [("cv", "ISI CV"),
           ("rel_fine", "reliability (fine)"),
           ("timing_frac", "timing info fraction"),
           ("j_half_norm", "jitter tolerance / tau"),
           ("signal_share", "signal-driven share"),
           ("mse_fir", "FIR decode MSE"),
           ("mse_rate", "rate decode MSE")]


def load_rows(d: Path):
    obj = json.loads((d / "results.json").read_text())
    return [r for r in obj["rows"] if not r.get("dead")]


def by_speed(rows, metric):
    """{speed: (gamma[], value[])} — seed 方向は平均してから返す。"""
    acc = {}
    for r in rows:
        v = r.get(metric)
        if v is None or not np.isfinite(v):
            continue
        acc.setdefault((r["speed"], r["sigma"]), []).append((r["gamma"], v))
    out = {}
    for (sp, _sig), vals in acc.items():
        g = float(np.mean([a for a, _ in vals]))
        m = float(np.mean([b for _, b in vals]))
        out.setdefault(sp, []).append((g, m))
    return {sp: (np.array([g for g, _ in sorted(v)]),
                 np.array([m for _, m in sorted(v)]))
            for sp, v in out.items()}


def collapse_quality(rows, metric, n_grid: int = 12):
    curves = by_speed(rows, metric)
    if len(curves) < 2:
        return float("nan"), float("nan")
    # 全速度が値を持つ Gamma の共通区間
    lo = max(c[0].min() for c in curves.values())
    hi = min(c[0].max() for c in curves.values())
    if not (hi > lo):
        return float("nan"), float("nan")
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), n_grid))
    stacked = []
    for g, m in curves.values():
        o = np.argsort(g)
        stacked.append(np.interp(grid, g[o], m[o]))
    stacked = np.array(stacked)
    rng = float(stacked.max() - stacked.min())
    if rng < 1e-9:
        return float("nan"), rng
    return float(np.mean(np.std(stacked, axis=0)) / rng), rng


def seed_spread(rows, metric):
    """同一 (speed, sigma) における seed 間 std の平均（再現性の目安）。"""
    acc = {}
    for r in rows:
        v = r.get(metric)
        if v is not None and np.isfinite(v):
            acc.setdefault((r["speed"], r["sigma"]), []).append(v)
    sds = [np.std(v) for v in acc.values() if len(v) > 1]
    return float(np.mean(sds)) if sds else float("nan")


def main():
    dirs = [Path(d) for d in sys.argv[1:]] or [Path("out/coding_tau")]
    tables = {}
    for d in dirs:
        rows = load_rows(d)
        tables[d.name] = {m: (collapse_quality(rows, m)[0], seed_spread(rows, m))
                          for m, _ in METRICS}
        speeds = sorted({r["speed"] for r in rows})
        seeds = sorted({r["seed"] for r in rows})
        print(f"{d.name}: {len(rows)} rows, speeds={speeds}, seeds={seeds}")

    hdr = "| metric | " + " | ".join(f"{n} C | {n} seed-sd" for n in tables) + " |"
    print("\n" + hdr)
    print("|---" * (1 + 2 * len(tables)) + "|")
    for key, label in METRICS:
        cells = []
        for n in tables:
            c, s = tables[n][key]
            cells += [f"{c:.3f}" if np.isfinite(c) else "—",
                      f"{s:.3f}" if np.isfinite(s) else "—"]
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("\nC = 速度間ばらつき / レンジ（小さいほど Gamma 一軸に collapse）")


if __name__ == "__main__":
    main()
