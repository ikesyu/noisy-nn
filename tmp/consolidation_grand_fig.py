"""consolidation_grand_fig.py — 統一容量比較 (§12.9.13) の統合図.

consolidation_grand.py の 2 実行分 (hard,soft2 / soft3,softA) と
consolidation_caseB.py --freqs 1,2,3,4 の results.json を読み、
確定版の比較図 out/consolidation_grand/fig_grand.png を生成する。
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fncl_driver import savefig  # noqa: E402
from pathlib import Path  # noqa: E402

OUT = Path("out/consolidation_grand")
K = 16

r1 = json.load(open(OUT / "results_hard_soft2.json"))["results"]["0"]
r2 = json.load(open(OUT / "results_soft3_softA.json"))["results"]["0"]
rb = json.load(open("out/consolidation_caseB16/results.json"))["results"]["0"]
arms = {**r1, **r2}
order = ["hard", "soft2", "soft3", "softA"]
labels = {"hard": "hard\n(12.8)", "soft2": "soft2\n(case 2)",
          "soft3": "soft3\n(case 3)", "softA": "softA\n(case A)",
          "caseB": "case B\n(shared field)"}

fig = plt.figure(figsize=(13, 4.8))
gs = fig.add_gridspec(1, 3)

# --- capacity bars: seq vs +pipeline, 案B 併記 ---
ax = fig.add_subplot(gs[0, 0])
idx = np.arange(len(order) + 1)
seq = [arms[a]["capacity_seq"] for a in order] + [rb["capacity"]]
pipe = [arms[a]["capacity_pipe"] for a in order] + [rb["capacity"]]
ax.bar(idx, pipe, 0.6, color="tab:blue", alpha=0.45, label="+pipeline")
ax.bar(idx, seq, 0.6, color="tab:blue", label="sequential")
ax.set_xticks(idx)
ax.set_xticklabels([labels[a] for a in order] + [labels["caseB"]], fontsize=8)
ax.set_ylabel(f"functions stored (/{K})")
ax.set_title("Capacity (unified: 16 tasks, multiscale, seed 0)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

# --- efficiency frontier: units vs capacity ---
ax = fig.add_subplot(gs[0, 1])
for a, c in zip(order, ("tab:gray", "tab:green", "tab:orange", "tab:red")):
    ax.scatter(arms[a]["union_after"], arms[a]["capacity_pipe"], s=70, c=c,
               label=labels[a].replace("\n", " "))
ax.scatter(rb["union"], rb["capacity"], s=70, c="tab:purple",
           label=labels["caseB"].replace("\n", " "))
ax.set_xlabel("units occupied after pipeline")
ax.set_ylabel(f"functions stored (/{K})")
ax.set_title("Resource vs capacity", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# --- per-task storage map ---
ax = fig.add_subplot(gs[0, 2])
names = None
M = np.zeros((len(order) + 1, K))
for i, a in enumerate(order):
    stored_seq = arms[a]["stored"]
    # pipeline 後の格納: 圧縮維持分 + retry 分を近似的にタスクへ割り当てる
    kept = {j for j, ok in enumerate(stored_seq) if ok}
    retried = {e["name"]: e["mse"] for e in arms[a]["retry"]}
    for j in range(K):
        M[i, j] = 1.0 if stored_seq[j] else 0.0
    for e in arms[a]["retry"]:
        pass
    M[i] = [1.0 if stored_seq[j] else 0.0 for j in range(K)]
M[len(order)] = [1.0 if ok else 0.0 for ok in rb["stored"]]
ax.imshow(M, aspect="auto", cmap="Greens", vmin=0, vmax=1.3,
          interpolation="nearest")
ax.set_yticks(range(len(order) + 1))
ax.set_yticklabels([labels[a].replace("\n", " ") for a in order]
                   + [labels["caseB"].replace("\n", " ")], fontsize=7)
ax.set_xticks(range(K))
ax.set_xticklabels([f"k{1 + i // 4}p{i % 4}" for i in range(K)], fontsize=6)
ax.set_title("Stored tasks (sequential phase; green = stored)", fontsize=10)
fig.suptitle("Grand unified capacity comparison (H=32x2, 16 tasks "
             "sin(kx+phi), thr=0.05)", fontsize=12)
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
savefig(fig, OUT / "fig_grand.png")
