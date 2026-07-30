"""
fncl_results_fig.py — MNIST 実験（docs/idea_ca.md §2, §3, §8, §10）の図を results.json から生成
（旧名: fncl_nf_fig.py．出力ディレクトリ tmp/out/ は旧名のまま）

生成物 (tmp/out/fncl_figs/):
  fig_curves_depth4.png   : depth 4 の test acc 学習曲線（backprop / cov_jac / diag / per16, 3 seed）
  fig_mirror_depth4.png   : depth 4 の深層ミラー r の推移（diag vs per16, 3 seed）
  fig_depth_bar.png       : 層深依存性（depth 2/3/4, diag vs per16 vs backprop, 3 seed）
  fig_fidelity_bar.png    : 勾配忠実度（cov_jac vs FA vs DFA, 層別 cosine, depth 4）
  fig_screening_bar.png   : S1 選別（ミラー推定量ごとの min r, ガウス/一様）

実行例: .venv/bin/python tmp/fncl_results_fig.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("tmp/out/fncl_figs")
OUT.mkdir(parents=True, exist_ok=True)
R = lambda p: json.load(open(p))["runs"]  # noqa: E731

C = {"backprop": "k", "cov_jac": "tab:red", "diag": "tab:orange",
     "per16": "tab:blue", "fa": "tab:green", "dfa": "tab:purple"}


def hist(path, arm):
    return R(path)[arm] if arm in R(path) else R(path)["4"][arm]


# ---- 系列の収集（depth 4, 3 seed）----
SRC = {
    "backprop": [("tmp/out/fncl_a1d/results.json", "backprop"),
                 ("tmp/out/fncl_a1d_s1/results.json", "backprop"),
                 ("tmp/out/fncl_a1d_s2/results.json", "backprop")],
    "cov_jac":  [("tmp/out/fncl_a1d/results.json", "cov_jac"),
                 ("tmp/out/fncl_a1d_s1/results.json", "cov_jac"),
                 ("tmp/out/fncl_a1d_s2/results.json", "cov_jac")],
    "diag":     [("tmp/out/fncl_a1e/results.json", "diag"),
                 ("tmp/out/fncl_nf3_s1/results.json", "diag"),
                 ("tmp/out/fncl_nf3_s2/results.json", "diag")],
    "per16":    [("tmp/out/fncl_nf2/results.json", "per16"),
                 ("tmp/out/fncl_nf3_s1/results.json", "per16"),
                 ("tmp/out/fncl_nf3_s2/results.json", "per16")],
}
LABEL_EN = {"backprop": "backprop", "cov_jac": "cov_jac (all mirrors/epoch)",
            "diag": "cov_jac (rotation)", "per16": "cov_jac + periodic noise field (per16)"}

# ---- 図 1: 学習曲線 ----
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, seeds in SRC.items():
    eps = [h["epoch"] for h in hist(*seeds[0])]
    ys = np.array([[h["test_acc"] for h in hist(*s)] for s in seeds])
    m, lo, hi = ys.mean(0), ys.min(0), ys.max(0)
    ax.plot(eps, m, color=C[name], label=LABEL_EN[name], lw=2)
    ax.fill_between(eps, lo, hi, color=C[name], alpha=0.15)
ax.set_xlabel("epoch")
ax.set_ylabel("test accuracy")
ax.set_ylim(0.55, 0.92)
ax.set_title("MNIST 784-256×4-10, train1000/test1000 (mean & range, 3 seeds)")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig_curves_depth4.png", dpi=150)
plt.close(fig)

# ---- 図 2: 深層ミラー r の推移 ----
fig, ax = plt.subplots(figsize=(7, 4.5))
for name in ("diag", "per16"):
    for si, s in enumerate(SRC[name]):
        h = hist(*s)
        eps = [e["epoch"] for e in h]
        rmin = [min(v for k, v in e["mirror_r"].items() if k != "z1") for e in h]
        ax.plot(eps, rmin, color=C[name], alpha=0.8, lw=1.5,
                label=LABEL_EN[name] if si == 0 else None)
ax.set_xlabel("epoch")
ax.set_ylabel("min mirror $r$ (deep mirrors z2–z4)")
ax.set_title("weight-mirror recovery (depth 4, 3 seeds)")
ax.legend(fontsize=9, loc="lower left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig_mirror_depth4.png", dpi=150)
plt.close(fig)

# ---- 図 3: 層深依存性 ----
depth_src = {
    2: [(f"tmp/out/fncl_nf_d2_s{s}/results.json") for s in (0, 1, 2)],
    3: [(f"tmp/out/fncl_nf_d3_s{s}/results.json") for s in (0, 1, 2)],
}
fig, ax = plt.subplots(figsize=(6.5, 4.2))
depths = [2, 3, 4]
width = 0.35
for i, name in enumerate(("diag", "per16")):
    means, stds = [], []
    for d in depths:
        if d == 4:
            vals = [hist(*s)[-1]["test_acc"] for s in SRC[name]]
        else:
            vals = [R(p)[name][-1]["test_acc"] for p in depth_src[d]]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    ax.bar(np.arange(3) + (i - 0.5) * width, means, width, yerr=stds, capsize=4,
           color=C[name], label=LABEL_EN[name])
bp = [hist(*s)[-1]["test_acc"] for s in SRC["backprop"]]
ax.axhline(np.mean(bp), color="k", ls="--", lw=1,
           label=f"backprop depth4 ({np.mean(bp):.3f})")
ax.set_xticks(range(3))
ax.set_xticklabels([f"depth {d}" for d in depths])
ax.set_ylabel("final test accuracy (3 seeds)")
ax.set_ylim(0.80, 0.91)
ax.set_title("depth dependence: periodic noise field matters at depth 4")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "fig_depth_bar.png", dpi=150)
plt.close(fig)

# ---- 図 4: 勾配忠実度 (cov_jac vs FA vs DFA) ----
a2b = json.load(open("tmp/out/fncl_a2b/results.json"))["runs"]["4"]
layers = ["w0", "w1", "w2", "w3"]
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for ax, state in zip(axes, ("untrained", "pretrained")):
    for i, tag in enumerate(("cov_jac", "fa", "dfa")):
        vals = [a2b[state][tag][k]["cos"] for k in layers]
        ax.bar(np.arange(4) + (i - 1) * 0.27, vals, 0.27, color=C[tag], label=tag)
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(layers)
    ax.set_title(state)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("cosine to true gradient")
axes[0].legend(fontsize=9)
fig.suptitle("gradient fidelity (MNIST 784-256x4-10, CE): FA/DFA nearly orthogonal to true gradient", y=1.0)
fig.tight_layout()
fig.savefig(OUT / "fig_fidelity_bar.png", dpi=150)
plt.close(fig)

# ---- 図 5: S1 選別（ミラー推定量ごとの min r）----
fig, ax = plt.subplots(figsize=(8, 4.2))
order = ["diag", "cm", "rank1", "unif", "r1lo", "wood", "blk*", "mv",
         "per4", "per8", "per16", "per32"]
gauss = json.load(open("tmp/out/fncl_nf1c/results.json"))["arms"]
uni = json.load(open("tmp/out/fncl_nf1_uni/results.json"))["arms"]
blk = json.load(open("tmp/out/fncl_a1f/results.json"))["runs"]
blk_min = min(min(v for k, v in h[-1]["mirror_r"].items()) for h in blk.values())
for i, (arms, lab, col) in enumerate([(gauss, "gaussian", "tab:blue"),
                                      (uni, "uniform", "tab:cyan")]):
    vals = []
    for a in order:
        if a == "blk*":
            vals.append(blk_min if lab == "gaussian" else np.nan)
        else:
            mr = arms[a]["mirror_r"]
            vals.append(min(v for k, v in mr.items() if k != "z1"))
    ax.bar(np.arange(len(order)) + (i - 0.5) * 0.38, vals, 0.38,
           color=col, label=lab)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=30, ha="right", fontsize=9)
ax.set_ylim(0.84, 1.005)
ax.set_ylabel("min mirror $r$ (deep mirrors)")
ax.set_title("mirror interventions: algebraic fixes fall short; only noise design (per*) and full mv recover")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "fig_screening_bar.png", dpi=150)
plt.close(fig)

for f in sorted(OUT.glob("*.png")):
    print("saved", f)
