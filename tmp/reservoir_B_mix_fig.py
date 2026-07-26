"""(B) の詰め — 図: σ 混合が (A) との差を消し、単一ラグ場を救済する (docs §10.19).

 (a) NARMA-x: ESN / (B)diag / (B)mix / (A). (B)mix が (A) にほぼ到達
     -> (A)-(B) の差の本体は「対角 vs 混合」の接続性、残差=単調vsバンプ.
 (b) 単一ラグ場テスト (NARMA-20): delay/cascade/LDN で (B)diag vs (B)mix.
     混合はノイズマップ側でラグを結合できるので、純メモリ(遅延線)場でも成立
     -> 「記憶は場・非線形は NNN」の分離が最も clean に実現する.

Run:  .venv/bin/python tmp/reservoir_B_mix_fig.py [--quick]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import reservoir as R

C_ESN = "#2563eb"; C_DIAG = "#94a3b8"; C_MIX = "#0d9488"; C_A = "#ea7317"
OUT = os.path.join(os.path.dirname(__file__), "out", "reservoir_poc")


def masks(T, washout=300, frac=0.7):
    idx = np.arange(washout, T); ntr = int(len(idx) * frac)
    tr = np.zeros(T, bool); tr[idx[:ntr]] = True
    te = np.zeros(T, bool); te[idx[ntr:]] = True
    return tr, te


def run(quick=False):
    os.makedirs(OUT, exist_ok=True)
    T = 4000 if quick else 6000
    epochs = 400 if quick else 500
    H = 48
    n_list = [10, 20, 30, 40]
    seeds = [0, 1, 2]
    tr, te = masks(T)

    esn_x, bd, bm, ba = [], [], [], []
    for n in n_list:
        e_s, d_s, m_s, a_s = [], [], [], []
        for sd in seeds:
            u, y = R.narma_x(T, n, seed=1 + sd)
            A = R.LDNField(H=H, theta=60.0).run(u)
            e_s.append(R.task_nrmse(R.LeakyESN(H=H, seed=sd).run(u), y, washout=300))
            d_s.append(R.NoiseModulatedMap(A, y, tr, Ho=H, mix=False, seed=100 + sd).eval(te, epochs))
            m_s.append(R.NoiseModulatedMap(A, y, tr, Ho=H, mix=True, seed=100 + sd).eval(te, epochs))
            a_s.append(R.LearnedCrossingMap(A, y, tr, Ho=H, seed=100 + sd).eval(te, epochs))
        esn_x.append(np.mean(e_s)); bd.append(np.mean(d_s))
        bm.append(np.mean(m_s)); ba.append(np.mean(a_s))

    fields = {"delay": lambda: R.DelayLineField(H=H),
              "cascade": lambda: R.CascadeField(H=H, a=0.92),
              "LDN": lambda: R.LDNField(H=H, theta=60.0)}
    fd, fm = {}, {}
    for fn, mk in fields.items():
        d_s, m_s = [], []
        for sd in seeds:
            u, y = R.narma_x(T, 20, seed=1 + sd)
            Af = mk().run(u)
            d_s.append(R.NoiseModulatedMap(Af, y, tr, Ho=H, mix=False, seed=100 + sd).eval(te, epochs))
            m_s.append(R.NoiseModulatedMap(Af, y, tr, Ho=H, mix=True, seed=100 + sd).eval(te, epochs))
        fd[fn] = np.mean(d_s); fm[fn] = np.mean(m_s)

    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    ax[0].plot(n_list, esn_x, "s-", color=C_ESN, lw=2.2, label="ESN (reservoir does both)")
    ax[0].plot(n_list, bd, "o--", color=C_DIAG, lw=1.8, label="(B) diagonal: 1 field coord -> noise")
    ax[0].plot(n_list, bm, "o-", color=C_MIX, lw=2.6, label="(B) mixed: learned combo -> noise")
    ax[0].plot(n_list, ba, "^:", color=C_A, lw=1.6, alpha=0.8, label="(A) field = input (upper ref.)")
    ax[0].set(title="(a) mixing the noise map closes the gap to (A)",
              xlabel="NARMA memory order x", ylabel="test NRMSE")
    ax[0].legend(fontsize=8.3)

    labels = list(fields); x = np.arange(len(labels)); w = 0.36
    ax[1].bar(x - w / 2, [fd[k] for k in labels], w, color=C_DIAG, label="(B) diagonal")
    ax[1].bar(x + w / 2, [fm[k] for k in labels], w, color=C_MIX, label="(B) mixed")
    ax[1].axhline(esn_x[1], color=C_ESN, ls="--", lw=1, label=f"ESN (NARMA-20) = {esn_x[1]:.2f}")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[1].set(title="(b) mixing rescues single-lag fields: the noise MAP does the lag-mixing",
              ylabel="test NRMSE (NARMA-20)")
    ax[1].legend(fontsize=8.3)

    fig.suptitle("Refining (B): the noise map's connectivity — not the crossing shape — is what "
                 "separated it from (A); mixing makes a pure-memory field suffice", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    figpath = os.path.join(OUT, "poc_B_mix.png")
    fig.savefig(figpath, dpi=130)

    print("(a) NARMA-x:")
    print(f"  {'x':>4} {'ESN':>7} {'(B)diag':>8} {'(B)mix':>8} {'(A)':>7}")
    for n, e, d, m, a in zip(n_list, esn_x, bd, bm, ba):
        print(f"  {n:>4} {e:>7.3f} {d:>8.3f} {m:>8.3f} {a:>7.3f}")
    print("(b) single-lag fields (NARMA-20):")
    for k in labels:
        print(f"  {k:>8}  diag={fd[k]:.3f}  mix={fm[k]:.3f}")
    print("saved:", figpath)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    run(quick=a.quick)
