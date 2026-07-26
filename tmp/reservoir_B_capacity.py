"""軸1の調査 — 2 つの独立な容量ダイヤル: Ho(非線形) と Hf(メモリ) (docs §10.21 予定).

分離アーキの核心的うまみ = メモリ容量(場サイズ Hf)と非線形容量(NNN 隠れ幅 Ho)を
独立に配分できる(ESN は 1 つの H が両方を担う)。それを 2 つのスイープで示す:
  行1 (Ho スイープ, Hf=48 固定): 短記憶(NARMA-10)は非線形律速 -> Ho で大改善、
      長記憶(20/40)はメモリ律速 -> Ho を増やしても plateau。
  行2 (Hf スイープ, Ho=96 固定): 長記憶は Hf(場のメモリ解像度)で改善、短記憶は Hf に鈍感。
幅を揃えた ESN(H=Ho or H=Hf)を参照に置き、分離の優位が両軸で保たれるか見る。

Run:  .venv/bin/python tmp/reservoir_B_capacity.py [--quick]
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

C_ESN = "#2563eb"; C_MIX = "#0d9488"; C_A = "#ea7317"; C_ESNW = "#93c5fd"
OUT = os.path.join(os.path.dirname(__file__), "out", "reservoir_poc")


def masks(T, washout=300, frac=0.7):
    idx = np.arange(washout, T); ntr = int(len(idx) * frac)
    tr = np.zeros(T, bool); tr[idx[:ntr]] = True
    te = np.zeros(T, bool); te[idx[ntr:]] = True
    return tr, te


def run(quick=False):
    os.makedirs(OUT, exist_ok=True)
    T = 3500 if quick else 4000
    epochs = 300 if quick else 350
    x_list = [10, 20, 40]
    seeds = list(range(3))
    tr, te = masks(T)

    Ho_list = [12, 24, 48, 96, 144]           # row 1: vary Ho, Hf fixed
    Hf_fix = 48
    Hf_list = [24, 48, 96]                     # row 2: vary Hf, Ho fixed
    Ho_fix = 96

    # ---- row 1: Ho sweep (Hf = 48 fixed) ----
    r1 = {x: {"A": [], "Bmix": [], "ESNw": []} for x in x_list}
    esn_fixed = {}
    for x in x_list:
        e_f = []
        for sd in seeds:
            u, y = R.narma_x(T, x, seed=1 + sd)
            e_f.append(R.task_nrmse(R.LeakyESN(H=Hf_fix, seed=sd).run(u), y, washout=300))
        esn_fixed[x] = np.mean(e_f)
        for Ho in Ho_list:
            a_s, m_s, ew_s = [], [], []
            for sd in seeds:
                u, y = R.narma_x(T, x, seed=1 + sd)
                A = R.LDNField(H=Hf_fix, theta=60.0).run(u)
                a_s.append(R.LearnedCrossingMap(A, y, tr, Ho=Ho, seed=100 + sd).eval(te, epochs))
                m_s.append(R.NoiseModulatedMap(A, y, tr, Ho=Ho, mix=True, seed=100 + sd).eval(te, epochs))
                ew_s.append(R.task_nrmse(R.LeakyESN(H=Ho, seed=sd).run(u), y, washout=300))
            r1[x]["A"].append(np.mean(a_s)); r1[x]["Bmix"].append(np.mean(m_s)); r1[x]["ESNw"].append(np.mean(ew_s))

    # ---- row 2: Hf sweep (Ho = 96 fixed) ----
    r2 = {x: {"A": [], "Bmix": [], "ESNw": []} for x in x_list}
    for x in x_list:
        for Hf in Hf_list:
            a_s, m_s, ew_s = [], [], []
            for sd in seeds:
                u, y = R.narma_x(T, x, seed=1 + sd)
                A = R.LDNField(H=Hf, theta=60.0).run(u)
                a_s.append(R.LearnedCrossingMap(A, y, tr, Ho=Ho_fix, seed=100 + sd).eval(te, epochs))
                m_s.append(R.NoiseModulatedMap(A, y, tr, Ho=Ho_fix, mix=True, seed=100 + sd).eval(te, epochs))
                ew_s.append(R.task_nrmse(R.LeakyESN(H=Hf, seed=sd).run(u), y, washout=300))
            r2[x]["A"].append(np.mean(a_s)); r2[x]["Bmix"].append(np.mean(m_s)); r2[x]["ESNw"].append(np.mean(ew_s))

    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, ax = plt.subplots(2, len(x_list), figsize=(5 * len(x_list), 8.6))
    for j, x in enumerate(x_list):
        a = ax[0, j]
        a.plot(Ho_list, r1[x]["Bmix"], "o-", color=C_MIX, lw=2.4, label="(B-mix) field=noise")
        a.plot(Ho_list, r1[x]["A"], "^:", color=C_A, lw=1.8, label="(A) field=input")
        a.plot(Ho_list, r1[x]["ESNw"], "s-", color=C_ESNW, lw=1.8, label="ESN (H = Ho)")
        a.axhline(esn_fixed[x], color=C_ESN, ls="--", lw=1, label=f"ESN H=48 = {esn_fixed[x]:.2f}")
        a.axvline(Hf_fix, color="#888", ls=":", lw=1)
        a.set(title=f"NARMA-{x}: vary Ho (Hf=48 fixed)", xlabel="NNN hidden units Ho",
              ylabel="test NRMSE" if j == 0 else "")
        a.set_xscale("log"); a.set_xticks(Ho_list); a.set_xticklabels(Ho_list)
        if j == 0: a.legend(fontsize=8.0)

        b = ax[1, j]
        b.plot(Hf_list, r2[x]["Bmix"], "o-", color=C_MIX, lw=2.4, label="(B-mix) field=noise")
        b.plot(Hf_list, r2[x]["A"], "^:", color=C_A, lw=1.8, label="(A) field=input")
        b.plot(Hf_list, r2[x]["ESNw"], "s-", color=C_ESNW, lw=1.8, label="ESN (H = Hf)")
        b.axvline(Ho_fix, color="#888", ls=":", lw=1)
        b.set(title=f"NARMA-{x}: vary Hf (Ho=96 fixed)", xlabel="field size Hf (LDN coeffs)",
              ylabel="test NRMSE" if j == 0 else "")
        b.set_xscale("log"); b.set_xticks(Hf_list); b.set_xticklabels(Hf_list)
        if j == 0: b.legend(fontsize=8.0)

    fig.suptitle("Axis 1 — two independent capacity dials: Ho (nonlinearity) and Hf (memory). "
                 "Short memory is Ho-bound; long memory is Hf-bound — separation lets you set each.", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figpath = os.path.join(OUT, "poc_B_capacity.png")
    fig.savefig(figpath, dpi=130)

    print("=== row 1: vary Ho (Hf=48) ===")
    for x in x_list:
        print(f"NARMA-{x} (ESN H=48 = {esn_fixed[x]:.3f}):  Ho -> (B-mix)/(A)/ESN(Ho)")
        for i, Ho in enumerate(Ho_list):
            print(f"  {Ho:>4} {r1[x]['Bmix'][i]:.3f} / {r1[x]['A'][i]:.3f} / {r1[x]['ESNw'][i]:.3f}")
    print("=== row 2: vary Hf (Ho=96) ===")
    for x in x_list:
        print(f"NARMA-{x}:  Hf -> (B-mix)/(A)/ESN(Hf)")
        for i, Hf in enumerate(Hf_list):
            print(f"  {Hf:>4} {r2[x]['Bmix'][i]:.3f} / {r2[x]['A'][i]:.3f} / {r2[x]['ESNw'][i]:.3f}")
    print("saved:", figpath)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    run(quick=a.quick)
