"""Minimal, crisp proof-of-concept — the ON-BRAND (B) model (docs §10.18-10.19).

THESIS. A dissipative NOISE FIELD (raw temporal memory) supplies the ADDITIVE
per-unit NOISE of an NNN whose inputs are FIXED operating points; only the
feedforward map is learned, FORWARD-ONLY (no BPTT). The field is the NOISE, not
the signal input. Each unit's noise scale is a LEARNED linear combination of the
field coordinates (mix), so the NNN's noise map itself does the nonlinear
lag-mixing — a pure-memory field then suffices and (B) reaches the (A) upper
reference while beating a standard ESN. The field decays to zero (ESP).

Uses only the clean library tmp/reservoir/. Panels:
 (a) Ours (B-mix) beats ESN and reaches (A) on NARMA-x; diagonal (B) is weaker.
 (b) the noise field decays to zero without input (ESP).
 (c) the LDN noise field = hippocampal TIME CELLS (biological grounding).
 (d) the noise MAP does the lag-mixing: mixing rescues single-lag (delay/cascade)
     fields that diagonal (B) fails on — memory in the field, nonlinearity in the NNN.

Run:  .venv/bin/python tmp/reservoir_poc.py [--quick]
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

C_ESN = "#2563eb"; C_MIX = "#0d9488"; C_DIAG = "#94a3b8"; C_A = "#ea7317"; C_CAS = "#9333ea"
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

    # (a) NARMA-x: ESN vs Ours(B-mix) [headline] vs (B-diag) vs (A upper ref)
    esn_x, bMix, bDiag, bA = [], [], [], []
    for n in n_list:
        e_s, m_s, d_s, a_s = [], [], [], []
        for sd in seeds:
            u, y = R.narma_x(T, n, seed=1 + sd)
            A = R.LDNField(H=H, theta=60.0).run(u)
            e_s.append(R.task_nrmse(R.LeakyESN(H=H, seed=sd).run(u), y, washout=300))
            m_s.append(R.NoiseModulatedMap(A, y, tr, Ho=H, mix=True, seed=100 + sd).eval(te, epochs))
            d_s.append(R.NoiseModulatedMap(A, y, tr, Ho=H, mix=False, seed=100 + sd).eval(te, epochs))
            a_s.append(R.LearnedCrossingMap(A, y, tr, Ho=H, seed=100 + sd).eval(te, epochs))
        esn_x.append(np.mean(e_s)); bMix.append(np.mean(m_s))
        bDiag.append(np.mean(d_s)); bA.append(np.mean(a_s))

    # (d) the noise MAP does the mixing: diag vs mix on single-lag / pre-mixed fields
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
    esn20 = esn_x[n_list.index(20)]

    # (b) dissipation
    decay = {"LDN": R.pulse_decay(R.LDNField(H=H, theta=60.0)),
             "cascade": R.pulse_decay(R.CascadeField(H=H)),
             "damped-orth": R.pulse_decay(R.DampedOrthField(H=H, rho=0.97))}

    # (c) LDN time cells
    ldn = R.LDNField(H=H, theta=60.0)
    Tp = 130; pulse = 5; up = np.zeros(Tp); up[pulse] = 1.0
    M = ldn.run(up)
    rs = np.linspace(0.02, 0.98, 48)
    Cw = np.array([ldn.decode_weights(r) for r in rs])
    act = M @ Cw.T; act /= (np.abs(act).max(0, keepdims=True) + 1e-9)
    tt = np.arange(Tp) - pulse

    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    ax[0, 0].plot(n_list, esn_x, "s-", color=C_ESN, lw=2.4, label="ESN (reservoir does both)")
    ax[0, 0].plot(n_list, bMix, "o-", color=C_MIX, lw=2.6, label="Ours (B-mix): field = additive NOISE")
    ax[0, 0].plot(n_list, bDiag, "o--", color=C_DIAG, lw=1.6, alpha=0.9, label="(B-diagonal): 1 coord -> noise")
    ax[0, 0].plot(n_list, bA, "^:", color=C_A, lw=1.6, alpha=0.8, label="(A) field = input (upper ref.)")
    ax[0, 0].set(title="(a) Ours (B-mix) beats ESN and reaches (A) — the field is the NNN's NOISE",
                 xlabel="NARMA memory order x", ylabel="test NRMSE"); ax[0, 0].legend(fontsize=8.3)

    for k, c in (("LDN", C_MIX), ("cascade", C_CAS), ("damped-orth", "#a855f7")):
        ax[0, 1].plot(decay[k], color=c, lw=2, label=k)
    ax[0, 1].set(title="(b) the noise field decays to zero without input (ESP)",
                 xlabel="steps after an input pulse", ylabel="‖field‖ (normalised)"); ax[0, 1].legend(fontsize=9)

    im = ax[1, 0].imshow(act.T, aspect="auto", origin="lower", cmap="magma",
                         extent=[tt[0], tt[-1], rs[0] * 60, rs[-1] * 60], vmin=-0.3, vmax=1)
    ax[1, 0].set(title="(c) the LDN noise field reads out as hippocampal TIME CELLS",
                 xlabel="time since stimulus (steps)", ylabel="cell's preferred delay")
    ax[1, 0].set_xlim(-2, 88); fig.colorbar(im, ax=ax[1, 0], label="norm. activity")

    labels = list(fields); x = np.arange(len(labels)); w = 0.36
    ax[1, 1].bar(x - w / 2, [fd[k] for k in labels], w, color=C_DIAG, label="(B-diagonal)")
    ax[1, 1].bar(x + w / 2, [fm[k] for k in labels], w, color=C_MIX, label="(B-mix)")
    ax[1, 1].axhline(esn20, color=C_ESN, ls="--", lw=1, label=f"ESN = {esn20:.2f}")
    ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(labels)
    ax[1, 1].set(title="(d) the noise MAP does the lag-mixing: (B-mix) rescues single-lag fields",
                 ylabel="test NRMSE (NARMA-20)"); ax[1, 1].legend(fontsize=8.3)

    fig.suptitle("Ours (B): a dissipative noise field is the NNN's additive noise; the learned noise map "
                 "does the lag-mixing — beats ESN, reaches (A), decays to zero, forward-only", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    figpath = os.path.join(OUT, "poc_summary.png")
    fig.savefig(figpath, dpi=130)

    print("MINIMAL PROOF OF CONCEPT (B: field = additive noise, mix, forward-only, fair vs ESN):")
    print(f"  {'x':>4} {'ESN':>7} {'Ours(B-mix)':>12} {'(B-diag)':>9} {'(A) ref':>8}")
    for n, e, m, d, a in zip(n_list, esn_x, bMix, bDiag, bA):
        print(f"  {n:>4} {e:>7.3f} {m:>12.3f} {d:>9.3f} {a:>8.3f}")
    print("  (d) noise-map mixing (NARMA-20): "
          + ", ".join(f"{k} diag={fd[k]:.3f}/mix={fm[k]:.3f}" for k in labels) + f"; ESN={esn20:.3f}")
    print(f"  saved: {figpath}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    run(quick=a.quick)
