"""(B) の詰め — 軸1(σ の線形混合)と 軸4((A) との差の分解).

(1) まず (B)-mix の forward-only 勾配を有限差分で検証(readout W,mu,sd を凍結した
    損失に対する d,c,M の勾配が一致するか).
(2) NARMA-x: (A), (B)-diag, (B)-mix, ESN を多 seed 比較.
(3) 単一ラグ場テスト: delay/cascade/LDN で (B)-diag vs (B)-mix.
    混合がノイズマップ側でラグ結合できるなら, (B)-mix は単一ラグ場でも効くはず.

Run:  .venv/bin/python tmp/reservoir_B_mix.py [--quick]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import reservoir as R


def masks(T, washout=300, frac=0.7):
    idx = np.arange(washout, T); ntr = int(len(idx) * frac)
    tr = np.zeros(T, bool); tr[idx[:ntr]] = True
    te = np.zeros(T, bool); te[idx[ntr:]] = True
    return tr, te


def grad_check():
    """Finite-difference check of (B)-mix forward-only credit (W,mu,sd frozen)."""
    rng = np.random.default_rng(0)
    T, Hf, Ho = 400, 8, 6
    A = rng.standard_normal((T, Hf))
    y = rng.standard_normal(T)
    tr, _ = masks(T, washout=50)
    m = R.NoiseModulatedMap(A, y, tr, Ho=Ho, mix=True, seed=1)
    m._t = 1

    def loss_frozen(W, mu, sd):
        z, _, _, _ = m._feat()
        X = np.concatenate([z, np.ones((len(z), 1))], axis=1)
        e = np.where(m.tr, (X - mu) / sd @ W - m.y, 0.0)
        return (e ** 2).sum() / m.tr.sum()

    # analytic grads from step() internals, with W,mu,sd frozen
    z, sig, dz_darg, pre = m._feat()
    W, mu, sd, X = m._readout(z)
    e = np.where(m.tr, (X - mu) / sd @ W - m.y, 0.0); n = m.tr.sum()
    base = (2.0 / n) * e[:, None] * (W[:m.Ho] / sd[0, :m.Ho])[None, :] * dz_darg
    gd = (base / sig).sum(0)
    dLdpre = base * (-m.d / sig ** 2) * m._sigmoid(pre)
    gc = dLdpre.sum(0)
    gM = dLdpre.T @ m.A

    eps = 1e-6; errs = {}
    for name, p, gan in (("d", m.d, gd), ("c", m.c, gc), ("M", m.M, gM)):
        gnum = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        while not it.finished:
            i = it.multi_index; o = p[i]
            p[i] = o + eps; lp = loss_frozen(W, mu, sd)
            p[i] = o - eps; lm = loss_frozen(W, mu, sd)
            p[i] = o; gnum[i] = (lp - lm) / (2 * eps)
            it.iternext()
        rel = np.linalg.norm(gan - gnum) / (np.linalg.norm(gnum) + 1e-12)
        errs[name] = rel
    return errs


def run(quick=False):
    errs = grad_check()
    print("(1) grad-check (B)-mix (rel err vs finite-diff):",
          ", ".join(f"{k}={v:.2e}" for k, v in errs.items()))
    ok = all(v < 1e-4 for v in errs.values())
    print("    -> " + ("PASS" if ok else "FAIL"))

    T = 4000 if quick else 6000
    epochs = 400 if quick else 500
    H = 48
    n_list = [10, 20, 30, 40]
    seeds = [0, 1, 2]
    tr, te = masks(T)

    print("\n(2) NARMA-x  (test NRMSE, mean over seeds):")
    print(f"  {'x':>4} {'ESN':>7} {'(B)diag':>8} {'(B)mix':>8} {'(A)':>7}")
    rows = []
    for n in n_list:
        e_s, bd_s, bm_s, a_s = [], [], [], []
        for sd in seeds:
            u, y = R.narma_x(T, n, seed=1 + sd)
            A = R.LDNField(H=H, theta=60.0).run(u)
            e_s.append(R.task_nrmse(R.LeakyESN(H=H, seed=sd).run(u), y, washout=300))
            bd_s.append(R.NoiseModulatedMap(A, y, tr, Ho=H, mix=False, seed=100 + sd).eval(te, epochs))
            bm_s.append(R.NoiseModulatedMap(A, y, tr, Ho=H, mix=True, seed=100 + sd).eval(te, epochs))
            a_s.append(R.LearnedCrossingMap(A, y, tr, Ho=H, seed=100 + sd).eval(te, epochs))
        row = (n, np.mean(e_s), np.mean(bd_s), np.mean(bm_s), np.mean(a_s))
        rows.append(row)
        print(f"  {row[0]:>4} {row[1]:>7.3f} {row[2]:>8.3f} {row[3]:>8.3f} {row[4]:>7.3f}")

    print("\n(3) single-lag-field test (NARMA-20): does mixing let (B) work on delay/cascade?")
    fields = {"delay": lambda: R.DelayLineField(H=H),
              "cascade": lambda: R.CascadeField(H=H, a=0.92),
              "LDN": lambda: R.LDNField(H=H, theta=60.0)}
    print(f"  {'field':>8} {'(B)diag':>8} {'(B)mix':>8}")
    for fn, mk in fields.items():
        d_s, m_s = [], []
        for sd in seeds:
            u, y = R.narma_x(T, 20, seed=1 + sd)
            Af = mk().run(u)
            d_s.append(R.NoiseModulatedMap(Af, y, tr, Ho=H, mix=False, seed=100 + sd).eval(te, epochs))
            m_s.append(R.NoiseModulatedMap(Af, y, tr, Ho=H, mix=True, seed=100 + sd).eval(te, epochs))
        print(f"  {fn:>8} {np.mean(d_s):>8.3f} {np.mean(m_s):>8.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    run(quick=a.quick)
