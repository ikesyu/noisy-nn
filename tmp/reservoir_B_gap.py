"""軸3の調査 — なぜ (B)-mix が (A) を稀に上回るか (docs §10.20 予定).

再解釈: (A) も (B)-mix も「場の学習線形射影 + 固定形の非線形活性 + 線形読出」で、
違いは活性の形だけ:
  (A):     phi_A(u) = 2Phi(u)(1-Phi(u))            = バンプ(局所・非単調, RBF的)
  (B)-mix: phi_B(u) = 2Phi(d/(floor+softplus(u)))(1-...) = 飽和単調(シグモイド的, ridge特徴)
仮説: 大域的に滑らかな NARMA では、大域シグモイド基底(B-mix)が少数ユニットで汎化し、
局所バンプ基底(A)は過学習しやすい。決定的シグネチャ = train-test ギャップ。

(3.1) 効果は実在か: x を細かく, 多 seed で (A)/(B-mix) の test NRMSE と paired gap.
(3.2) 過学習シグネチャ: 同時に train NRMSE も測り, generalization gap = test - train.
(3.3) 特徴の実効ランク(participation ratio)を x=10 で比較.

Run:  .venv/bin/python tmp/reservoir_B_gap.py [--quick]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import reservoir as R
from reservoir.readout import nrmse, standardize_fit


def masks(T, washout=300, frac=0.7):
    idx = np.arange(washout, T); ntr = int(len(idx) * frac)
    tr = np.zeros(T, bool); tr[idx[:ntr]] = True
    te = np.zeros(T, bool); te[idx[ntr:]] = True
    return tr, te


def fit_scores(model, tr, te, epochs, y):
    """Train the forward-only map, return (train NRMSE, test NRMSE)."""
    for _ in range(epochs):
        model.step()
    yhat = model.predict_all()
    return nrmse(y[tr], yhat[tr]), nrmse(y[te], yhat[te])


def feat_z(model):
    """Post-training hidden features z (first return of _feat)."""
    out = model._feat()
    return out[0]


def participation_ratio(Z, tr):
    Zs = (Z - Z[tr].mean(0)) / (Z[tr].std(0) + 1e-9)
    s = np.linalg.svd(Zs[tr], compute_uv=False)
    s2 = s ** 2
    return float((s2.sum() ** 2) / (np.sum(s2 ** 2) + 1e-30))


def run(quick=False):
    T = 4000 if quick else 6000
    epochs = 400 if quick else 500
    H = 48
    x_list = [8, 10, 12, 15, 20, 30, 40]
    seeds = list(range(4 if quick else 6))
    tr, te = masks(T)

    print("(3.1)+(3.2) test/train NRMSE and gaps (mean over seeds):")
    print(f"  {'x':>3} | {'A_te':>6} {'B_te':>6} {'gap(A-B)':>9} | "
          f"{'A_tr':>6} {'B_tr':>6} | {'A_ovf':>6} {'B_ovf':>6}")
    for x in x_list:
        Ate, Btr_, Bte, Atr, gaps = [], [], [], [], []
        for sd in seeds:
            u, y = R.narma_x(T, x, seed=1 + sd)
            A = R.LDNField(H=H, theta=60.0).run(u)
            a = R.LearnedCrossingMap(A, y, tr, Ho=H, seed=100 + sd)
            b = R.NoiseModulatedMap(A, y, tr, Ho=H, mix=True, seed=100 + sd)
            atr, ate = fit_scores(a, tr, te, epochs, y)
            btr, bte = fit_scores(b, tr, te, epochs, y)
            Ate.append(ate); Bte.append(bte); Atr.append(atr); Btr_.append(btr)
            gaps.append(ate - bte)
        Ate, Bte, Atr, Btr_ = map(np.array, (Ate, Bte, Atr, Btr_))
        g = np.array(gaps)
        print(f"  {x:>3} | {Ate.mean():>6.3f} {Bte.mean():>6.3f} "
              f"{g.mean():>+6.3f}±{g.std():>4.3f} | {Atr.mean():>6.3f} {Btr_.mean():>6.3f} | "
              f"{(Ate-Atr).mean():>6.3f} {(Bte-Btr_).mean():>6.3f}")

    print("\n(3.3) feature participation ratio at x=10 (mean over seeds; H=48):")
    prA, prB = [], []
    for sd in seeds:
        u, y = R.narma_x(T, 10, seed=1 + sd)
        A = R.LDNField(H=H, theta=60.0).run(u)
        a = R.LearnedCrossingMap(A, y, tr, Ho=H, seed=100 + sd)
        b = R.NoiseModulatedMap(A, y, tr, Ho=H, mix=True, seed=100 + sd)
        for _ in range(epochs): a.step()
        for _ in range(epochs): b.step()
        prA.append(participation_ratio(feat_z(a), tr))
        prB.append(participation_ratio(feat_z(b), tr))
    print(f"  (A) bump PR   = {np.mean(prA):.2f}")
    print(f"  (B-mix) PR    = {np.mean(prB):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    run(quick=a.quick)
