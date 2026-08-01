"""(b1) Polynomial vs non-polynomial complementarity with NG-RC
(docs/idea_reservoir.md §13.3 path (b)).

Hypothesis tested and REFUTED: that a nonlinear functional of a long window makes
NG-RC pay ~theta^2 features while LDN compresses. It does not -- NG-RC represents
a windowed quadratic EXACTLY and cheaply (NRMSE 0.000), and a smooth window is
sub-samplable (stride). What IS real is a clean complementarity in the OUTER
nonlinearity: with s(t) = m1(t)^2 - m2(t)^2 the windowed quadratic argument
(m_r = r-th Legendre moment of a length-theta window),

    y(t) = g(s(t)),   g in {linear, tanh, sign-threshold, gaussian-detector}.

NG-RC's read-out is LINEAR over polynomial features, so it fits g=polynomial
perfectly but hits a budget-INDEPENDENT floor on non-polynomial g. Ours' learned
crossing (bump) map is a universal-but-imperfect nonlinearity: worse than NG-RC on
the polynomial g, better than NG-RC's floor on non-polynomial g. This characterises
the two methods' home turfs (NARMA / polynomials = NG-RC ; thresholds / detectors
= Ours), rather than claiming Ours 'wins'.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import eval_legendre

import reservoir as R
from reservoir.baselines import NGRC, budget_ngrc, budget_ours, budget_lmu
from reservoir.metrics import task_nrmse

THETA, RRANK, HF = 40, 2, 16


def masks(T, washout=300, frac=0.7):
    idx = np.arange(washout, T); ntr = int(len(idx) * frac)
    tr = np.zeros(T, bool); tr[idx[:ntr]] = True
    te = np.zeros(T, bool); te[idx[ntr:]] = True
    return tr, te


def quad_arg(u, theta=THETA, Rr=RRANK):
    """s(t) = m1^2 - m2^2, m_r = r-th Legendre moment of the length-theta window."""
    T = len(u); xs = 2.0 * np.arange(theta) / (theta - 1) - 1.0
    M = np.zeros((T, Rr))
    for ri, r in enumerate(range(1, Rr + 1)):
        f = eval_legendre(r, xs)
        for i in range(theta):
            M[i + 1:, ri] += f[i] * u[:T - i - 1]
    M = M / (M.std(0) + 1e-9)
    return M[:, 0] ** 2 - M[:, 1] ** 2


OUTERS = {
    "linear (polynomial)": lambda s: s,
    "tanh(s-1)": lambda s: np.tanh(s - 1),
    "sign(s-0.3)": lambda s: np.sign(s - 0.3),
    "gauss exp(-s^2)": lambda s: np.exp(-s ** 2),
}


def target(u, g):
    y = g(quad_arg(u)); return y - y.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--out", default="out/reservoir_longwindow")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    Hos = [4, 8, 16, 32, 64]
    ng_lags = [8, 12, 16, 24, 32, 42]

    def data(sd, g):
        u = np.random.default_rng(sd).uniform(-1, 1, args.T)
        return u, target(u, g), *masks(args.T)

    def ours_err(g, Ho):
        vals = []
        for sd in range(args.seeds):
            u, y, tr, te = data(sd, g)
            A = R.LDNField(H=HF, theta=float(THETA)).run(u)
            vals.append(R.NoiseModulatedMap(A, y, tr, Ho=Ho, mix=True,
                                            seed=100 + sd).eval(te, args.epochs))
        return float(np.mean(vals))

    def lmu_err(g, Ho):
        vals = []
        for sd in range(args.seeds):
            u, y, tr, te = data(sd, g)
            A = R.LDNField(H=HF, theta=float(THETA)).run(u)
            vals.append(R.LearnedCrossingMap(A, y, tr, Ho=Ho,
                                             seed=100 + sd).eval(te, args.epochs))
        return float(np.mean(vals))

    def ngrc_err(g, n):
        stride = 1 if n >= 42 else max(1, round(THETA / n))
        vals = []
        for sd in range(args.seeds):
            u, y, tr, te = data(sd, g)
            X = NGRC(delay=n, degree=2, stride=stride).features(u)
            vals.append(task_nrmse(X, y, alpha=1e-4))
        return float(np.mean(vals))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # panel 1: budget sweep on the gaussian-detector (non-polynomial) task
    gname = "gauss exp(-s^2)"; g = OUTERS[gname]
    a = ax[0]
    print(f"=== budget sweep, outer = {gname} ===")
    for method, ferr, budg, col, mk in (
            ("Ours(B)", ours_err, budget_ours, "#C44E52", "D"),
            ("LMU(A)", lmu_err, budget_lmu, "#4C72B0", "o")):
        xs, ys = [], []
        for Ho in Hos:
            e = ferr(g, Ho)
            xs.append(budg(Ho, HF)["trainable"]); ys.append(e)
            print(f"  {method} Ho={Ho:3d} p={xs[-1]:5d} NRMSE={e:.3f}")
        a.plot(xs, ys, mk + "-", color=col, label=method)
    xs, ys = [], []
    for n in ng_lags:
        e = ngrc_err(g, n)
        F = NGRC(delay=n, degree=2).feature_dim()
        xs.append(F + 1); ys.append(e)
        print(f"  NG-RC n={n:3d} p={xs[-1]:5d} NRMSE={e:.3f}")
    a.plot(xs, ys, "^-", color="#55A868", label="NG-RC")
    a.set_xscale("log")
    a.set(xlabel="trainable params / feature dim", ylabel="test NRMSE",
          title="(1) budget sweep — non-polynomial target g=gaussian detector")
    a.grid(alpha=0.25); a.legend(fontsize=9)

    # panel 2: complementarity across outer functions (best of each method)
    a = ax[1]
    names = list(OUTERS)
    ng, ou = [], []
    print("=== complementarity across outer g(s) ===")
    for nm in names:
        g = OUTERS[nm]
        e_ng = ngrc_err(g, 42)
        e_ou = min(ours_err(g, 16), ours_err(g, 32))
        ng.append(e_ng); ou.append(e_ou)
        print(f"  {nm:20s} NG-RC={e_ng:.3f}  Ours={e_ou:.3f}")
    x = np.arange(len(names)); w = 0.38
    a.bar(x - w / 2, ng, w, color="#55A868", label="NG-RC (polynomial features)")
    a.bar(x + w / 2, ou, w, color="#C44E52", label="Ours(B) (learned bump features)")
    a.set_xticks(x); a.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    a.set(ylabel="test NRMSE",
          title="(2) complementarity: polynomial vs non-polynomial target")
    a.grid(alpha=0.25, axis="y"); a.legend(fontsize=9)

    fig.suptitle("(b1) NG-RC vs Ours: home turfs are polynomial vs "
                 "non-polynomial (long-window quadratic argument)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fp = os.path.join(args.out, "longwindow.png")
    fig.savefig(fp, dpi=130)
    print(f"\nsaved -> {fp}")


if __name__ == "__main__":
    main()
