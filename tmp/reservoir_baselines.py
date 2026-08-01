"""Matched-budget 'error vs budget' comparison against prior art
(docs/idea_reservoir.md §13.3, decision method (a)).

Methods on a common budget axis (trainable params; total state also recorded):
    ESN        LeakyESN, read-out only.
    NG-RC      Gauthier 2021: delay line + degree-2 polynomial, ridge only.
    LMU        (A) LearnedCrossingMap on an LDN field: LDN memory + learned
               input-coupled crossing + linear read-out, forward-only.
    Ours(B)    NoiseModulatedMap(mix): LDN memory as MULTIPLICATIVE noise scale.

Each method's size is swept; NARMA-x test NRMSE is plotted vs trainable params.
Answers the "Ours just has more trainable params" rebuttal by matching budget,
and exposes how NARMA (a polynomial, exact-lag recurrence) favours NG-RC.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import reservoir as R
from reservoir.baselines import (NGRC, budget_ngrc, budget_esn, budget_lmu,
                                  budget_ours)
from reservoir.metrics import task_nrmse

ALPHA = 1e-2
HF = 48


def eval_point(method, size, u, y, tr, te, seed, epochs):
    """Return (trainable, state, feature, nrmse) for one method at one size."""
    if method == "esn":
        X = R.LeakyESN(H=size, seed=seed).run(u)
        e = task_nrmse(X, y, alpha=ALPHA)
        b = budget_esn(size)
    elif method == "ngrc":
        X = NGRC(delay=size, degree=2).features(u)
        e = task_nrmse(X, y, alpha=ALPHA)
        b = budget_ngrc(size)
    elif method in ("lmu", "ours"):
        A = R.LDNField(H=HF, theta=60.0).run(u)
        if method == "lmu":
            m = R.LearnedCrossingMap(A, y, tr, Ho=size, seed=seed)
            b = budget_lmu(size, HF)
        else:
            m = R.NoiseModulatedMap(A, y, tr, Ho=size, mix=True, seed=seed)
            b = budget_ours(size, HF)
        e = m.eval(te, epochs)
    else:
        raise ValueError(method)
    return b["trainable"], b["state"], b["feature"], e


def masks(T, washout=300, frac=0.7):
    idx = np.arange(washout, T)
    ntr = int(len(idx) * frac)
    tr = np.zeros(T, bool); tr[idx[:ntr]] = True
    te = np.zeros(T, bool); te[idx[ntr:]] = True
    return tr, te


SIZES = {
    "esn":  [12, 24, 48, 96, 192, 384, 768],
    "ngrc": [6, 10, 14, 18, 22, 26, 32],       # delay; F ~ 1+d+d(d+1)/2
    "lmu":  [4, 8, 16, 32, 48, 64],
    "ours": [4, 8, 16, 32, 48, 64],
}
STYLE = {"esn": ("#7f7f7f", "s"), "ngrc": ("#2ca02c", "^"),
         "lmu": ("#4C72B0", "o"), "ours": ("#C44E52", "D")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xs", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--T", type=int, default=3000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="out/reservoir_baselines")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.epochs, args.T = 2, 120, 2000
        for k in SIZES:
            SIZES[k] = SIZES[k][::2]

    os.makedirs(args.out, exist_ok=True)
    results = {}                    # (x, method) -> list over sizes of (tr,state,nrmse) per seed
    for x in args.xs:
        for method in ("esn", "ngrc", "lmu", "ours"):
            pts = []
            for size in SIZES[method]:
                errs, trp, st = [], None, None
                for sd in range(args.seeds):
                    u, y = R.narma_x(args.T, x, seed=sd)
                    tr, te = masks(args.T)
                    trp, st, _, e = eval_point(method, size, u, y, tr, te,
                                               100 + sd, args.epochs)
                    errs.append(e)
                pts.append((trp, st, np.mean(errs), np.std(errs)))
                print(f"  NARMA-{x} {method:5s} size={size:3d} "
                      f"params={trp:5d} NRMSE={np.mean(errs):.3f}")
            results[(x, method)] = pts

    # ---- figure: NRMSE vs trainable params ----
    nx = len(args.xs)
    fig, ax = plt.subplots(1, nx, figsize=(6.2 * nx, 5), squeeze=False)
    for j, x in enumerate(args.xs):
        a = ax[0, j]
        for method in ("esn", "ngrc", "lmu", "ours"):
            pts = results[(x, method)]
            xs = [p[0] for p in pts]; ys = [p[2] for p in pts]
            es = [p[3] for p in pts]
            col, mk = STYLE[method]
            a.errorbar(xs, ys, yerr=es, marker=mk, color=col, capsize=2,
                       label=method.upper() if method != "ours" else "Ours(B)")
        a.set_xscale("log")
        a.set(xlabel="trainable parameters", ylabel=f"NARMA-{x} test NRMSE",
              title=f"NARMA-{x} — error vs budget")
        a.grid(alpha=0.25)
        if j == 0:
            a.legend(fontsize=9)
    fig.suptitle(f"Matched-budget comparison vs prior art (§13.3, {args.seeds} "
                 f"seeds, LDN Hf={HF})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fpng = os.path.join(args.out, "budget_curve.png")
    fig.savefig(fpng, dpi=130)
    print(f"\nsaved figure -> {fpng}")

    # ---- CSV ----
    for x in args.xs:
        for method in ("esn", "ngrc", "lmu", "ours"):
            M = np.array([[p[0], p[1], p[2], p[3]]
                          for p in results[(x, method)]])
            fcsv = os.path.join(args.out, f"budget_narma{x}_{method}.csv")
            np.savetxt(fcsv, M, delimiter=",",
                       header="trainable,state,nrmse_mean,nrmse_std", comments="# ")
    print("saved csvs")


if __name__ == "__main__":
    main()
