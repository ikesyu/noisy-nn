"""SR curve for the (B)-mix noise-modulated reservoir (docs/idea_reservoir.md
§13.1): is the additive noise a functional RESOURCE, or just a smooth way to
implement a monotone nonlinearity?

Same-weights diagnostic (idea_neuromod.md §8 TODO). Train ONE mixed (B) map with the
analytic (mean-field) crossing, freeze (d, M, c), then sweep the global noise
strength s and score BOTH crossing responses on the SAME weights:
    analytic  z = 2 Phi(d/sigma)(1-Phi(d/sigma))              (mean-field, h=0)
    sample    finite-numT Monte-Carlo crossing, threshold h>0 (real mechanism)
sigma_k(t) = s * (floor + softplus((M A)_k + c_k)) is the trained noise scale
scaled by s. h is FIXED, so small s falls below threshold and cannot cross -- the
SR barrier the mean-field lacks.

Two global knobs (--knob):
    sigma : scale the whole sigma by s (signal+noise together; the preview form).
    floor : add baseline noise, sigma = s*floor0 + softplus(...), signal gain M
            fixed (textbook additive-noise SR).

Pass (§13.1): sample -> interior optimum (reverse-U) across seeds, analytic ->
optimum gone / much shallower. Fail -> drop "noise-modulated", rename "gated".

Run:  python reservoir_sr.py                       # default LDN, NARMA-10/20
      python reservoir_sr.py --knob floor --quick  # fast preview
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import reservoir as R
from reservoir import sr


def masks(T, washout=300, frac=0.7):
    idx = np.arange(washout, T)
    ntr = int(len(idx) * frac)
    tr = np.zeros(T, bool); tr[idx[:ntr]] = True
    te = np.zeros(T, bool); te[idx[ntr:]] = True
    return tr, te


def sigma_scaled(m, s, knob):
    """Global-noise-strength knob applied to a trained mix map -> sigma[T,Ho]."""
    pre = m.A @ m.M.T + m.c
    sp = sr._softplus(pre)
    if knob == "sigma":
        return s * (m.floor + sp)               # scale signal + noise together
    if knob == "floor":
        return s * m.floor + sp                 # add baseline noise, gain fixed
    raise ValueError(knob)


def run_seed(x, seed, field_fn, s_grid, knob, H, epochs, numT, h, kmax, T):
    u, y = R.narma_x(T, x, seed=seed)
    tr, te = masks(T)
    A = field_fn(u)
    m = R.NoiseModulatedMap(A, y, tr, Ho=H, mix=True, seed=100 + seed)
    for _ in range(epochs):                      # train the map (analytic)
        m.step()
    out = {k: np.empty(len(s_grid)) for k in
           ("na", "ns", "mca", "mcs")}
    for i, s in enumerate(s_grid):
        sig = sigma_scaled(m, s, knob)
        za = sr.analytic_z(m.d, sig)
        zs = sr.sample_z(m.d, sig, numT=numT, h=h, seed=1000 + i)
        out["na"][i] = sr.task_nrmse_feats(za, y, tr, te)
        out["ns"][i] = sr.task_nrmse_feats(zs, y, tr, te)
        out["mca"][i] = sr.memory_capacity_feats(za, u, tr, te, kmax=kmax)
        out["mcs"][i] = sr.memory_capacity_feats(zs, u, tr, te, kmax=kmax)
    return out


def summarise(curve):
    """mean/std/argmin over seeds; interior-optimum verdict for a NRMSE curve."""
    arr = np.stack(curve)                        # [seeds, s]
    mean, std = arr.mean(0), arr.std(0)
    amins = arr.argmin(1)
    n = arr.shape[1]
    interior = np.mean((amins > 0) & (amins < n - 1))
    depth = float(np.mean(np.max(arr, 1) - np.min(arr, 1)))
    return mean, std, amins, interior, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", choices=["sigma", "floor"], default="sigma")
    ap.add_argument("--xs", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--H", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--numT", type=int, default=64)
    ap.add_argument("--h", type=float, default=0.2)
    ap.add_argument("--kmax", type=int, default=40)
    ap.add_argument("--T", type=int, default=3000)
    ap.add_argument("--s-steps", type=int, default=21)
    ap.add_argument("--s-min", type=float, default=0.1)
    ap.add_argument("--s-max", type=float, default=10.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="out/reservoir_sr")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.epochs, args.s_steps, args.numT, args.T = 3, 120, 13, 48, 2000
        args.kmax = 25

    s_grid = np.geomspace(args.s_min, args.s_max, args.s_steps)
    field_fn = lambda u: R.LDNField(H=args.H, theta=60.0).run(u)
    os.makedirs(args.out, exist_ok=True)

    results = {}                                 # x -> dict of seed-stacked curves
    for x in args.xs:
        acc = {k: [] for k in ("na", "ns", "mca", "mcs")}
        for sd in range(args.seeds):
            o = run_seed(x, sd, field_fn, s_grid, args.knob, args.H,
                         args.epochs, args.numT, args.h, args.kmax, args.T)
            for k in acc:
                acc[k].append(o[k])
            print(f"  x={x} seed={sd} done")
        results[x] = acc

    # ---- verdict ----
    print(f"\n=== SR verdict (knob={args.knob}, h={args.h}, numT={args.numT}) ===")
    verdict_lines = []
    for x in args.xs:
        for tag, key in (("sample", "ns"), ("analytic", "na")):
            _, _, amins, interior, depth = summarise(results[x][key])
            s_stars = s_grid[amins]
            line = (f"NARMA-{x:<2d} {tag:8s}: interior-opt frac={interior:.2f} "
                    f"depth={depth:.3f}  s*={np.round(s_stars, 2)}")
            print("  " + line)
            verdict_lines.append(line)

    # ---- figure ----
    nx = len(args.xs)
    fig, ax = plt.subplots(2, nx, figsize=(5.2 * nx, 8), squeeze=False)
    for j, x in enumerate(args.xs):
        for row, (mkey, akey, ylab, better) in enumerate([
                ("ns", "na", f"NARMA-{x} test NRMSE", "lower=better"),
                ("mcs", "mca", "total memory capacity", "higher=better")]):
            a = ax[row, j]
            for key, col, lab in ((akey, "#4C72B0", "analytic (mean-field)"),
                                  (mkey, "#C44E52", "sample (mechanism)")):
                arr = np.stack(results[x][key])
                mean, std = arr.mean(0), arr.std(0)
                a.plot(s_grid, mean, "-o", ms=4, color=col, label=lab)
                a.fill_between(s_grid, mean - std, mean + std, color=col, alpha=0.2)
                if row == 0:                     # mark interior optimum of sample
                    im = mean.argmin()
                    a.axvline(s_grid[im], color=col, ls=":", alpha=0.5)
            a.set_xscale("log")
            a.set(xlabel="global noise strength  s", ylabel=ylab,
                  title=f"NARMA-{x} — {ylab.split()[-1]} ({better})")
            a.grid(alpha=0.25)
            if row == 0 and j == 0:
                a.legend(fontsize=9)
    fig.suptitle(f"SR curve, same weights (knob={args.knob}, h={args.h}, "
                 f"numT={args.numT}, {args.seeds} seeds) — §13.1", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fpng = os.path.join(args.out, f"sr_curve_{args.knob}.png")
    fig.savefig(fpng, dpi=130)
    print(f"\nsaved figure -> {fpng}")

    # ---- CSV (per task: s, na_mean, ns_mean, mca_mean, mcs_mean) ----
    for x in args.xs:
        rows = [s_grid]
        for k in ("na", "ns", "mca", "mcs"):
            rows.append(np.stack(results[x][k]).mean(0))
        M = np.column_stack(rows)
        fcsv = os.path.join(args.out, f"sr_{args.knob}_narma{x}.csv")
        np.savetxt(fcsv, M, delimiter=",",
                   header="s,na_mean,ns_mean,mca_mean,mcs_mean", comments="# ")
        print(f"saved csv    -> {fcsv}")


if __name__ == "__main__":
    main()
