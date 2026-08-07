"""Does raising gamma cure lambda's weak identification?  (follow-up to sec10.35)

sec10.35 found lambda barely moves from its initialisation (init 0.1/0.5/0.9 ->
final 0.238/0.598/0.836) and explained why:

    dz/dlambda = flip - rate,   and at p = 0:  rate = Q(0) = 0.5,
                                               flip = 2*0.5*0.5 = 0.5
    => dz/dlambda = 0 EXACTLY at the centre, which is where BN puts every unit.

The drive only appears off-centre and grows with the effective gain:
    E[dz/dlambda] = -0.064 / -0.167 / -0.296  at gamma = 0.5 / 1 / 2.

PREDICTION: raising gamma should restore identifiability -- final lambda should
become independent of its initialisation.  The IDENTIFICATION METRIC is the
SPREAD of final lambda across inits (init spread is 0.8; a spread near 0 means
identified, a spread near 0.6 means it just stayed where it started).

Secondary prediction: once identified, lambda should settle LOW under a plain
regression loss (the 1st moment wins at matched gamma, sec10.33), and the
on-brand penalty should still be able to pull it back up.
"""
import numpy as np
import torch

from reservoir_lambda_local import masks
from reservoir_gamma_recheck import SignField, parity_task
from reservoir_lambda_learn import run

GAMMAS = (0.5, 1.0, 2.0, 3.0, 5.0)
INITS = (0.1, 0.5, 0.9)
SEEDS = 2
DEPTH = 2


def block(A, y, tr, te, beta, tag):
    print(f"\n--- beta = {beta}  ({tag}) ---")
    print(f"  {'gamma':>5} | " + " | ".join(f"init{i:.1f}" for i in INITS)
          + f" | {'spread':>6} {'NRMSE':>6} {'free':>6}")
    for g in GAMMAS:
        lams, errs, frees = [], [], []
        for lam0 in INITS:
            r = [run(A, y, tr, te, DEPTH, g, beta, lam0=lam0, seed=s) for s in range(SEEDS)]
            lams.append(np.mean([x[1] for x in r]))
            errs.append(np.mean([x[0] for x in r]))
            frees.append(np.mean([x[3] for x in r]))
        spread = max(lams) - min(lams)
        mark = "  <- identified" if spread < 0.15 else ("  (partly)" if spread < 0.35 else "")
        print(f"  {g:5.1f} | " + " | ".join(f" {l:.3f}" for l in lams)
              + f" | {spread:6.3f} {np.mean(errs):6.3f} {np.mean(frees):6.3f}{mark}",
              flush=True)


def main():
    T = 3000; u, y = parity_task(T); tr, te = masks(T)
    A = SignField(H=32, gain=8.0).run(u)
    print("=" * 78)
    print(f"Does gamma cure lambda's weak identification?  parity depth={DEPTH}, "
          f"{SEEDS} seeds")
    print("  identification metric = SPREAD of final lambda across inits")
    print("  (init spread is 0.8; spread ~0 = identified, ~0.6 = stayed put)")
    print("=" * 78)
    block(A, y, tr, te, 0.0, "plain regression: predicted lambda -> LOW once identified")
    block(A, y, tr, te, 1.0, "with on-brand penalty: predicted lambda -> HIGH")


if __name__ == "__main__":
    main()
