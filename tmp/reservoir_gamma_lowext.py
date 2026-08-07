"""Extension: the crossing's best gamma sat at the LOWER EDGE of the grid.

reservoir_gamma_recheck.py found, contrary to sec10.24(3)/sec10.27, that the
crossing (bump) does NOT collapse for depth >= 2 -- it just needs a LOWER
effective gain: depth2 0.47 and depth3 0.44 at gamma=0.5, the smallest gamma
tested, versus the ~1.17 "collapse" reported earlier at gamma ~ 1 (which is what
BatchNorm(affine init 1) gives).  So the earlier "deep collapse" looks like an
over-sharpening artefact, and the true optimum is off the bottom of that grid.

Here: extend gamma downward (0.15 .. 0.7) for the crossing, and re-check the
monotone activations at the same low gammas so the comparison stays symmetric.
Also probes whether gamma* keeps DECREASING with depth (composition of
non-monotone bumps amplifies, so each layer must be gentler).
"""
import numpy as np

from reservoir_gamma_recheck import (SignField, parity_task, train_eval, masks)

LOW = (0.15, 0.2, 0.3, 0.4, 0.5, 0.7)
ACTS = ("crossing", "threshold", "tanh")


def main():
    T = 3000; u, y = parity_task(T); tr, te = masks(T)
    A = SignField(H=32, gain=8.0).run(u)
    SEEDS = 3
    print("=" * 74)
    print("LOW-gamma extension (3 seeds).  Does the crossing's optimum keep")
    print("dropping with depth, and how low does the NRMSE actually go?")
    print("=" * 74)
    best = {}
    for depth in (1, 2, 3, 4):
        print(f"\n--- depth {depth} ---")
        print("  act        " + "  ".join(f"g={g:<5g}" for g in LOW) + " | min_g  argmin")
        for act in ACTS:
            row = [np.mean([train_eval(A, y, tr, te, depth, act, g, seed=s)
                            for s in range(SEEDS)]) for g in LOW]
            i = int(np.argmin(row)); best[(act, depth)] = (row[i], LOW[i])
            print(f"  {act:<10s} " + "  ".join(f"{v:5.2f}  " for v in row)
                  + f"| {row[i]:5.2f}  g={LOW[i]:g}", flush=True)

    print("\n" + "=" * 74)
    print("gamma* vs depth (does the crossing need a gentler gain when deeper?)")
    print("  depth |  crossing        |  threshold       |  tanh")
    for depth in (1, 2, 3, 4):
        s = f"  {depth:5d} |"
        for act in ACTS:
            v, g = best[(act, depth)]
            s += f"  {v:.2f} @ g={g:<4g} |"
        print(s)
    print("\nVERDICT on 'the crossing collapses for depth >= 2':")
    for depth in (2, 3, 4):
        v, g = best[("crossing", depth)]
        print(f"  depth{depth}: best {v:.2f} at gamma={g:g}  -> "
              f"{'COLLAPSED' if v > 0.9 else 'HEALTHY (claim refuted)'}")


if __name__ == "__main__":
    main()
