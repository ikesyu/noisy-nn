"""rl_variance -- Step B / M2-(b): credit-variance vs network width (idea_rl.md §20.6).

At a FIXED state and FIXED weights, the only randomness is the NNN internal noise.  We
draw many independent internal-noise realizations, form each method's estimate of
d log pi(a|s)/dW, and measure its normalized variance

    Var = E|| g - E g ||^2 / || E g ||^2         (inverse SNR^2)

as the hidden width H grows.  cov_jac distributes the output error through the weight
mirror (structured); node perturbation correlates every unit directly with the output
(flat).  The central claim (§18-C, Fiete-Seung O(N)): the flat estimator's variance grows
faster with width, so the cov_jac / node_pert variance gap widens with H.

    .venv/bin/python tmp/rl_variance.py --H 16 32 64 128 256 --samples 200 --reps 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from rl.policy import CartPolePolicy
from rl import credit as C
from rl import metrics as M
from rl.env import collect_states

METHODS = {"cov_jac": C.cov_jac_grad, "node_pert": C.node_pert_grad,
           "true_transpose": C.true_transpose_grad}


def variance_at(hidden, t, obs, seed, samples, layers=2):
    torch.manual_seed(seed)
    policy = CartPolePolicy(obs_dim=obs.shape[1], hidden=hidden, t=t,
                            n_hidden_layers=layers)
    keys = C.param_keys(policy)
    vecs = {m: [] for m in METHODS}
    for _ in range(samples):
        step = policy.rollout_step(obs)          # fresh internal noise each draw
        for m, fn in METHODS.items():
            vecs[m].append(M.flatten(fn(policy, step), keys))
    return {m: M.normalized_variance(v) for m, v in vecs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--reps", type=int, default=4)   # (state, init) repetitions
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, nargs="+", default=[2],
                    help="hidden-layer counts to sweep (depth); the §18-C claim is that "
                         "cov_jac's advantage over flat node perturbation grows with depth")
    args = ap.parse_args()

    states = collect_states(n_states=args.reps, seed=args.seed)
    print(f"# M2-(b) normalized credit variance vs depth/width  (T={args.T}, "
          f"{args.samples} noise draws, {args.reps} reps)")
    print(f"{'layers':>6} {'H':>5} | {'cov_jac':>10} {'node_pert':>10} {'true_tr':>10} | "
          f"{'node/cov':>9}")
    print("-" * 62)
    rows = []
    combos = [(L, H) for L in args.layers for H in args.H]
    for layers, hidden in combos:
        acc = {m: [] for m in METHODS}
        for rep in range(args.reps):
            obs = states[rep:rep + 1]
            v = variance_at(hidden, args.T, obs, args.seed + rep, args.samples, layers)
            for m in METHODS:
                acc[m].append(v[m])
        med = {m: float(np.median(acc[m])) for m in METHODS}
        ratio = med["node_pert"] / max(med["cov_jac"], 1e-12)
        rows.append((layers, hidden, med, ratio))
        print(f"{layers:>6} {hidden:>5} | {med['cov_jac']:>10.3f} {med['node_pert']:>10.3f} "
              f"{med['true_transpose']:>10.3f} | {ratio:>9.2f}")

    OUT = Path(__file__).resolve().parent / "out"
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "rl_variance.npz",
             layers=np.array([r[0] for r in rows]),
             H=np.array([r[1] for r in rows]),
             cov_jac=np.array([r[2]["cov_jac"] for r in rows]),
             node_pert=np.array([r[2]["node_pert"] for r in rows]),
             true_transpose=np.array([r[2]["true_transpose"] for r in rows]))
    print(f"saved {OUT / 'rl_variance.npz'}")


if __name__ == "__main__":
    main()
