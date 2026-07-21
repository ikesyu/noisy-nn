"""rl_stepA_cosine -- Step A of the stage-1 protocol (idea_rl.md §20.6, §20.10).

Measures M1: how well the forward-only cov_jac credit reproduces the autograd
policy-gradient d log pi(a|s)/dW, ONLINE (N=1, one state at a time), with NO learning.
This is the G1 go/no-go gate: if the online weight mirror cannot recover the gradient
direction, the whole forward-covariance RL story is in question (idea_rl.md §17.2-4).

Reported per (H, T):
  cos(true_transpose, gold)  -- sanity: recursion correctness (should be ~1)
  cos(cov_jac, gold)         -- the headline mirror-quality number
  cos(cov_jac, true)         -- isolated mirror error (no slope/convention confound)
  cos(node_pert, gold)       -- the §18-C baseline, for reference

Usage:
    .venv/bin/python tmp/rl_stepA_cosine.py                 # default sweep
    .venv/bin/python tmp/rl_stepA_cosine.py --H 16 32 64 128 256 --T 16 32 64 128
    .venv/bin/python tmp/rl_stepA_cosine.py --states 128 --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))   # make `rl` importable
from rl.policy import CartPolePolicy
from rl import credit as C
from rl import metrics as M
from rl.env import collect_states


def measure(hidden, t, states, seed):
    torch.manual_seed(seed)
    policy = CartPolePolicy(obs_dim=states.shape[1], hidden=hidden, t=t)
    keys = C.param_keys(policy)
    cols = {"true_gold": [], "covjac_gold": [], "covjac_true": [], "node_gold": []}
    for i in range(states.shape[0]):
        obs = states[i:i + 1]                     # [1, 4]  -> ONLINE N=1
        step = policy.rollout_step(obs)
        g_cov = C.cov_jac_grad(policy, step)
        g_true = C.true_transpose_grad(policy, step)
        g_node = C.node_pert_grad(policy, step)
        g_gold = C.gold_grad(policy, step)        # consumes the autograd graph last
        cols["true_gold"].append(M.cosine(g_true, g_gold, keys))
        cols["covjac_gold"].append(M.cosine(g_cov, g_gold, keys))
        cols["covjac_true"].append(M.cosine(g_cov, g_true, keys))
        cols["node_gold"].append(M.cosine(g_node, g_gold, keys))
    return {k: np.nanmedian(v) for k, v in cols.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--T", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--states", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    states = collect_states(n_states=args.states, seed=args.seed)
    print(f"# Step A / M1 -- median per-step cosine over {states.shape[0]} online states "
          f"(N=1), seed={args.seed}")
    print(f"{'H':>5} {'T':>5} | {'true~gold':>10} {'covjac~gold':>12} "
          f"{'covjac~true':>12} {'node~gold':>10}")
    print("-" * 62)
    for hidden in args.H:
        for t in args.T:
            r = measure(hidden, t, states, args.seed)
            print(f"{hidden:>5} {t:>5} | {r['true_gold']:>10.3f} {r['covjac_gold']:>12.3f} "
                  f"{r['covjac_true']:>12.3f} {r['node_gold']:>10.3f}")


if __name__ == "__main__":
    main()
