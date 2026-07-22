"""rl_ppo_swingup -- PPO on the fully-NNN actor-critic (idea_rl.md §23.10).

Goal: remove the oscillation / best-checkpoint dependence of the A2C runs (§23.9
knowledge 4) with the clipped surrogate + epoch reuse, while keeping the whole system
backprop-free (cov_jac actor + learned NNN critic, both with persistent EMA mirrors).

    .venv/bin/python tmp/rl_ppo_swingup.py [--updates 300] [--seed 0]
Output: tmp/out/swingup_ppo.pt, eval table on stdout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from rl.ppo import train_ppo_nnn
from rl.a2c_swingup import eval_from_bottom

OUT = Path(__file__).resolve().parent / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--sigma-end", type=float, default=0.1)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    policy, critic, norm, cks, hist = train_ppo_nnn(
        seed=args.seed, updates=args.updates, horizon=args.horizon,
        ppo_epochs=args.ppo_epochs, clip_eps=args.clip_eps,
        sigma_explore_end=args.sigma_end, verbose=True)
    torch.save({"checkpoints": cks, "hist": hist, "seed": args.seed},
               OUT / f"swingup_ppo_s{args.seed}.pt")

    print("=== eval swing-up from BOTTOM (greedy, horizon 500) ===")
    tails = []
    for upd, st in cks:
        mc, fu, tail = eval_from_bottom(st, horizon=500)
        tails.append(tail)
        print(f"  upd {upd:4d}  mean cos {mc:+.3f}  frac_up {fu:.3f}  last100_up {tail:.3f}")
    if not tails:
        print("(no checkpoints -- updates < checkpoint_every)")
        return
    tails = np.array(tails)
    late = tails[len(tails) // 2:]
    print(f"best last100_up {tails.max():.3f}   "
          f"late-half mean {late.mean():.3f}  (A2C §23.9: best 1.000 but late-half ~0.4)")


if __name__ == "__main__":
    main()
