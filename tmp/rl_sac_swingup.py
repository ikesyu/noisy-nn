"""rl_sac_swingup -- feasibility test: SAC-style off-policy NNN actor-critic (§23.11).

Does the fully-NNN system (cov_jac actor + twin NNN Q-critics, EMA mirrors everywhere,
no backprop) learn swing-up + balance under an OFF-POLICY, replay-based, max-entropy
regime?  Applies the §23.10 prescriptions (i)-(v); see rl/sac.py for the design.

    .venv/bin/python tmp/rl_sac_swingup.py [--episodes 300] [--seed 0]
Output: tmp/out/swingup_sac_s0.pt, eval table on stdout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from rl.sac import train_sac_nnn
from rl.a2c_swingup import eval_from_bottom

OUT = Path(__file__).resolve().parent / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--lr-actor", type=float, default=0.003)
    ap.add_argument("--actor-mode", choices=["fresh", "replay"], default="fresh")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    policy, qs, norm, cks, hist = train_sac_nnn(
        seed=args.seed, episodes=args.episodes, horizon=args.horizon,
        rounds=args.rounds, alpha=args.alpha, lr_actor=args.lr_actor,
        actor_mode=args.actor_mode, verbose=True)
    torch.save({"checkpoints": cks, "hist": hist, "seed": args.seed},
               OUT / f"swingup_sac_s{args.seed}.pt")

    print("=== eval swing-up from BOTTOM (greedy, horizon 500) ===")
    tails = []
    for epi, st in cks:
        mc, fu, tail = eval_from_bottom(st, horizon=500)
        tails.append(tail)
        print(f"  epi {epi:4d}  mean cos {mc:+.3f}  frac_up {fu:.3f}  last100_up {tail:.3f}")
    if not tails:
        print("(no checkpoints)")
        return
    tails = np.array(tails)
    late = tails[len(tails) // 2:]
    print(f"best last100_up {tails.max():.3f}   late-half mean {late.mean():.3f}  "
          f"(PPO v4: best 1.000, late-half 1.000)")


if __name__ == "__main__":
    main()
