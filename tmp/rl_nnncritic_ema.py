"""rl_nnncritic_ema -- fix 1 alone for the §23.5 bottleneck (idea_rl.md §23.9).

The §23.5 fully-NNN actor-critic (cov_jac for BOTH, learned critic features) peaked at
last100_up ~ 0.44.  §23.8 later showed the per-step single-shot mirror is a bottleneck of
its own, but fixed it only together with a frozen-feature critic.  Here we test the
MINIMAL change that keeps the critic's features LEARNED: give both actor and critic a
persistent EMA mirror + Kolen-Pollack tracking (credit.MirrorEMA) and change nothing else.

    .venv/bin/python tmp/rl_nnncritic_ema.py --mirror-beta 0.1   # fix 1
    .venv/bin/python tmp/rl_nnncritic_ema.py --mirror-beta 0     # §23.5 replica (control)
Output: tmp/out/swingup_nnnac_ema.pt (or _rep.pt), eval table on stdout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent))
from rl.a2c_nnncritic import train_a2c_nnn
from rl.a2c_swingup import eval_from_bottom

OUT = Path(__file__).resolve().parent / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--mirror-beta", type=float, default=0.1)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    policy, critic, norm, cks, hist = train_a2c_nnn(
        seed=args.seed, updates=args.updates, horizon=args.horizon,
        mirror_beta=args.mirror_beta or None, verbose=True)
    tag = "_ema" if args.mirror_beta else "_rep"
    torch.save({"checkpoints": cks, "hist": hist, "mirror_beta": args.mirror_beta},
               OUT / f"swingup_nnnac{tag}.pt")

    print("=== eval swing-up from BOTTOM (greedy, horizon 500) ===")
    best = (0, -1.0)
    for upd, st in cks:
        mc, fu, tail = eval_from_bottom(st, horizon=500)
        if tail > best[1]:
            best = (upd, tail)
        print(f"  upd {upd:4d}  mean cos {mc:+.3f}  frac_up {fu:.3f}  last100_up {tail:.3f}")
    print(f"best last100_up {best[1]:.3f} at upd {best[0]}  "
          f"(baselines: §23.5 single-shot ~0.44, §23.8 frozen+EMA 1.00)")


if __name__ == "__main__":
    main()
