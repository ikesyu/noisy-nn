"""rl_field_prototypes -- noise-field Sub-A: does a NON-UNIFORM field beat uniform sigma?

The SR sweep (idea_rl.md §20.16) found a uniform-sigma tension: credit fidelity (mirror
cosine) prefers low sigma, control (return) prefers mid/high sigma.  Both the SR result
and the critic-unification result pointed to the SAME resolution: allocate noise
SPATIALLY.  Here we test the prerequisite for the field/option program directly -- train
CartPole under several FIXED per-unit fields and measure both:

    greedy return        (control)
    mirror cosine        (credit fidelity on the trained weights)

If a spatial field ('split'/'graded') reaches high return while keeping higher cosine than
the uniform field of matching return, spatial allocation escapes the tension -> the
field/option mechanism is worth building (Sub-B: reward-selected prototypes).

    .venv/bin/python tmp/rl_field_prototypes.py --seeds 0 1 --steps 30000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from rl.train import train, Hypers
from rl import field as F
from rl import credit as C
from rl import metrics as M
from rl.env import collect_states


def mirror_cosine(policy, states_norm, keys, cap=96):
    c = []
    for i in range(min(cap, states_norm.shape[0])):
        step = policy.rollout_step(states_norm[i:i + 1])
        c.append(M.cosine(C.cov_jac_grad(policy, step), C.gold_grad(policy, step), keys))
    return float(np.nanmedian(c))


def greedy_return(policy, mean, std, n_eval=6, max_len=500):
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    rets = []
    for e in range(n_eval):
        obs, _ = env.reset(seed=3000 + e)
        ret = 0.0
        for _ in range(max_len):
            x = torch.clamp((torch.tensor(obs, dtype=torch.float32) - mean) / std, -5, 5)
            step = policy.rollout_step(x.unsqueeze(0), greedy=True)
            obs, r, term, trunc, _ = env.step(int(step.action.item()))
            ret += r
            if term or trunc:
                break
        rets.append(ret)
    env.close()
    return float(np.mean(rets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--T", type=int, default=64)
    args = ap.parse_args()
    OUT = Path(__file__).resolve().parent / "out"
    OUT.mkdir(exist_ok=True)

    protos = F.prototypes(args.H)
    raw_states = collect_states(n_states=96, seed=99)
    print(f"# Field Sub-A: return & credit fidelity per fixed field "
          f"(H={args.H}, T={args.T}, {args.steps} steps, seeds={args.seeds})")
    print(f"{'field':>12} | {'return':>8} {'cosine':>8}")
    print("-" * 34)
    rows = {}
    for name, fld in protos.items():
        rets, coss = [], []
        for seed in args.seeds:
            hp = Hypers(hidden=args.H, t=args.T, total_steps=args.steps, field=fld)
            res = train("cov_jac", seed=seed, hp=hp, verbose=False)
            policy = res.final_policy
            keys = C.param_keys(policy)
            sn = torch.clamp((raw_states - res.norm_mean) / res.norm_std, -5, 5)
            coss.append(mirror_cosine(policy, sn, keys))
            rets.append(greedy_return(policy, res.norm_mean, res.norm_std))
        rows[name] = (np.mean(rets), np.mean(coss))
        print(f"{name:>12} | {np.mean(rets):>8.1f} {np.mean(coss):>8.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.5, 5))
    for name, (ret, cos) in rows.items():
        marker = "o" if name.startswith("uniform") else "*"
        plt.scatter(cos, ret, s=160 if marker == "*" else 90, marker=marker, label=name)
    plt.xlabel("mirror cosine (credit fidelity)")
    plt.ylabel("greedy return (control)")
    plt.title("Field Sub-A: can a spatial field escape the uniform tradeoff?\n"
              "(uniform = circles trace the tension; spatial = stars want upper-right)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "rl_field_subA.png", dpi=130)
    np.savez(OUT / "rl_field_subA.npz", **{k: np.array(v) for k, v in rows.items()})
    print(f"saved {OUT / 'rl_field_subA.png'}")


if __name__ == "__main__":
    main()
