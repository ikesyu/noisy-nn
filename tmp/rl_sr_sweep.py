"""rl_sr_sweep -- Step C: does ONE noise level serve computation + exploration + control?

The natural-integration core (idea_rl.md §17.2-3, §20.8, §20.15): in an NNN the noise the
network NEEDS to compute is the SAME noise that explores and that makes the forward credit
estimable.  If a single noise strength sigma jointly optimizes

  (a) computation / credit  -- the forward mirror recovers the gradient (cosine to autograd),
  (b) exploration           -- the policy actually varies its actions,
  (c) control               -- achieved return,

then RL is not bolted onto NNN; it falls out of how NNN computes.  Both credit and
exploration DIE as sigma -> 0 (no noise -> identical samples -> the covariance mirror is
undefined and the policy is deterministic), so an interior optimum is expected -- a
stochastic-resonance curve for RL.

Note: the covariance credit ONLY exists in the sample (mechanism) regime; the analytic
mean-field crossing has no T samples and cannot form the mirror at all -- so this sweep is
intrinsically about the sample mechanism (the point front_comp makes for SR).

    static (fast, no learning):
        .venv/bin/python tmp/rl_sr_sweep.py --mode static
    train (headline SR-for-RL curve; slow, run in background):
        .venv/bin/python tmp/rl_sr_sweep.py --mode train --steps 30000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from rl.policy import CartPolePolicy
from rl.train import train, Hypers
from rl import credit as C
from rl import metrics as M
from rl.env import collect_states

OUT = Path(__file__).resolve().parent / "out"


def mirror_cosine(policy, states_norm, keys, cap_states=96):
    c9 = []
    for i in range(min(cap_states, states_norm.shape[0])):
        step = policy.rollout_step(states_norm[i:i + 1])
        g_cov = C.cov_jac_grad(policy, step)
        g_gold = C.gold_grad(policy, step)
        c9.append(M.cosine(g_cov, g_gold, keys))
    return float(np.nanmedian(c9))


def logit_spread(policy, states_norm, cap_states=96):
    """Mean over states of std_T(o^(m)): the per-sample logit disagreement = raw
    exploration/credit signal magnitude injected by the noise."""
    sp = []
    for i in range(min(cap_states, states_norm.shape[0])):
        step = policy.rollout_step(states_norm[i:i + 1])
        sp.append(float(step.y_samples.squeeze(-1).std(dim=1).mean()))
    return float(np.mean(sp))


def eval_policy(policy, mean, std, n_eval=6, max_len=500):
    """Greedy return and stochastic-policy action entropy."""
    import gymnasium as gym
    env = gym.make("CartPole-v1")

    def norm(o):
        return torch.clamp((torch.tensor(o, dtype=torch.float32) - mean) / std, -5, 5)

    rets, ents = [], []
    for e in range(n_eval):
        obs, _ = env.reset(seed=2000 + e)
        ret, ent_ep = 0.0, []
        for _ in range(max_len):
            step = policy.rollout_step(norm(obs).unsqueeze(0), greedy=True)
            p = float(step.p.item())
            p = min(max(p, 1e-6), 1 - 1e-6)
            ent_ep.append(-(p * np.log(p) + (1 - p) * np.log(1 - p)))
            obs, r, term, trunc, _ = env.step(int(step.action.item()))
            ret += r
            if term or trunc:
                break
        rets.append(ret)
        ents.append(float(np.mean(ent_ep)))
    env.close()
    return float(np.mean(rets)), float(np.mean(ents))


def run_static(sigmas, H, T, seed):
    states = collect_states(n_states=96, seed=seed)
    sn = (states - states.mean(0)) / (states.std(0) + 1e-6)     # standardize
    print(f"# SR static: mirror cosine & logit spread vs sigma (H={H}, T={T}, no learning)")
    print(f"{'sigma':>6} | {'cosine':>8} {'logit_spread':>13}")
    print("-" * 34)
    rows = []
    for sg in sigmas:
        torch.manual_seed(seed)
        policy = CartPolePolicy(obs_dim=states.shape[1], hidden=H, t=T, std=sg)
        keys = C.param_keys(policy)
        cos = mirror_cosine(policy, sn, keys)
        spr = logit_spread(policy, sn)
        rows.append((sg, cos, spr))
        print(f"{sg:>6.2f} | {cos:>8.3f} {spr:>13.4f}")
    plot_static(rows)
    return rows


def run_train(sigmas, H, T, steps, seed):
    print(f"# SR train: return, mirror cosine, entropy vs sigma (H={H}, T={T}, "
          f"{steps} steps/point)")
    print(f"{'sigma':>6} | {'return':>8} {'cosine':>8} {'entropy':>8}")
    print("-" * 38)
    rows = []
    raw_states = collect_states(n_states=96, seed=seed + 7)
    for sg in sigmas:
        hp = Hypers(hidden=H, t=T, std=sg, total_steps=steps)
        res = train("cov_jac", seed=seed, hp=hp, verbose=False)
        policy = res.final_policy
        keys = C.param_keys(policy)
        sn = torch.clamp((raw_states - res.norm_mean) / res.norm_std, -5, 5)
        cos = mirror_cosine(policy, sn, keys)
        ret, ent = eval_policy(policy, res.norm_mean, res.norm_std)
        rows.append((sg, ret, cos, ent))
        print(f"{sg:>6.2f} | {ret:>8.1f} {cos:>8.3f} {ent:>8.3f}")
    plot_train(rows)
    return rows


def plot_static(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = [r[0] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.plot(s, [r[1] for r in rows], "o-", color="C0", label="mirror cosine (computation)")
    ax1.set_xlabel("noise strength sigma"); ax1.set_ylabel("mirror cosine", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(s, [r[2] for r in rows], "s--", color="C1", label="logit spread (exploration)")
    ax2.set_ylabel("per-sample logit spread", color="C1")
    ax1.set_title("SR static: shared sigma-dependence of credit and exploration")
    fig.tight_layout(); fig.savefig(OUT / "rl_sr_static.png", dpi=130)
    print(f"saved {OUT / 'rl_sr_static.png'}")


def plot_train(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = [r[0] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.plot(s, [r[1] for r in rows], "o-", color="C2", label="greedy return")
    ax1.set_xlabel("noise strength sigma (fixed during training)")
    ax1.set_ylabel("greedy return", color="C2")
    ax2 = ax1.twinx()
    ax2.plot(s, [r[2] for r in rows], "s--", color="C0", label="mirror cosine")
    ax2.plot(s, [r[3] for r in rows], "^:", color="C3", label="action entropy")
    ax2.set_ylabel("cosine / entropy")
    ax2.legend(loc="upper right", fontsize=8)
    ax1.set_title("SR-for-RL: does one sigma serve computation + exploration + control?")
    fig.tight_layout(); fig.savefig(OUT / "rl_sr_train.png", dpi=130)
    print(f"saved {OUT / 'rl_sr_train.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static", "train", "both"], default="static")
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.45, 0.6, 0.9, 1.3])
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if args.mode in ("static", "both"):
        rows = run_static(args.sigmas, args.H, args.T, args.seed)
        np.savez(OUT / "rl_sr_static.npz", data=np.array(rows))
    if args.mode in ("train", "both"):
        rows = run_train(args.sigmas, args.H, args.T, args.steps, args.seed)
        np.savez(OUT / "rl_sr_train.npz", data=np.array(rows))


if __name__ == "__main__":
    main()
