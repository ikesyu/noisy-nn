"""tmp/rl.swingup -- episodic REINFORCE swing-up trainer (forward-mirror credit).

The online one-step actor-critic collapsed to a constant push on swing-up.  Here we use
episodic REINFORCE with a per-timestep baseline (the same recipe that cleanly solved the
multi-mode reach, §21.2): Monte-Carlo return-to-go propagates the swing-up payoff back to
the pumping actions, and there is no critic to collapse.  Curriculum: a fraction of
episodes start near the BOTTOM (to practice the full swing-up), the rest at random angles.
Actor credit is the forward weight-mirror (cov_jac, no transposed weights).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import constants  # noqa: F401
from .policy import CartPolePolicy
from . import credit as C
from .envs_swingup import CartPoleSwingUp
from .train import ManualOpt, RunningNorm
from .multimode import _p


def train_swingup(seed=0, H=128, sigma=0.6, episodes=1500, horizon=300, gamma=0.99,
                  lr_actor=0.03, bottom_frac=0.5, force_mag=20.0, x_threshold=4.0,
                  checkpoint_every=100, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = CartPoleSwingUp(horizon=horizon, seed=seed, force_mag=force_mag,
                          x_threshold=x_threshold)
    policy = CartPolePolicy(obs_dim=5, hidden=H, std=sigma, t=64)
    keys = C.param_keys(policy)
    opt = ManualOpt("adam")
    norm = RunningNorm(5)
    base = np.zeros(horizon)
    base_seen = np.zeros(horizon)

    ret_hist, checkpoints = [], []
    for ep in range(episodes):
        # curriculum: half the episodes start near the bottom (full swing-up practice)
        if rng.random() < bottom_frac:
            start = math.pi + float(rng.uniform(-0.5, 0.5))
        else:
            start = float(rng.uniform(-math.pi, math.pi))
        obs, _ = env.reset(seed=None, start_theta=start)
        obs_t = _norm(norm, obs, update=True)
        psis, rews = [], []
        for _ in range(horizon):
            step = policy.rollout_step(obs_t)
            psis.append(C.cov_jac_grad(policy, step))
            obs, rew, te, tr, _ = env.step(int(step.action.item()))
            rews.append(rew)
            obs_t = _norm(norm, obs, update=True)
            if te or tr:
                break
        n = len(rews)
        G = np.zeros(n)
        acc = 0.0
        for t in range(n - 1, -1, -1):
            acc = rews[t] + gamma * acc
            G[t] = acc
        ret_hist.append(float(sum(rews)))

        adv = G - base[:n]
        std = adv.std() + 1e-6
        grad = {k: torch.zeros_like(_p(policy, k)) for k in keys}
        for t in range(n):
            a_t = float(adv[t] / std)
            for k in keys:
                grad[k] += a_t * psis[t][k]
        for k in keys:
            opt.update(str(k), _p(policy, k), -grad[k] / n, lr_actor)
        for t in range(n):
            base_seen[t] += 1
            base[t] += (G[t] - base[t]) / min(base_seen[t], 300)

        if checkpoint_every and (ep + 1) % checkpoint_every == 0:
            checkpoints.append((ep + 1, _snapshot(policy, norm, H)))
        if verbose and (ep + 1) % 100 == 0:
            print(f"  [swingup seed{seed}] ep {ep+1:5d}  return(last50) "
                  f"{np.mean(ret_hist[-50:]):8.1f}")
    return policy, env, norm, checkpoints, ret_hist


def _norm(norm, obs, update):
    rt = torch.tensor(obs, dtype=torch.float32)
    if update:
        norm.update(rt)
    return norm.normalize(rt).unsqueeze(0)


def _snapshot(policy, norm, H):
    std = torch.sqrt(norm.M2 / norm.count + norm.eps)
    return {"net": {k: v.detach().clone() for k, v in policy.state_dict().items()},
            "norm_mean": norm.mean.detach().clone(), "norm_std": std.detach().clone(),
            "hidden": H, "t": policy.t}


def eval_from_bottom(state, force_mag=20.0, x_threshold=4.0, horizon=400, seeds=(0, 1, 2)):
    p = CartPolePolicy(obs_dim=5, hidden=int(state["hidden"]), t=int(state["t"]))
    p.load_state_dict(state["net"]); p.eval()
    mean, std = state["norm_mean"], state["norm_std"]
    out = []
    for s in seeds:
        env = CartPoleSwingUp(horizon=horizon, random_start=False, seed=s,
                              force_mag=force_mag, x_threshold=x_threshold)
        obs, _ = env.reset(seed=s)
        cs = []
        for _ in range(horizon):
            x = torch.clamp((torch.tensor(obs, dtype=torch.float32) - mean) / std, -5, 5)
            a = int(p.rollout_step(x.unsqueeze(0), greedy=True).action.item())
            obs, r, te, tr, _ = env.step(a)
            cs.append(env.cos_theta())
        out.append((np.mean(cs), (np.array(cs) > 0.9).mean()))
    return float(np.mean([m for m, _ in out])), float(np.mean([f for _, f in out]))
