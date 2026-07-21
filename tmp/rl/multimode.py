"""tmp/rl.multimode -- reward learning of two behaviors multiplexed on ONE weight set,
addressed by the noise field (Sub-B core, §7.2 / §14.2; RL version of front_comp L1).

Each episode is assigned a regime r; the policy runs under noise field P_r (a recruitment
field over a disjoint subnetwork).  The regime is hidden from the observation, so the field
is the ONLY thing selecting which target to seek.  The shared weights must learn to drive
x -> target_0 under P_0 and x -> target_1 under P_1.  The actor credit is the forward
weight-mirror (no transposed weights); a minimal linear TD critic on the field-modulated
features supplies the advantage.

The decisive test (`behavior_under_field`): FIX the field and measure where the agent goes,
over episodes of BOTH regimes.  If the endpoint depends only on the field (not the true
regime), the field addresses the behavior on shared weights.
"""
from __future__ import annotations

import numpy as np
import torch

from . import constants  # noqa: F401
from .policy import CartPolePolicy
from . import credit as C
from . import field as F
from .envs_multimode import MultiModeReach
from .train import ManualOpt


def train_multimode(seed=0, H=64, sigma=0.6, episodes=4000, horizon=40,
                    gamma=0.95, lr_actor=0.03, quiet=0.0, fields=None, opt_kind="adam",
                    n_layers=2, verbose=True):
    """Episodic REINFORCE with the forward weight-mirror credit and a per-timestep
    baseline (no critic -- with all-negative dense rewards an unlearned critic gives no
    usable gradient direction).  Advantage A_t = G_t - baseline_t, whitened.
    `fields` overrides the default disjoint recruitment pair (e.g. an overlapping pair).
    `n_layers`=1 makes recruitment gating CLEAN at the readout (a sigma=0 unit gets a
    constant-across-T input, so it is truly dead) -- needed for the lesion test."""
    torch.manual_seed(seed)
    env = MultiModeReach(horizon=horizon, seed=seed)
    policy = CartPolePolicy(obs_dim=env.obs_dim, hidden=H, std=sigma, t=64,
                            n_hidden_layers=n_layers)
    if fields is None:
        fields = [F.recruit(H, sigma, 0, quiet=quiet, n_layers=n_layers),
                  F.recruit(H, sigma, 1, quiet=quiet, n_layers=n_layers)]
    keys = C.param_keys(policy)
    opt = ManualOpt(opt_kind)
    base = np.zeros(horizon)          # per-timestep EMA of return-to-go (baseline)
    base_seen = np.zeros(horizon)

    ret_hist = []
    for ep in range(episodes):
        r = ep % 2 if ep < 200 else int(np.random.randint(2))
        policy.field = fields[r]
        obs = env.reset(r)
        obs_t = torch.tensor(obs).unsqueeze(0)
        psis, rews = [], []
        for _ in range(horizon):
            step = policy.rollout_step(obs_t)
            psis.append(C.cov_jac_grad(policy, step))
            obs2, rew, done = env.step(int(step.action.item()))
            rews.append(rew)
            obs_t = torch.tensor(obs2).unsqueeze(0)
            if done:
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

        for t in range(n):                # update per-timestep baseline
            base_seen[t] += 1
            base[t] += (G[t] - base[t]) / min(base_seen[t], 200)

        if verbose and (ep + 1) % 500 == 0:
            print(f"  [multimode seed{seed}] ep {ep+1:5d}  return(last200) "
                  f"{np.mean(ret_hist[-200:]):7.2f}")
    return policy, fields, env, ret_hist


def behavior_under_field(policy, fields, env, field_idx, n_ep=40, horizon=40):
    """Endpoint x when the field is FIXED to fields[field_idx], averaged over episodes of
    BOTH true regimes (which are hidden).  Depends only on the field if addressing works."""
    policy.field = fields[field_idx]
    ends = []
    for e in range(n_ep):
        env.reset(e % 2)                       # regime varies but is hidden / irrelevant
        obs_t = torch.tensor(env._obs()).unsqueeze(0)
        for _ in range(horizon):
            step = policy.rollout_step(obs_t, greedy=True)
            obs2, _, done = env.step(int(step.action.item()))
            obs_t = torch.tensor(obs2).unsqueeze(0)
            if done:
                break
        ends.append(env.x)
    return float(np.mean(ends)), float(np.std(ends))


def _p(policy, key):
    l, which = key
    fc = policy.fcs[l]
    return fc.weight if which == "weight" else fc.bias
