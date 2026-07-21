"""tmp/rl.unified -- one NNN, one noise: policy + value + exploration + credit (Task #1).

The Step-B loop (tmp/rl/train.py) fit the critic with an EXTERNAL linear TD regression on
detached features.  Here the value readout shares the same NNN body as the actor, and its
hidden-layer credit flows through the SAME forward weight-mirror recursion as the actor's
policy score -- only the top-level signal and the readout differ (idea_rl.md §6, §17.2-6,
§20.15-2).  Nothing external remains: the single forward fluctuation supplies action
samples, exploration, local sensitivity, inter-layer credit (for BOTH heads), and
eligibility; reward modulates the traces.

    actor body credit : recursion from (a - p) through the actor-head mirror   -> trace, x TD
    value body credit : recursion from 1        through the value-head mirror  ->        x TD
    shared body weights receive BOTH; each head gets its own readout grad.
"""
from __future__ import annotations

import numpy as np
import torch

from . import constants  # noqa: F401
from .policy import CartPolePolicy
from .agents import Agent  # noqa: F401  (kept for parity / external comparison)
from .mirror import MirrorState
from . import credit as C
from .train import Hypers, Result, RunningNorm, _clip


def _param(policy, key):
    l, which = key
    if l == "v":
        p = policy.value_head
    else:
        p = policy.fcs[l]
    return p.weight if which == "weight" else p.bias


def train_unified(seed: int = 0, hp: Hypers = None, checkpoint_every: int = 0,
                  value_body_coef: float = 1.0, verbose: bool = True) -> Result:
    """`value_body_coef` scales how strongly the value credit reaches the SHARED body
    (the value head always learns at full strength).  1.0 = fully shared representation;
    smaller values ease the actor/critic tug-of-war on the body without severing the
    unified credit path."""
    import gymnasium as gym
    hp = hp or Hypers()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    policy = CartPolePolicy(obs_dim=len(obs), hidden=hp.hidden, std=hp.std, h=hp.h, t=hp.t)
    policy.field = hp.field
    mirror = MirrorState(beta=hp.mirror_beta, value_head=True)
    n_hidden = len(policy.crossings)

    actor_keys = C.param_keys(policy)                       # body + actor readout
    body_keys = [(l, w) for l in range(n_hidden) for w in ("weight", "bias")]
    elig = {k: torch.zeros_like(_param(policy, k)) for k in actor_keys}

    res = Result(kind="unified")
    ep_return = 0.0
    norm = RunningNorm(len(obs))

    def to_norm(raw, update):
        rt = torch.tensor(raw, dtype=torch.float32)
        if update:
            norm.update(rt)
        return norm.normalize(rt).unsqueeze(0)

    obs_t = to_norm(obs, True)

    for t in range(hp.total_steps):
        step = policy.rollout_step(obs_t)
        mirror.observe(policy, step)
        slope_full, slope_mean = C.slopes(policy, step)

        psi = C.recursion_from_weights(policy, step, mirror.W_out, mirror.W_hidden,
                                       slope_full, slope_mean)          # actor: d log pi/dW
        gval = C.value_grad(policy, step, mirror.W_hidden, mirror.W_vout,
                            slope_full, slope_mean)                     # critic: dV/dW
        V_s = float(step.value.item())

        obs2, r, term, trunc, _ = env.step(int(step.action.item()))
        done = term or trunc
        ep_return += r
        obs2_t = to_norm(obs2, False)
        V_s2 = 0.0 if done else policy.value_of(obs2_t)
        td = float(r) + hp.gamma * V_s2 - V_s

        # actor eligibility (policy gradient trace); value credit is TD(0), no trace
        applied = {}
        for k in actor_keys:
            elig[k] = hp.gamma * hp.lam * elig[k] + psi[k]

        for k in actor_keys + [("v", "weight"), ("v", "bias")]:
            dec = torch.zeros_like(_param(policy, k))
            if k in elig:                                   # actor part (advantage x trace)
                dec = dec + hp.lr_actor * _clip(-td * elig[k], hp.max_grad)
            if k in gval:                                   # value part (semi-gradient TD)
                coef = 1.0 if k[0] == "v" else value_body_coef   # full on head, scaled on body
                dec = dec + coef * hp.lr_critic * _clip(-td * gval[k], hp.max_grad)
            _param(policy, k).data -= dec
            applied[k] = dec

        # Kolen-Pollack: shift every mirror by the known decrement of the weight it mirrors
        shift = {l: applied[(l, "weight")] for l in range(1, n_hidden)}
        shift["out"] = applied[(n_hidden, "weight")]
        shift["vout"] = applied[("v", "weight")]
        mirror.kp_predict(shift)

        if checkpoint_every and (t % checkpoint_every == 0):
            res.checkpoints.append((t, _cpu_state(policy, norm)))

        if done:
            res.ep_returns.append(ep_return)
            res.ep_end_steps.append(t)
            if verbose and len(res.ep_returns) % 25 == 0:
                print(f"  [unified seed{seed}] step {t:6d}  ep {len(res.ep_returns):4d}  "
                      f"return(last25) {np.mean(res.ep_returns[-25:]):6.1f}")
            ep_return = 0.0
            for k in actor_keys:
                elig[k].zero_()
            obs, _ = env.reset()
            obs_t = to_norm(obs, True)
        else:
            obs_t = to_norm(obs2, True)

    if checkpoint_every:
        res.checkpoints.append((hp.total_steps, _cpu_state(policy, norm)))
    res.final_policy = policy
    res.norm_mean = norm.mean.detach().clone()
    res.norm_std = torch.sqrt(norm.M2 / norm.count + norm.eps).detach().clone()
    env.close()
    return res


def _cpu_state(policy, norm):
    std = torch.sqrt(norm.M2 / norm.count + norm.eps)
    return {"net": {k: v.detach().clone() for k, v in policy.state_dict().items()},
            "norm_mean": norm.mean.detach().clone(), "norm_std": std.detach().clone(),
            "hidden": policy.hidden, "t": policy.t}
