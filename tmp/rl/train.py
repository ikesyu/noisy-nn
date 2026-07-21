"""tmp/rl.train -- online forward-fluctuation actor-critic on CartPole (§20.5, §20.11).

One env step at a time (N=1).  The actor credit is the forward-only policy-score
gradient from the chosen agent; it is accumulated into an eligibility trace and
modulated by the TD error.  The critic is a minimal linear TD value head on the NNN's
last-hidden features (so the comparison stays about the actor credit, §20.5).  No
external RL algorithm and no transposed-weight backward for cov_jac.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from . import constants  # noqa: F401
from .policy import CartPolePolicy
from .agents import Agent
from . import credit as C
from data_nce.fncl.train import ManualOpt


@dataclass
class Hypers:
    hidden: int = 64
    t: int = 64
    std: float = 0.6
    h: float = 0.15
    gamma: float = 0.99
    lam: float = 0.9          # actor eligibility persistence
    lam_v: float = 0.0        # critic eligibility persistence (0 = stable TD(0))
    lr_actor: float = 0.02    # SGD; Adam interacts badly with eligibility traces here
    lr_critic: float = 0.05
    opt: str = "sgd"
    mirror_beta: float = 0.99
    total_steps: int = 50000
    max_grad: float = 10.0    # per-key gradient clip for stability
    field: object = None      # per-unit noise field (list of [H] std vectors); None=uniform


class RunningNorm:
    """Online standardization of observations (Welford). CartPole's raw features have
    very different scales; feeding them to fixed-noise crossings saturates units, so we
    normalize before the policy sees them."""

    def __init__(self, dim, clip=5.0, eps=1e-8):
        self.mean = torch.zeros(dim)
        self.M2 = torch.ones(dim)
        self.count = 1e-4
        self.clip = clip
        self.eps = eps

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.M2 += delta * (x - self.mean)

    def normalize(self, x):
        std = torch.sqrt(self.M2 / self.count + self.eps)
        return torch.clamp((x - self.mean) / std, -self.clip, self.clip)


@dataclass
class Result:
    kind: int
    ep_returns: list = field(default_factory=list)      # return per finished episode
    ep_end_steps: list = field(default_factory=list)    # env step at which each ep ended
    checkpoints: list = field(default_factory=list)     # (step, state_dict) snapshots
    final_policy: object = None                          # trained policy (for eval sweeps)
    norm_mean: object = None
    norm_std: object = None


def _clip(g, m):
    n = g.norm()
    return g * (m / n) if (m is not None and n > m) else g


def train(kind: str, seed: int = 0, hp: Hypers = None,
          checkpoint_every: int = 0, env_fn=None, verbose: bool = True) -> Result:
    hp = hp or Hypers()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if env_fn is None:
        import gymnasium as gym
        env = gym.make("CartPole-v1")
    else:
        env = env_fn()
    obs, _ = env.reset(seed=seed)
    policy = CartPolePolicy(obs_dim=len(obs), hidden=hp.hidden, std=hp.std, h=hp.h, t=hp.t)
    policy.field = hp.field
    agent = Agent(policy, kind, mirror_beta=hp.mirror_beta)
    keys = C.param_keys(policy)
    actor_opt = ManualOpt(hp.opt)

    # actor eligibility traces and critic (linear TD value head on last-hidden features)
    elig = {k: torch.zeros_like(_param(policy, k)) for k in keys}
    v_w = torch.zeros(hp.hidden)
    v_b = torch.zeros(())
    e_vw = torch.zeros(hp.hidden)     # critic eligibility (TD(lambda))
    e_vb = torch.zeros(())

    res = Result(kind=kind)
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
        psi = agent.logpi_grad(step)

        phi_s = step.z[-1].mean(dim=1).squeeze(0)            # [H] detached
        V_s = v_w @ phi_s + v_b

        obs2, r, term, trunc, _ = env.step(int(step.action.item()))
        done = term or trunc
        ep_return += r
        obs2_t = to_norm(obs2, False)
        phi_s2 = policy.features(obs2_t).squeeze(0)
        V_s2 = (v_w @ phi_s2 + v_b) if not done else torch.zeros(())
        td = float(r) + hp.gamma * V_s2 - V_s               # scalar tensor

        # actor: e <- gamma*lam*e + psi ;  theta += lr * td * e  (ascent on log pi)
        applied = {}
        for k in keys:
            elig[k] = hp.gamma * hp.lam * elig[k] + psi[k]
            grad = _clip(-td * elig[k], hp.max_grad)          # ManualOpt does param -= lr*grad
            applied[k] = actor_opt.update(str(k), _param(policy, k), grad, hp.lr_actor)
        agent.post_actor_update(applied)

        # critic: semi-gradient TD(lambda)
        e_vw = hp.gamma * hp.lam_v * e_vw + phi_s
        e_vb = hp.gamma * hp.lam_v * e_vb + 1.0
        v_w = v_w + hp.lr_critic * td * e_vw
        v_b = v_b + hp.lr_critic * td * e_vb

        if checkpoint_every and (t % checkpoint_every == 0):
            res.checkpoints.append((t, _cpu_state(policy, v_w, v_b, norm)))

        if done:
            res.ep_returns.append(ep_return)
            res.ep_end_steps.append(t)
            if verbose and len(res.ep_returns) % 25 == 0:
                recent = np.mean(res.ep_returns[-25:])
                print(f"  [{kind} seed{seed}] step {t:6d}  ep {len(res.ep_returns):4d}  "
                      f"return(last25 mean) {recent:6.1f}")
            ep_return = 0.0
            for k in keys:
                elig[k].zero_()
            e_vw.zero_()
            e_vb = torch.zeros(())
            obs, _ = env.reset()
            obs_t = to_norm(obs, True)
        else:
            obs_t = to_norm(obs2, True)

    if checkpoint_every:
        res.checkpoints.append((hp.total_steps, _cpu_state(policy, v_w, v_b, norm)))
    res.final_policy = policy
    res.norm_mean = norm.mean.detach().clone()
    res.norm_std = torch.sqrt(norm.M2 / norm.count + norm.eps).detach().clone()
    env.close()
    return res


def _param(policy, key):
    l, which = key
    fc = policy.fcs[l]
    return fc.weight if which == "weight" else fc.bias


def _cpu_state(policy, v_w, v_b, norm):
    std = torch.sqrt(norm.M2 / norm.count + norm.eps)
    return {
        "net": {k: v.detach().clone() for k, v in policy.state_dict().items()},
        "v_w": v_w.detach().clone(),
        "v_b": v_b.detach().clone(),
        "norm_mean": norm.mean.detach().clone(),
        "norm_std": std.detach().clone(),
        "hidden": policy.hidden,
        "t": policy.t,
    }
