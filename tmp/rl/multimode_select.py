"""tmp/rl.multimode_select -- reward-driven AUTONOMOUS field selection (§7.2, §9; Task #2).

Sub-B showed that a GIVEN field addresses a behavior on shared weights.  Here the field is
no longer handed out per regime: a learned selector picks which noise-field prototype to
use from a context cue, and reward trains the selector AND the behavior body jointly.  The
prototypes carry NO pre-assigned meaning (§7.2); the system must self-organize a consistent
context -> field -> behavior mapping.

  selector : softmax preferences theta[context, mode]; sample a mode per episode (REINFORCE)
  body     : CartPolePolicy under the selected field; forward weight-mirror credit
             (REINFORCE + per-timestep baseline), sees only x (not the context)

Success = for each context the composed (context -> selected field -> body behavior) reaches
that context's target, with the two contexts routed to DIFFERENT fields.
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
from .multimode import _p, behavior_under_field  # noqa: F401


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def train_select(seed=0, H=64, sigma=0.6, episodes=5000, horizon=40, gamma=0.95,
                 lr_actor=0.05, lr_sel=0.1, quiet=0.0, K=2, n_ctx=2, opt_kind="sgd",
                 verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = MultiModeReach(horizon=horizon, seed=seed)
    policy = CartPolePolicy(obs_dim=env.obs_dim, hidden=H, std=sigma, t=64)
    fields = [F.recruit(H, sigma, k, quiet=quiet) for k in range(K)]
    keys = C.param_keys(policy)
    opt = ManualOpt(opt_kind)

    base = np.zeros(horizon)           # body per-timestep baseline
    base_seen = np.zeros(horizon)
    theta = np.zeros((n_ctx, K))       # selector preferences
    base_sel = np.zeros(n_ctx)         # selector per-context baseline
    ret_std = 1.0

    ret_hist, sel_hist = [], []
    for ep in range(episodes):
        ctx = ep % n_ctx if ep < 200 else int(rng.integers(n_ctx))
        pi = _softmax(theta[ctx])
        o = int(rng.choice(K, p=pi))            # selector picks a field (mode)
        policy.field = fields[o]

        obs = env.reset(ctx)
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
        ep_ret = float(sum(rews))
        ret_hist.append(ep_ret)
        sel_hist.append((ctx, o))
        ret_std = 0.99 * ret_std + 0.01 * (abs(ep_ret - base_sel[ctx]) + 1e-6)

        # --- body update (forward-mirror REINFORCE + per-timestep baseline) ---
        adv = (G - base[:n])
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
            base[t] += (G[t] - base[t]) / min(base_seen[t], 200)

        # --- selector update (softmax REINFORCE over modes) ---
        adv_sel = (ep_ret - base_sel[ctx]) / ret_std
        for k in range(K):
            theta[ctx, k] += lr_sel * adv_sel * ((1.0 if k == o else 0.0) - pi[k])
        base_sel[ctx] += 0.02 * (ep_ret - base_sel[ctx])

        if verbose and (ep + 1) % 1000 == 0:
            routes = " ".join(f"c{c}->P{int(theta[c].argmax())}" for c in range(n_ctx))
            print(f"  [select seed{seed}] ep {ep+1:5d}  return(last200) "
                  f"{np.mean(ret_hist[-200:]):7.2f}  routes: {routes}")
    return policy, fields, env, theta, ret_hist


def composed_behavior(policy, fields, env, theta, ctx, n_ep=40, horizon=40):
    """Endpoint x when the SELECTOR (argmax) chooses the field for `ctx`."""
    o = int(theta[ctx].argmax())
    m, s = behavior_under_field(policy, fields, env, o, n_ep=n_ep, horizon=horizon)
    return o, m, s
