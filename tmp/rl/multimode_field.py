"""tmp/rl.multimode_field -- the field itself as a CONTINUOUS latent action moved by reward
(§7.3, §19; the 本丸 minimal version).

Instead of choosing among fixed prototypes, the noise field is generated from a continuous
centre c in [0,1] (a Gaussian recruitment bump, `field.bump`).  A field-level policy holds
a learnable centre mu_c[context]; each episode samples c = mu_c[ctx] + noise, runs the body
under bump(c), and reward moves BOTH the field centre (a field-level policy gradient, §19.3)
and the behavior body (forward-mirror REINFORCE).  The prototypes carry no meaning; reward
must discover distinct centres and the matching behaviors.

Distinctive test (§7.3, §14.2): after learning, sweep c continuously between the two learnt
centres and watch the endpoint move smoothly through UNTRAINED intermediate behaviors --
the field is a continuous option embedding, not two discrete switches.
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
from .multimode import _p


def train_field(seed=0, H=64, sigma=0.6, episodes=5000, horizon=40, gamma=0.95,
                lr_actor=0.05, lr_field=0.1, xi_std=0.12, tau=0.15, n_ctx=2,
                init_spread=0.0, verbose=True):
    """`init_spread`=0 starts both context centres at 0.5 (reward must break symmetry);
    >0 seeds them apart by that amount."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = MultiModeReach(horizon=horizon, seed=seed)
    policy = CartPolePolicy(obs_dim=env.obs_dim, hidden=H, std=sigma, t=64, n_hidden_layers=1)
    keys = C.param_keys(policy)
    opt = ManualOpt("sgd")

    base = np.zeros(horizon)
    base_seen = np.zeros(horizon)
    mu_c = np.full(n_ctx, 0.5) + init_spread * (np.arange(n_ctx) - (n_ctx - 1) / 2)
    base_f = np.zeros(n_ctx)
    ret_std = 1.0

    ret_hist, muc_hist = [], []
    for ep in range(episodes):
        ctx = ep % n_ctx if ep < 200 else int(rng.integers(n_ctx))
        c = float(np.clip(mu_c[ctx] + rng.normal(0, xi_std), 0.0, 1.0))
        policy.field = F.bump(H, c, sigma, tau=tau)

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
        ret_std = 0.99 * ret_std + 0.01 * (abs(ep_ret - base_f[ctx]) + 1e-6)

        # body update (forward-mirror REINFORCE + per-timestep baseline)
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
            base[t] += (G[t] - base[t]) / min(base_seen[t], 200)

        # field-centre update (move the centre toward c when the advantage is positive)
        adv_f = (ep_ret - base_f[ctx]) / ret_std
        mu_c[ctx] = float(np.clip(mu_c[ctx] + lr_field * adv_f * (c - mu_c[ctx]), 0.0, 1.0))
        base_f[ctx] += 0.02 * (ep_ret - base_f[ctx])

        muc_hist.append(mu_c.copy())
        if verbose and (ep + 1) % 1000 == 0:
            print(f"  [field seed{seed}] ep {ep+1:5d}  return(last200) "
                  f"{np.mean(ret_hist[-200:]):7.2f}  mu_c={np.round(mu_c,3)}")
    return policy, env, mu_c, np.array(muc_hist), ret_hist


def endpoint_at_center(policy, env, center, sigma, tau, n_ep=30, horizon=40, greedy=True):
    """Mean endpoint when the field is bump(center), over episodes of both regimes."""
    policy.field = F.bump(policy.hidden, center, sigma, tau=tau)
    ends = []
    for e in range(n_ep):
        env.reset(e % 2)
        obs_t = torch.tensor(env._obs()).unsqueeze(0)
        for _ in range(horizon):
            step = policy.rollout_step(obs_t, greedy=greedy)
            obs2, _, done = env.step(int(step.action.item()))
            obs_t = torch.tensor(obs2).unsqueeze(0)
            if done:
                break
        ends.append(env.x)
    return float(np.mean(ends)), float(np.std(ends))
