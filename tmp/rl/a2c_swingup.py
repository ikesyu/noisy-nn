"""tmp/rl.a2c_swingup -- 方向1: NNN cov_jac continuous-force actor + external GAE critic.

Actor  : ContinuousNNNPolicy, policy gradient from the forward mirror (cov_jac) -- the NNN
         contribution, no transposed-weight backprop.
Critic : external ValueMLP trained by backprop on GAE returns (the integration compromise).
Advantage: GAE(gamma, lambda); advantages are normalized per batch.  Curriculum: a fraction
of episodes start near the bottom (full swing-up practice).  Exploration is intrinsic (the
NNN sample spread), so no entropy bonus is needed.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import constants  # noqa: F401
from .policy_cont import ContinuousNNNPolicy
from .critic import ValueMLP
from . import credit as C
from .envs_swingup import CartPoleSwingUp
from .train import ManualOpt, RunningNorm
from .multimode import _p


def _norm(norm, obs, update):
    rt = torch.tensor(obs, dtype=torch.float32)
    if update:
        norm.update(rt)
    return norm.normalize(rt)


def _set_field(policy, fields, cos_theta, gate_k, gate_c):
    """Noise-field OPTION gate (方向3a): blend P_pump (fields[0]) and P_balance (fields[1])
    by proximity to upright.  g -> 1 near the top (balance mode), g -> 0 far (pump mode);
    the two modes are addressed on the SAME shared weights by the field."""
    g = 1.0 / (1.0 + math.exp(-gate_k * (cos_theta - gate_c)))
    policy.field = [(1.0 - g) * fields[0][l] + g * fields[1][l]
                    for l in range(len(fields[0]))]


def train_a2c(seed=0, H=128, sigma=0.6, updates=400, episodes_per_update=3, horizon=400,
              gamma=0.99, lam=0.95, lr_actor=0.01, lr_critic=1e-3, critic_epochs=8,
              bottom_frac=0.5, force_mag=20.0, x_threshold=4.0,
              sigma_explore=0.4, sigma_explore_end=0.1,
              fields=None, gate_k=6.0, gate_c=0.0,
              fixed_field=None, start_center=None, start_range=0.5,
              init_body=None, freeze_mask=None, n_hidden_layers=2, energy_reward=True,
              norm_obj=None, update_norm=True,
              checkpoint_every=25, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = CartPoleSwingUp(horizon=horizon, seed=seed, force_mag=force_mag,
                          x_threshold=x_threshold, continuous=True,
                          energy_reward=energy_reward)
    policy = ContinuousNNNPolicy(obs_dim=5, hidden=H, std=sigma, t=64,
                                 force_max=force_mag, sigma_explore=sigma_explore,
                                 n_hidden_layers=n_hidden_layers)
    if init_body is not None:
        policy.net.load_state_dict(init_body)
    if fixed_field is not None:
        policy.field = fixed_field
    critic = ValueMLP(5)
    keys = C.param_keys(policy)
    actor_opt = ManualOpt("adam")
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    norm = norm_obj if norm_obj is not None else RunningNorm(5)

    hist, checkpoints = [], []
    for upd in range(updates):
        policy.sigma_explore = sigma_explore + (sigma_explore_end - sigma_explore) * (
            upd / max(1, updates - 1))                       # anneal exploration
        b_psi, b_adv, b_ret, b_obs = [], [], [], []
        ep_returns = []
        for _ in range(episodes_per_update):
            if start_center is not None:                     # balance pre-train: near a center
                start = start_center + float(rng.uniform(-start_range, start_range))
            else:
                start = (math.pi + float(rng.uniform(-0.5, 0.5))
                         if rng.random() < bottom_frac else float(rng.uniform(-math.pi, math.pi)))
            obs, _ = env.reset(seed=None, start_theta=start)
            on = _norm(norm, obs, update_norm)
            psis, rews, vals, obses = [], [], [], []
            for _ in range(horizon):
                if fields is not None and fixed_field is None:
                    _set_field(policy, fields, float(obs[2]), gate_k, gate_c)  # cos(theta)
                step = policy.rollout_step(on.unsqueeze(0))
                psis.append(C.cov_jac_grad(policy, step))
                with torch.no_grad():
                    vals.append(float(critic(on.unsqueeze(0))))
                obses.append(on)
                obs, r, te, tr, _ = env.step(float(step.action.item()))
                rews.append(r)
                on = _norm(norm, obs, update_norm)
                if te or tr:
                    break
            with torch.no_grad():
                v_last = 0.0 if te else float(critic(on.unsqueeze(0)))  # bootstrap
            n = len(rews)
            adv = np.zeros(n)
            gae = 0.0
            for t in range(n - 1, -1, -1):
                v_next = vals[t + 1] if t + 1 < n else v_last
                delta = rews[t] + gamma * v_next - vals[t]
                gae = delta + gamma * lam * gae
                adv[t] = gae
            ret = adv + np.array(vals)
            b_psi.extend(psis)
            b_adv.extend(adv.tolist())
            b_ret.extend(ret.tolist())
            b_obs.extend(obses)
            ep_returns.append(float(sum(rews)))
        hist.append(np.mean(ep_returns))

        # --- actor update: forward-mirror policy gradient, GAE-weighted ---
        A = np.array(b_adv)
        A = (A - A.mean()) / (A.std() + 1e-6)
        N = len(b_psi)
        grad = {k: torch.zeros_like(_p(policy, k)) for k in keys}
        for t in range(N):
            for k in keys:
                grad[k] += float(A[t]) * b_psi[t][k]
        for k in keys:
            g = grad[k] / N
            if freeze_mask is not None and k in freeze_mask:
                g = g * (1.0 - freeze_mask[k])               # freeze subnetwork-A params
            actor_opt.update(str(k), _p(policy, k), -g, lr_actor)

        # --- critic update: backprop MSE to GAE returns ---
        obs_b = torch.stack(b_obs)
        ret_b = torch.tensor(b_ret, dtype=torch.float32)
        for _ in range(critic_epochs):
            critic_opt.zero_grad()
            loss = ((critic(obs_b) - ret_b) ** 2).mean()
            loss.backward()
            critic_opt.step()

        if checkpoint_every and (upd + 1) % checkpoint_every == 0:
            checkpoints.append((upd + 1, _snapshot(policy, norm, H, force_mag,
                                                   fields, gate_k, gate_c)))
        if verbose and (upd + 1) % 10 == 0:
            print(f"  [a2c seed{seed}] upd {upd+1:4d}  ep_return/step "
                  f"{np.mean(hist[-10:]) / horizon:+.3f}")
    return policy, env, norm, critic, checkpoints, hist


def _snapshot(policy, norm, H, force_mag, fields=None, gate_k=6.0, gate_c=0.0):
    std = torch.sqrt(norm.M2 / norm.count + norm.eps)
    return {"net": {k: v.detach().clone() for k, v in policy.net.state_dict().items()},
            "norm_mean": norm.mean.detach().clone(), "norm_std": std.detach().clone(),
            "hidden": H, "t": policy.t, "force_mag": force_mag,
            "n_layers": len(policy.crossings),
            "fields": fields, "gate_k": gate_k, "gate_c": gate_c}


def build_policy(state):
    p = ContinuousNNNPolicy(obs_dim=5, hidden=int(state["hidden"]), t=int(state["t"]),
                            force_max=state["force_mag"], sigma_explore=0.1,
                            n_hidden_layers=int(state.get("n_layers", 2)))
    p.net.load_state_dict(state["net"])
    p.eval()
    p._opt_fields = state.get("fields")            # noise-field option (方向3a), or None
    p._gate_k, p._gate_c = state.get("gate_k", 6.0), state.get("gate_c", 0.0)
    return p, state["norm_mean"], state["norm_std"]


def eval_from_bottom(state, horizon=500, seeds=(0, 1, 2)):
    p, mean, std = build_policy(state)
    fm = state["force_mag"]
    out = []
    for s in seeds:
        env = CartPoleSwingUp(horizon=horizon, random_start=False, seed=s,
                              force_mag=fm, x_threshold=4.0, continuous=True)
        obs, _ = env.reset(seed=s)
        cs = []
        for _ in range(horizon):
            if p._opt_fields is not None:
                _set_field(p, p._opt_fields, float(obs[2]), p._gate_k, p._gate_c)
            on = torch.clamp((torch.tensor(obs, dtype=torch.float32) - mean) / std, -5, 5)
            step = p.rollout_step(on.unsqueeze(0), greedy=True)
            obs, r, te, tr, _ = env.step(float(step.action.item()))
            cs.append(env.cos_theta())
        cs = np.array(cs)
        out.append((cs.mean(), (cs > 0.9).mean(), (cs[-100:] > 0.9).mean()))
    return (float(np.mean([o[0] for o in out])), float(np.mean([o[1] for o in out])),
            float(np.mean([o[2] for o in out])))
