"""tmp/rl.a2c_nnncritic -- fully-NNN actor-critic: both trained by cov_jac, no backprop.

The final integration step (ユーザ要望): replace the external backprop critic with an NNN
critic trained by the SAME forward mirror (cov_jac) as the actor -- regressing the GAE
returns.  Now the WHOLE system (policy + value + exploration + credit for both) is one
forward-only NNN computation: no transposed-weight backward pass anywhere.  Returns are
standardized (running mean/std) so the value readout stays at a manageable scale.

`mirror_beta` (fix 1, idea_rl.md §23.9): BOTH the actor and the critic use a persistent
EMA weight mirror with Kolen-Pollack tracking (credit.MirrorEMA) instead of the original
per-step single-shot mirror.  The critic body keeps LEARNING by cov_jac (features are
trained, forward-only) -- only the mirror estimate is stabilized.  §23.9 showed this
alone lifts §23.5's 0.44 to full balance, so beta=0.1 is now the DEFAULT; pass
mirror_beta=None for the historical §23.5 single-shot behaviour.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import constants  # noqa: F401
from .policy_cont import ContinuousNNNPolicy
from .critic import NNNCritic
from . import credit as C
from .envs_swingup import CartPoleSwingUp
from .train import ManualOpt, RunningNorm
from .multimode import _p
from .a2c_swingup import _norm, _set_field


def train_a2c_nnn(seed=0, H=128, Hc=64, sigma=0.6, updates=400, episodes_per_update=3,
                  horizon=400, gamma=0.99, lam=0.95, lr_actor=0.01, lr_critic=0.02,
                  critic_epochs=4, bottom_frac=0.5, force_mag=20.0, x_threshold=4.0,
                  sigma_explore=0.4, sigma_explore_end=0.1, fields=None, gate_k=6.0,
                  gate_c=0.0, mirror_beta=0.1, checkpoint_every=25, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = CartPoleSwingUp(horizon=horizon, seed=seed, force_mag=force_mag,
                          x_threshold=x_threshold, continuous=True)
    policy = ContinuousNNNPolicy(obs_dim=5, hidden=H, std=sigma, t=64,
                                 force_max=force_mag, sigma_explore=sigma_explore)
    critic = NNNCritic(obs_dim=5, hidden=Hc, std=sigma, t=64)
    akeys, ckeys = C.param_keys(policy), C.param_keys(critic)
    actor_opt, critic_opt = ManualOpt("adam"), ManualOpt("adam")
    norm = RunningNorm(5)
    amir = C.MirrorEMA(mirror_beta) if mirror_beta else None      # fix 1: persistent mirrors
    cmir = C.MirrorEMA(mirror_beta) if mirror_beta else None
    ret_mean, ret_std, ret_seen = 0.0, 1.0, 0.0            # running return standardization

    hist, checkpoints = [], []
    for upd in range(updates):
        policy.sigma_explore = sigma_explore + (sigma_explore_end - sigma_explore) * (
            upd / max(1, updates - 1))
        b_psi, b_cstep, b_vstd, b_adv, b_ret = [], [], [], [], []
        ep_returns = []
        for _ in range(episodes_per_update):
            start = (math.pi + float(rng.uniform(-0.5, 0.5)) if rng.random() < bottom_frac
                     else float(rng.uniform(-math.pi, math.pi)))
            obs, _ = env.reset(seed=None, start_theta=start)
            on = _norm(norm, obs, True)
            psis, csteps, vstds, rews, vals = [], [], [], [], []
            for _ in range(horizon):
                if fields is not None:
                    _set_field(policy, fields, float(obs[2]), gate_k, gate_c)
                step = policy.rollout_step(on.unsqueeze(0))
                psis.append(amir.grad(policy, step) if amir is not None
                            else C.cov_jac_grad(policy, step))
                v_std, cstep = critic.value_step(on.unsqueeze(0))   # NNN critic forward
                csteps.append(cstep); vstds.append(v_std)
                vals.append(v_std * ret_std + ret_mean)
                obs, r, te, tr, _ = env.step(float(step.action.item()))
                rews.append(r)
                on = _norm(norm, obs, True)
                if te or tr:
                    break
            v_last = 0.0 if te else (critic.value_step(on.unsqueeze(0))[0] * ret_std + ret_mean)
            n = len(rews)
            adv = np.zeros(n)
            gae = 0.0
            for t in range(n - 1, -1, -1):
                v_next = vals[t + 1] if t + 1 < n else v_last
                gae = (rews[t] + gamma * v_next - vals[t]) + gamma * lam * gae
                adv[t] = gae
            b_psi.extend(psis)
            b_cstep.extend(csteps)
            b_vstd.extend(vstds)
            b_adv.extend(adv.tolist())
            b_ret.extend((adv + np.array(vals)).tolist())
            ep_returns.append(float(sum(rews)))
        hist.append(np.mean(ep_returns))

        # update return standardization from this batch's returns
        rb = np.array(b_ret)
        ret_seen += 1
        ret_mean += (rb.mean() - ret_mean) / min(ret_seen, 100)
        ret_std = 0.99 * ret_std + 0.01 * (rb.std() + 1e-6)

        # --- actor: forward-mirror policy gradient, GAE-weighted ---
        A = np.array(b_adv)
        A = (A - A.mean()) / (A.std() + 1e-6)
        N = len(b_psi)
        ga = {k: torch.zeros_like(_p(policy, k)) for k in akeys}
        for t in range(N):
            for k in akeys:
                ga[k] += float(A[t]) * b_psi[t][k]
        if amir is not None:
            amir.snapshot_weights(policy)
        for k in akeys:
            actor_opt.update(str(k), _p(policy, k), -ga[k] / N, lr_actor)
        if amir is not None:
            amir.kp_track(policy)

        # --- critic: cov_jac regression to standardized returns (no backprop) ---
        # single pass over the stored forward internals; value-error score = V_std - target
        tgt = (rb - ret_mean) / (ret_std + 1e-6)
        gc = {k: torch.zeros_like(_p(critic, k)) for k in ckeys}
        for t in range(N):
            cstep = b_cstep[t]
            cstep.score = torch.tensor([[b_vstd[t] - float(tgt[t])]])   # value-error score
            psi = (cmir.grad(critic, cstep) if cmir is not None
                   else C.cov_jac_grad(critic, cstep))
            for k in ckeys:
                gc[k] += psi[k]
        if cmir is not None:
            cmir.snapshot_weights(critic)
        for k in ckeys:
            critic_opt.update(str(k), _p(critic, k), gc[k] / N, lr_critic)
        if cmir is not None:
            cmir.kp_track(critic)

        if checkpoint_every and (upd + 1) % checkpoint_every == 0:
            checkpoints.append((upd + 1, _snap(policy, norm, H, force_mag, fields,
                                               gate_k, gate_c)))
        if verbose and (upd + 1) % 10 == 0:
            v = np.array(b_vstd)
            r2 = float(1.0 - ((v - tgt) ** 2).mean() / (tgt.var() + 1e-8))
            print(f"  [nnn-ac seed{seed}] upd {upd+1:4d}  ep_return/step "
                  f"{np.mean(hist[-10:]) / horizon:+.3f}  value R2 {r2:+.3f}")
    return policy, critic, norm, checkpoints, hist


def _snap(policy, norm, H, force_mag, fields, gate_k, gate_c):
    std = torch.sqrt(norm.M2 / norm.count + norm.eps)
    return {"net": {k: v.detach().clone() for k, v in policy.net.state_dict().items()},
            "norm_mean": norm.mean.detach().clone(), "norm_std": std.detach().clone(),
            "hidden": H, "t": policy.t, "force_mag": force_mag,
            "n_layers": len(policy.crossings),
            "fields": fields, "gate_k": gate_k, "gate_c": gate_c}
