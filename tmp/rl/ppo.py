"""tmp/rl.ppo -- PPO on the fully-NNN actor-critic (no backprop anywhere).

Motivation (idea_rl.md §23.6-B, §23.9 knowledge 4): the A2C runs reach full balance but
oscillate (1.000 checkpoints separated by collapses), forcing best-checkpoint selection.
PPO's clipped surrogate bounds how far one update can move the policy, which is exactly
the missing stabilizer -- and it needs almost nothing new from the NNN side:

  - The continuous policy is Gaussian a ~ N(mu(s), sigma_e^2) with FIXED exploration std,
    so log pi = -(a - mu)^2 / (2 sigma_e^2) + const is closed-form.  The ratio
    r = pi_new/pi_old only needs mu_new from a fresh forward of the stored obs.
  - cov_jac supplies d log pi / dW at the CURRENT policy point: re-forward the stored
    obs, set score = (a_stored - mu_new)/sigma_e^2, run the mirror recursion.  The
    clipped-surrogate gradient is then  clip_mask * r * A * psi  -- the clip factor just
    SCALES the existing credit.
  - Multiple epochs reuse the same rollouts (sample efficiency), which the clip makes safe.

NNN-specific correction (learned the hard way -- the naive port FREEZES the policy):
mu is a T-sample ensemble mean, so it carries estimation noise sigma_mu ~ std_T/sqrt(T).
The log-ratio then fluctuates by ~ sigma_mu*|a-mu|/var even when the policy has NOT
moved; with annealed sigma_e this exceeds clip_eps and ~50% of samples get clipped by
NOISE alone, killing the gradient (observed: clip_frac ~0.5, evals frozen from upd 75).
Fixes applied:
  (i)  marginal-policy variance  var_t = sigma_e^2 + sigma_mu(t)^2  in both the score
       and the ratio (the executed action IS drawn from the marginal N(mu_bar, var_t));
  (ii) noise-deadband clip: per-sample threshold eps_t = clip_eps + 2*sigma_logr(t),
       where sigma_logr(t) = sigma_mu(t)*|a-mu_old|/var_t -- clip only fires when the
       measured ratio deviation exceeds its own noise floor;
  (iii) the gradient scale uses the CLAMPED ratio clamp(r, 1-eps_t, 1+eps_t), bounding
       noise amplification through r.

Both actor and critic keep persistent EMA mirrors + Kolen-Pollack tracking (§23.9
default).  The critic (learned NNN, cov_jac) is refit for several epochs per update
against the frozen GAE returns of the batch.
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
from .a2c_swingup import _norm
from .a2c_nnncritic import _snap


def train_ppo_nnn(seed=0, H=128, Hc=64, sigma=0.6, updates=300, episodes_per_update=3,
                  horizon=400, gamma=0.99, lam=0.95, lr_actor=0.01, lr_critic=0.02,
                  ppo_epochs=4, critic_epochs=4, clip_eps=0.2, kl_target=0.02,
                  bottom_frac=0.5,
                  force_mag=20.0, x_threshold=4.0,
                  sigma_explore=0.4, sigma_explore_end=0.1,
                  mirror_beta=0.1, checkpoint_every=25, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = CartPoleSwingUp(horizon=horizon, seed=seed, force_mag=force_mag,
                          x_threshold=x_threshold, continuous=True)
    policy = ContinuousNNNPolicy(obs_dim=5, hidden=H, std=sigma, t=64,
                                 force_max=force_mag, sigma_explore=sigma_explore)
    critic = NNNCritic(obs_dim=5, hidden=Hc, std=sigma, t=64)
    akeys, ckeys = C.param_keys(policy), C.param_keys(critic)
    actor_opt, critic_opt = ManualOpt("adam"), ManualOpt("adam")
    amir = C.MirrorEMA(mirror_beta) if mirror_beta else None
    cmir = C.MirrorEMA(mirror_beta) if mirror_beta else None
    norm = RunningNorm(5)
    ret_mean, ret_std, ret_seen = 0.0, 1.0, 0.0

    hist, checkpoints = [], []
    for upd in range(updates):
        sig_e = sigma_explore + (sigma_explore_end - sigma_explore) * (
            upd / max(1, updates - 1))
        policy.sigma_explore = sig_e
        b_obs, b_anorm, b_muold, b_sigmu, b_adv, b_ret = [], [], [], [], [], []  # a/Fmax
        ep_returns = []
        for _ in range(episodes_per_update):
            start = (math.pi + float(rng.uniform(-0.5, 0.5)) if rng.random() < bottom_frac
                     else float(rng.uniform(-math.pi, math.pi)))
            obs, _ = env.reset(seed=None, start_theta=start)
            on = _norm(norm, obs, True)
            obses, anorms, muolds, sigmus, rews, vals = [], [], [], [], [], []
            for _ in range(horizon):
                step = policy.rollout_step(on.unsqueeze(0))
                v_std, _ = critic.value_step(on.unsqueeze(0))
                vals.append(v_std * ret_std + ret_mean)
                obses.append(on)
                # recover the UNCLAMPED normalized action from the stored Gaussian score
                # (a = mu + score*sigma_e^2; step.action is clamped and would bias the
                # score/ratio toward saturation)
                anorms.append(float(step.p.item())
                              + float(step.score.item()) * policy.sigma_explore ** 2)
                muolds.append(float(step.p.item()))    # collection-time mu = old policy
                sigmus.append(float(step.y_samples.std()) / math.sqrt(policy.t))  # mu noise
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
            b_obs.extend(obses)
            b_anorm.extend(anorms)
            b_muold.extend(muolds)
            b_sigmu.extend(sigmus)
            b_adv.extend(adv.tolist())
            b_ret.extend((adv + np.array(vals)).tolist())
            ep_returns.append(float(sum(rews)))
        hist.append(np.mean(ep_returns))

        rb = np.array(b_ret)
        ret_seen += 1
        ret_mean += (rb.mean() - ret_mean) / min(ret_seen, 100)
        ret_std = 0.99 * ret_std + 0.01 * (rb.std() + 1e-6)

        A = np.array(b_adv)
        A = (A - A.mean()) / (A.std() + 1e-6)
        N = len(b_obs)
        var_e = sig_e ** 2

        mu_old = np.array(b_muold)                # collection-time mu = pi_old (no re-forward)
        sig_mu = np.array(b_sigmu)
        var_t = var_e + sig_mu ** 2               # marginal-policy variance per sample
        sig_logr = sig_mu * np.abs(np.array(b_anorm) - mu_old) / var_t  # ratio noise floor
        eps_t = clip_eps + 2.0 * sig_logr         # noise-deadband clip threshold

        # --- actor: clipped-surrogate epochs on the same rollouts ---
        # KL early stop: halt the epoch loop once the mean policy shift exceeds the
        # trust region (standard PPO practice; without it the reused epochs overshoot
        # into the constant-action attractor -- observed in run 2).
        clip_frac, kl = 0.0, 0.0
        for ep in range(ppo_epochs):
            ga = {k: torch.zeros_like(_p(policy, k)) for k in akeys}
            n_clip = 0
            kl_sum = 0.0
            for t in range(N):
                step = policy.rollout_step(b_obs[t].unsqueeze(0), greedy=True)
                mu_new = float(step.p.item())
                kl_sum += (mu_new - mu_old[t]) ** 2 / (2 * var_t[t])
                a = b_anorm[t]
                logr = (-(a - mu_new) ** 2 + (a - mu_old[t]) ** 2) / (2 * var_t[t])
                r = float(np.exp(np.clip(logr, -10, 10)))
                adv_t = float(A[t])
                # clipped surrogate with per-sample noise deadband
                if (adv_t >= 0 and r > 1 + eps_t[t]) or (adv_t < 0 and r < 1 - eps_t[t]):
                    n_clip += 1
                    continue
                step.score = torch.tensor([[(a - mu_new) / var_t[t]]], dtype=torch.float32)
                psi = (amir.grad(policy, step) if amir is not None
                       else C.cov_jac_grad(policy, step))
                r_use = float(np.clip(r, 1 - eps_t[t], 1 + eps_t[t]))  # bound noise in r
                for k in akeys:
                    ga[k] += (r_use * adv_t) * psi[k]
            kl = kl_sum / N
            if ep > 0 and kl > kl_target:          # trust region exceeded: stop reusing
                clip_frac = n_clip / N
                break
            if amir is not None:
                amir.snapshot_weights(policy)
            for k in akeys:
                actor_opt.update(str(k), _p(policy, k), -ga[k] / N, lr_actor)
            if amir is not None:
                amir.kp_track(policy)
            clip_frac = n_clip / N

        # --- critic: cov_jac regression epochs to the frozen standardized returns ---
        tgt = (rb - ret_mean) / (ret_std + 1e-6)
        r2 = 0.0
        for ep in range(critic_epochs):
            gc = {k: torch.zeros_like(_p(critic, k)) for k in ckeys}
            v_pred = np.zeros(N)
            for t in range(N):
                v_std, cstep = critic.value_step(b_obs[t].unsqueeze(0))
                v_pred[t] = v_std
                cstep.score = torch.tensor([[v_std - float(tgt[t])]])
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
            r2 = float(1.0 - ((v_pred - tgt) ** 2).mean() / (tgt.var() + 1e-8))

        if checkpoint_every and (upd + 1) % checkpoint_every == 0:
            checkpoints.append((upd + 1, _snap(policy, norm, H, force_mag,
                                               None, 6.0, 0.0)))
        if verbose and (upd + 1) % 10 == 0:
            print(f"  [ppo seed{seed}] upd {upd+1:4d}  ep_return/step "
                  f"{np.mean(hist[-10:]) / horizon:+.3f}  value R2 {r2:+.3f}  "
                  f"clip_frac {clip_frac:.2f}  kl {kl:.4f}")
    return policy, critic, norm, checkpoints, hist
