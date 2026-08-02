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
from .a2c_swingup import _norm, _set_field, _snapshot
from .a2c_nnncritic import _snap


def train_ppo_nnn(seed=0, H=128, Hc=64, sigma=0.6, updates=300, episodes_per_update=3,
                  horizon=400, gamma=0.99, lam=0.95, lr_actor=0.01, lr_critic=0.02,
                  ppo_epochs=4, critic_epochs=4, clip_eps=0.2, kl_target=0.02,
                  bottom_frac=0.5, top_frac=0.0, top_range=0.3,
                  force_mag=20.0, x_threshold=4.0,
                  sigma_explore=0.4, sigma_explore_end=0.1,
                  mirror_beta=0.1, checkpoint_every=25, verbose=True,
                  wall_mode="stop", wall_penalty=3.0, x_barrier=0.0, alive_bonus=0.0,
                  top_center=0.0, fill_batch=False, init_policy=None, init_critic=None,
                  lr_var_scale=False,
                  internal_noise=False, temp_fields=None, gate_k=6.0, gate_c=0.0,
                  temp_out=None, draw_mode="sample", var_override=None):
    # temp_out = (hot, cold): readout-unit noise sigma_out gated per step by
    # g = sigmoid(gate_k*(cos theta - gate_c)) -- hot while pumping (g~0), cold near
    # upright (g~1).  This is the temperature lever with actual leverage (the body
    # field saturates, see policy_cont.sigma_out).  hot == cold gives a constant
    # internal temperature (ablation arm).
    # internal_noise (§25 Stage 1): drop the external sigma_e entirely; the executed
    # action is a real internal readout sample and the per-step marginal variance
    # var = Var_m(o)*(1+1/T) replaces sigma_e^2 in the score/ratio/KL accounting.
    # temp_fields = [field_hot, field_cold] (per-layer per-unit sigma tensors): the
    # noise field is blended per step by the context gate g = sigmoid(gate_k*(cos
    # theta - gate_c)), so the exploration TEMPERATURE is a field-controlled physical
    # quantity (hot while pumping, cold near upright).  None = constant field.
    # lr_var_scale: scale the actor lr by (sig_e/sigma_explore)^2.  The Gaussian score
    # (a-mu)/var grows as 1/var when sigma_e anneals, so a constant lr means the
    # effective policy step BLOWS UP at low sigma -- the documented v4 terminal
    # collapse at sigma_e -> 0.1 (§23.10 limitation).  Scaling lr with var keeps the
    # effective step roughly constant, allowing sigma_e to anneal below 0.2 for the
    # fine balance control that the wall-free hold needs.
    # wall_mode="end" (no-stopper task): wall contact terminates the episode, so early
    # rollouts are SHORT; fill_batch=True keeps collecting episodes until the batch holds
    # ~episodes_per_update*horizon steps, so the PPO batch statistics (advantage
    # normalization, KL average) stay comparable across training.
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = CartPoleSwingUp(horizon=horizon, seed=seed, force_mag=force_mag,
                          x_threshold=x_threshold, continuous=True,
                          wall_mode=wall_mode, wall_penalty=wall_penalty,
                          x_barrier=x_barrier, alive_bonus=alive_bonus,
                          top_center=top_center)
    policy = ContinuousNNNPolicy(obs_dim=5, hidden=H, std=sigma, t=64,
                                 force_max=force_mag, sigma_explore=sigma_explore,
                                 noise_mode="internal" if internal_noise else "external")
    policy.draw_mode = draw_mode
    policy.var_override = var_override
    critic = NNNCritic(obs_dim=5, hidden=Hc, std=sigma, t=64)
    akeys, ckeys = C.param_keys(policy), C.param_keys(critic)
    actor_opt, critic_opt = ManualOpt("adam"), ManualOpt("adam")
    amir = C.MirrorEMA(mirror_beta) if mirror_beta else None
    cmir = C.MirrorEMA(mirror_beta) if mirror_beta else None
    norm = RunningNorm(5)
    # warm start (curriculum phase 2): resume the policy/critic weights and the obs
    # normalizer of an earlier run.  Optimizer moments, mirrors (EMA window ~19 updates)
    # and the return standardizer re-estimate within a few updates.
    if init_policy is not None:
        policy.net.load_state_dict(init_policy["net"])
        norm.mean = init_policy["norm_mean"].clone()
        norm.count = 1e5
        norm.M2 = init_policy["norm_std"].clone() ** 2 * norm.count
    if init_critic is not None:
        critic.net.load_state_dict(init_critic)
    ret_mean, ret_std, ret_seen = 0.0, 1.0, 0.0

    hist, stats, checkpoints = [], [], []
    for upd in range(updates):
        sig_e = sigma_explore + (sigma_explore_end - sigma_explore) * (
            upd / max(1, updates - 1))
        policy.sigma_explore = sig_e
        b_obs, b_anorm, b_muold, b_sigmu, b_adv, b_ret = [], [], [], [], [], []  # a/Fmax
        b_var, b_cos = [], []          # per-step collection variance / raw cos(theta)
        ep_returns, ep_lens = [], []
        wall_hits = 0
        target_steps = episodes_per_update * horizon
        while True:
            # start curriculum: bottom (learn to pump), near-top (learn the wall-free
            # catch/hold -- with stoppers the old policy balanced by LEANING on a wall,
            # so holding upright on an open track is a regime that bottom/uniform starts
            # alone under-sample), or any angle (transitions).
            u = rng.random()
            if u < bottom_frac:
                start = math.pi + float(rng.uniform(-0.5, 0.5))
            elif u < bottom_frac + top_frac:
                start = float(rng.uniform(-top_range, top_range))
            else:
                start = float(rng.uniform(-math.pi, math.pi))
            obs, _ = env.reset(seed=None, start_theta=start)
            on = _norm(norm, obs, True)
            obses, anorms, muolds, sigmus, rews, vals, varss, coss = \
                [], [], [], [], [], [], [], []
            for _ in range(horizon):
                if temp_fields is not None:
                    _set_field(policy, temp_fields, float(obs[2]), gate_k, gate_c)
                if temp_out is not None:
                    g = 1.0 / (1.0 + math.exp(-gate_k * (float(obs[2]) - gate_c)))
                    policy.sigma_out = temp_out[0] * (1.0 - g) + temp_out[1] * g
                step = policy.rollout_step(on.unsqueeze(0))
                v_std, _ = critic.value_step(on.unsqueeze(0))
                vals.append(v_std * ret_std + ret_mean)
                obses.append(on)
                if internal_noise:
                    spread2 = float(step.y_samples.var().item())
                    var_c = (policy.var_override if policy.var_override is not None
                             else max(spread2 * (1.0 + 1.0 / policy.t),
                                      policy.var_floor))
                    sig_mu = math.sqrt(spread2 / policy.t)
                else:
                    var_c = policy.sigma_explore ** 2
                    sig_mu = float(step.y_samples.std()) / math.sqrt(policy.t)  # mu noise
                # recover the UNCLAMPED normalized action from the stored Gaussian score
                # (a = mu + score*var_c; step.action is clamped and would bias the
                # score/ratio toward saturation)
                anorms.append(float(step.p.item()) + float(step.score.item()) * var_c)
                muolds.append(float(step.p.item()))    # collection-time mu = old policy
                sigmus.append(sig_mu)
                varss.append(var_c)
                coss.append(float(obs[2]))
                obs, r, te, tr, info = env.step(float(step.action.item()))
                rews.append(r)
                wall_hits += int(bool(info.get("wall", False)))
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
            b_var.extend(varss)
            b_cos.extend(coss)
            b_adv.extend(adv.tolist())
            b_ret.extend((adv + np.array(vals)).tolist())
            ep_returns.append(float(sum(rews)))
            ep_lens.append(n)
            if fill_batch:
                if len(b_obs) >= target_steps or len(ep_returns) >= 20 * episodes_per_update:
                    break
            elif len(ep_returns) >= episodes_per_update:
                break
        hist.append(np.mean(ep_returns))

        rb = np.array(b_ret)
        ret_seen += 1
        ret_mean += (rb.mean() - ret_mean) / min(ret_seen, 100)
        ret_std = 0.99 * ret_std + 0.01 * (rb.std() + 1e-6)

        A = np.array(b_adv)
        A = (A - A.mean()) / (A.std() + 1e-6)
        N = len(b_obs)

        mu_old = np.array(b_muold)                # collection-time mu = pi_old (no re-forward)
        sig_mu = np.array(b_sigmu)
        # marginal-policy variance per sample: internal-noise var already contains the
        # mu-estimation term (spread^2*(1+1/T)); the external sigma_e^2 does not.
        var_t = (np.array(b_var) if internal_noise
                 else np.array(b_var) + sig_mu ** 2)
        if internal_noise:
            sig_e = float(np.sqrt(np.mean(var_t)))   # reported temperature (physical)
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
                if temp_fields is not None:
                    _set_field(policy, temp_fields, b_cos[t], gate_k, gate_c)
                if temp_out is not None:
                    g = 1.0 / (1.0 + math.exp(-gate_k * (b_cos[t] - gate_c)))
                    policy.sigma_out = temp_out[0] * (1.0 - g) + temp_out[1] * g
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
            lr_a = lr_actor * ((sig_e / sigma_explore) ** 2 if lr_var_scale else 1.0)
            for k in akeys:
                actor_opt.update(str(k), _p(policy, k), -ga[k] / N, lr_a)
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

        stats.append({"upd": upd + 1, "ep_return": float(np.mean(ep_returns)),
                      "ret_step": float(np.mean([r / max(1, n) for r, n in
                                                 zip(ep_returns, ep_lens)])),
                      "ep_len": float(np.mean(ep_lens)), "n_eps": len(ep_returns),
                      "wall_hits": wall_hits, "r2": r2, "clip_frac": clip_frac,
                      "kl": kl, "sig_e": sig_e})

        if checkpoint_every and (upd + 1) % checkpoint_every == 0:
            if temp_fields is not None:
                checkpoints.append((upd + 1, _snapshot(policy, norm, H, force_mag,
                                                       fields=temp_fields, gate_k=gate_k,
                                                       gate_c=gate_c)))
            else:
                checkpoints.append((upd + 1, _snap(policy, norm, H, force_mag,
                                                   None, 6.0, 0.0)))
        if verbose and (upd + 1) % 10 == 0:
            print(f"  [ppo seed{seed}] upd {upd+1:4d}  ep_return/step "
                  f"{stats[-1]['ret_step']:+.3f}  ep_len {stats[-1]['ep_len']:5.0f}  "
                  f"walls {wall_hits:2d}  value R2 {r2:+.3f}  "
                  f"clip_frac {clip_frac:.2f}  kl {kl:.4f}", flush=True)
    return policy, critic, norm, checkpoints, hist, stats
