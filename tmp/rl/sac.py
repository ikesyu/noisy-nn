"""tmp/rl.sac -- SAC-style off-policy actor-critic on the fully-NNN system (no backprop).

Feasibility test (idea_rl.md §23.11): can the (i)-(v) prescriptions from the PPO port
(§23.10) carry cov_jac into an OFF-POLICY, replay-based, twin-Q, max-entropy algorithm?

Design (likelihood-ratio SAC):

  - Twin Q-critics Q1, Q2 are NNNs on the concatenated input [s, a] (obs_dim+1), trained
    by cov_jac regression (batched, EMA mirror each) to the TD target
        y = r + gamma (1-d) ( min(Q1', Q2')(s', a') - alpha log pi(a'|s') ),
    with polyak-averaged target copies Q1', Q2' and a' freshly sampled from the CURRENT
    policy at s' (SAC's soft value).  Regression to a target is exactly the regime cov_jac
    was validated for; the twin-min counters overestimation from noisy Q evaluation.
  - The actor cannot use SAC's reparameterized gradient (it would need backprop THROUGH
    Q into the policy).  Instead we use the score-function form of the same objective:
        grad J = E_{a~pi} [ grad log pi(a|s) * ( minQ(s,a) - alpha log pi(a|s) - b ) ],
    with a freshly sampled at replay states and b = batch mean (advantage normalization).
    cov_jac supplies grad log pi via the usual score recursion -- so SAC's actor step is
    "weight * psi", the same shape as A2C/PPO with the GAE advantage replaced by the
    entropy-regularized Q.  Because the cov_jac recursion is LINEAR in the top score,
    per-sample weights are applied by pre-multiplying the score, which lets the whole
    minibatch run as ONE batched [B, T, H] forward (large speedup over per-step loops).

Off-policy and the mirror (the §23.6-B2 concern, resolved): the weight mirror estimates
the CURRENT weights from the T-sample fluctuations of the CURRENT forward pass; the
replay distribution only chooses WHERE the estimate is probed.  Staleness of data never
enters the mirror -- EMA + Kolen-Pollack tracking works unchanged.

PPO prescriptions applied (idea_rl.md §23.10 knowledge 3):
  (i)   marginal variance var_t = sigma_e^2 + sigma_mu^2 in every log pi and score;
  (ii)  (no ratio in SAC -- not needed);
  (iii) (no epoch reuse of one batch -- fresh minibatches; trust region via small lr);
  (iv)  replay stores the UNCLAMPED action (recovered from the Gaussian score);
  (v)   sigma_e is a CONSTANT 0.3 (no anneal): SAC-style persistent exploration and the
        sigma-floor lesson in one.  With fixed sigma the entropy of pi is constant, so
        the alpha term only reweights actions (penalizes reinforcing high-density ones);
        learning sigma itself stays the open §22.2(c) item.

Additional off-policy lesson (run 1, negative result): the running TARGET standardization
that served A2C/PPO diverges under bootstrapping (Qmin -> -1e14).  There the regression
target (GAE return) is bounded by real rewards; in SAC the target passes through the
DE-standardized critic itself, so the running scale and the bootstrap feed each other
(y ~ q_std * Q_hat -> q_std grows -> y grows ...).  Fix: NO adaptive standardization --
fixed reward_scale (Q stays O(10) in scaled units) plus a hard clamp of the TD target
at +/- r_max/(1-gamma), the standard SAC treatment.

Actor-baseline lesson (run 2, negative result): with ONE sampled action per state and a
BATCH-mean baseline, the weight w = Q(s,a) - mean_batch(Q) is dominated by STATE-value
differences across the heterogeneous replay minibatch, not by action advantage -- the
policy gradient drowns (run 2 never left the hang-down region).  A2C/PPO avoided this
because GAE is already a per-state action advantage.  Fix: sample K actions per state
and use the PER-STATE (leave-one-out style) baseline w_k = Q(s,a_k) - mean_k Q(s,a_.);
since sum_k w_k = 0 and the cov_jac recursion is linear in the top score, the K scores
share one forward:  score = sum_k w_k a_k / (K var_t)  -- effectively a Q-weighted
covariance of the state's own action samples, evaluated on one set of internals.

Run 3-4 (fresh-sample actor, negative): the policy escapes the hang but stalls in the
wall attractor (best 0.21-0.38) even with antithetic K=8 pairs and 2-pass Q averaging --
the action-sensitivity of Q around mu is too small relative to its residual noise, so
the FRESH-sample soft-Q covariance stays low-SNR.  Run 5 (`actor_mode="replay"`) instead
anchors the actor on the EXECUTED replay actions (which carry the real outcomes Q was
trained on): w = A_soft(s, a_replay) with the same per-state K-sample baseline, made
off-policy-safe by prescription (ii): the importance ratio pi_new/pi_old with the
NOISE-DEADBAND clip from §23.10 (mu_old, sig_mu_old stored in replay at collection).
This is the point where all five PPO prescriptions are exercised inside SAC.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch

from . import constants  # noqa: F401
from .policy_cont import ContinuousNNNPolicy
from nnn import model
from data_nce.fncl.network import Capture
from .policy import StepData
from . import credit as C
from .envs_swingup import CartPoleSwingUp
from .train import ManualOpt, RunningNorm
from .multimode import _p
from .a2c_swingup import _norm
from .a2c_nnncritic import _snap


class QNNN(torch.nn.Module):
    """Q(s, a) as an NNN on the concatenated [s, a] input; cov_jac-trainable."""

    def __init__(self, obs_dim, hidden=64, std=0.6, h=0.15, t=64, n_hidden_layers=2):
        super().__init__()
        structure = [obs_dim + 1] + [hidden] * n_hidden_layers + [1]
        self.net = model.SimpleNNNSample(structure=structure, std=std, h=h, t=t,
                                         output_bias=True)
        self.t = t

    @property
    def crossings(self):
        return self.net.gaussian_crossing

    @property
    def fcs(self):
        return self.net.fcs

    def q_step(self, sa):
        """sa [N, obs_dim+1] -> (q [N], StepData with forward internals)."""
        cap = Capture(self.net)
        try:
            q = self.net(sa)                                 # [N, 1] ensemble mean
        finally:
            cap.remove()
        step = StepData(obs=sa, p=None, action=None, logp=None,
                        d=[t.clone() for t in cap.d],
                        z=[t.clone() for t in cap.z],
                        y_samples=cap.y_samples.clone(), value=q.detach(),
                        v_samples=None, score=None)
        return q.detach().squeeze(1), step

    def q_eval(self, sa):
        with torch.no_grad():
            return self.net(sa).squeeze(1)


def _pi_forward(policy, s):
    """Batched policy forward at states s [B, 5]: returns (mu [B], sig_mu [B], step)."""
    step = policy.rollout_step(s, greedy=True)               # greedy: internals + mu only
    mu = step.p.squeeze(1)
    sig_mu = step.y_samples.squeeze(-1).std(dim=1) / math.sqrt(policy.t)
    return mu, sig_mu, step


def _logpi(a, mu, var_t):
    return -(a - mu) ** 2 / (2 * var_t) - 0.5 * torch.log(2 * math.pi * var_t)


def train_sac_nnn(seed=0, H=128, Hq=64, sigma=0.6, episodes=300, horizon=400,
                  gamma=0.99, tau=0.01, lr_actor=0.003, lr_critic=0.02, alpha=0.1,
                  batch=64, rounds=32, act_samples=8, warmup_eps=5, replay_cap=60000,
                  sigma_explore=0.3, bottom_frac=0.5, force_mag=20.0, x_threshold=4.0,
                  reward_scale=0.05, mirror_beta=0.1, actor_mode="fresh", clip_eps=0.2,
                  checkpoint_every=25, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = CartPoleSwingUp(horizon=horizon, seed=seed, force_mag=force_mag,
                          x_threshold=x_threshold, continuous=True)
    policy = ContinuousNNNPolicy(obs_dim=5, hidden=H, std=sigma, t=64,
                                 force_max=force_mag, sigma_explore=sigma_explore)
    q1, q2 = QNNN(5, hidden=Hq, std=sigma, t=64), QNNN(5, hidden=Hq, std=sigma, t=64)
    q1t, q2t = copy.deepcopy(q1), copy.deepcopy(q2)
    akeys = C.param_keys(policy)
    qkeys = C.param_keys(q1)
    actor_opt = ManualOpt("adam")
    q1_opt, q2_opt = ManualOpt("adam"), ManualOpt("adam")
    amir = C.MirrorEMA(mirror_beta) if mirror_beta else None
    m1 = C.MirrorEMA(mirror_beta) if mirror_beta else None
    m2 = C.MirrorEMA(mirror_beta) if mirror_beta else None
    norm = RunningNorm(5)
    var_e = sigma_explore ** 2
    y_clamp = 4.0 * reward_scale / (1 - gamma)               # |r|<~4 -> principled bound

    # replay (normalized obs); mu_old / sig_mu_old at collection feed the (ii) ratio
    buf_s, buf_a, buf_r, buf_s2, buf_d, buf_mu, buf_sm = [], [], [], [], [], [], []

    def _push(s, a, r, s2, d, mu_o, sm_o):
        if len(buf_s) >= replay_cap:
            for b in (buf_s, buf_a, buf_r, buf_s2, buf_d, buf_mu, buf_sm):
                b.pop(0)
        buf_s.append(s); buf_a.append(a); buf_r.append(r); buf_s2.append(s2)
        buf_d.append(d); buf_mu.append(mu_o); buf_sm.append(sm_o)

    def _sample(B):
        idx = rng.integers(0, len(buf_s), size=B)
        s = torch.stack([buf_s[i] for i in idx])
        a = torch.tensor([buf_a[i] for i in idx], dtype=torch.float32)
        r = torch.tensor([buf_r[i] for i in idx], dtype=torch.float32)
        s2 = torch.stack([buf_s2[i] for i in idx])
        d = torch.tensor([buf_d[i] for i in idx], dtype=torch.float32)
        mu_o = torch.tensor([buf_mu[i] for i in idx], dtype=torch.float32)
        sm_o = torch.tensor([buf_sm[i] for i in idx], dtype=torch.float32)
        return s, a, r, s2, d, mu_o, sm_o

    def _q_update(qnet, qopt, qmir, s_a, y_std):
        q_pred, step = qnet.q_step(s_a)
        step.score = (q_pred - y_std).unsqueeze(1).to(torch.float32)   # [B,1] value error
        psi = qmir.grad(qnet, step) if qmir is not None else C.cov_jac_grad(qnet, step)
        if qmir is not None:
            qmir.snapshot_weights(qnet)
        for k in qkeys:
            qopt.update(str(k), _p(qnet, k), psi[k], lr_critic)
        if qmir is not None:
            qmir.kp_track(qnet)
        return float(((q_pred - y_std) ** 2).mean())

    hist, checkpoints = [], []
    for epi in range(episodes):
        # ---- collection (per-step N=1, intrinsic exploration sigma_e fixed) ----
        start = (math.pi + float(rng.uniform(-0.5, 0.5)) if rng.random() < bottom_frac
                 else float(rng.uniform(-math.pi, math.pi)))
        obs, _ = env.reset(seed=None, start_theta=start)
        on = _norm(norm, obs, True)
        ep_ret = 0.0
        for _ in range(horizon):
            step = policy.rollout_step(on.unsqueeze(0))
            a_unc = (float(step.p.item())
                     + float(step.score.item()) * sigma_explore ** 2)   # (iv) unclamped
            sm_o = float(step.y_samples.std()) / math.sqrt(policy.t)
            obs, r, te, tr, _ = env.step(float(step.action.item()))
            on2 = _norm(norm, obs, True)
            _push(on, a_unc, r * reward_scale, on2, float(te),
                  float(step.p.item()), sm_o)
            on = on2
            ep_ret += r
            if te or tr:
                break
        hist.append(ep_ret)

        # ---- updates (batched minibatches from replay) ----
        if epi + 1 < warmup_eps:
            continue
        qloss = wmean = 0.0
        for _ in range(rounds):
            s, a, r, s2, d, mu_o, sm_o = _sample(batch)
            with torch.no_grad():
                # soft TD target from the CURRENT policy at s'
                mu2, sig_mu2, _ = _pi_forward(policy, s2)
                var2 = var_e + sig_mu2 ** 2                            # (i) marginal
                a2 = mu2 + sigma_explore * torch.randn_like(mu2)
                logp2 = _logpi(a2, mu2, var2)
                sa2 = torch.cat([s2, a2.unsqueeze(1)], dim=1)
                qmin_t = torch.minimum(q1t.q_eval(sa2), q2t.q_eval(sa2))
                y = (r + gamma * (1 - d) * (qmin_t - alpha * logp2)
                     ).clamp(-y_clamp, y_clamp)             # fixed-scale, bounded target

            sa = torch.cat([s, a.unsqueeze(1)], dim=1)
            qloss = _q_update(q1, q1_opt, m1, sa, y)
            _q_update(q2, q2_opt, m2, sa, y)

            # ---- actor: score-function soft-Q ascent, K actions per replay state ----
            # antithetic pairs a = mu +/- sigma*|xi| (variance reduction for the per-state
            # action-Q covariance -- the same trick as the KDE slope's antithetic pair),
            # and the Q used to rank actions is averaged over 2 stochastic passes to
            # push its evaluation noise below the action-sensitivity signal.
            mu, sig_mu, astep = _pi_forward(policy, s)
            var_t = var_e + sig_mu ** 2                                # (i)
            K = act_samples
            half = torch.abs(torch.randn(len(mu), K // 2)) * sigma_explore
            a_k = mu.unsqueeze(1) + torch.cat([half, -half], dim=1)    # [B,K] antithetic
            with torch.no_grad():
                sa_k = torch.cat([s.repeat_interleave(K, dim=0),
                                  a_k.reshape(-1, 1)], dim=1)          # [B*K, 6]
                qmin = torch.zeros(len(sa_k))
                for _ in range(2):
                    qmin += torch.minimum(q1.q_eval(sa_k), q2.q_eval(sa_k)) / 2
                qmin = qmin.reshape(-1, K)
            if actor_mode == "replay":
                # v5: anchor on EXECUTED actions; K fresh samples serve only as the
                # per-state baseline.  Off-policy safety = prescription (ii): noise-
                # deadband importance ratio wrt the stored collection-time policy.
                with torch.no_grad():
                    sa_r = torch.cat([s, a.unsqueeze(1)], dim=1)
                    q_r = (torch.minimum(q1.q_eval(sa_r), q2.q_eval(sa_r))
                           + torch.minimum(q1.q_eval(sa_r), q2.q_eval(sa_r))) / 2
                adv = q_r - qmin.mean(dim=1)                           # soft advantage
                adv = adv / (adv.std() + 1e-6)
                logr = (-(a - mu) ** 2 + (a - mu_o) ** 2) / (2 * var_t)
                r_imp = torch.exp(logr.clamp(-10, 10))
                sig_logr = sm_o * (a - mu_o).abs() / var_t
                eps_t = clip_eps + 2.0 * sig_logr                      # (ii) deadband
                keep = ~((adv >= 0) & (r_imp > 1 + eps_t) | (adv < 0) & (r_imp < 1 - eps_t))
                r_use = r_imp.clamp(1 - eps_t, 1 + eps_t) * keep
                score = r_use * adv * (a - mu) / var_t
            else:
                logp = _logpi(a_k, mu.unsqueeze(1), var_t.unsqueeze(1))    # [B,K]
                w = qmin - alpha * logp                                    # soft-Q weight
                w = w - w.mean(dim=1, keepdim=True)                        # PER-STATE baseline
                w = w / (w.std() + 1e-6)                                   # global scale only
                # shared-internals composition: sum_k w_k (a_k - mu) / (K var_t)
                score = (w * (a_k - mu.unsqueeze(1))).sum(dim=1) / (K * var_t)
            astep.score = score.unsqueeze(1).to(torch.float32)
            psi = amir.grad(policy, astep) if amir is not None else C.cov_jac_grad(policy, astep)
            if amir is not None:
                amir.snapshot_weights(policy)
            for k in akeys:
                actor_opt.update(str(k), _p(policy, k), -psi[k], lr_actor)
            if amir is not None:
                amir.kp_track(policy)
            wmean = float(qmin.mean())

            with torch.no_grad():                                      # polyak targets
                for qt, q in ((q1t, q1), (q2t, q2)):
                    for pt, p in zip(qt.parameters(), q.parameters()):
                        pt.mul_(1 - tau).add_(tau * p)

        if checkpoint_every and (epi + 1) % checkpoint_every == 0:
            checkpoints.append((epi + 1, _snap(policy, norm, H, force_mag,
                                               None, 6.0, 0.0)))
        if verbose and (epi + 1) % 10 == 0:
            print(f"  [sac seed{seed}] epi {epi+1:4d}  ep_return/step "
                  f"{np.mean(hist[-10:]) / horizon:+.3f}  qloss {qloss:.4f}  "
                  f"Qmin(pi) {wmean:+.2f}  replay {len(buf_s)}")
    return policy, (q1, q2), norm, checkpoints, hist
