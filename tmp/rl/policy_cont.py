"""tmp/rl.policy_cont -- continuous-force NNN policy (§3.1: the sample set IS the policy).

The linear readout produces a per-sample normalized force o^(m); the policy is the Gaussian
fitted to the T internal samples: mean mu = mean_m o^(m), std sigma = std_m o^(m).  The
executed (normalized) action is a ~ N(mu, sigma^2); the physical force sent to the env is
force_max * a.  The top-level output-space score is the Gaussian score

    score = (a - mu) / sigma^2          (= d log pi / d mu),

which the forward mirror (cov_jac) propagates to the body -- no transposed-weight backprop.
Because sigma is the intrinsic NNN sample spread (never collapses to 0), exploration is
persistent, avoiding the deterministic collapse that trapped the discrete Bernoulli policy.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import constants  # noqa: F401
from nnn import model
from data_nce.fncl.network import Capture
from .policy import StepData


class ContinuousNNNPolicy(nn.Module):
    def __init__(self, obs_dim, hidden=128, std=0.6, h=0.15, t=64, force_max=20.0,
                 sigma_explore=0.3, n_hidden_layers=2):
        super().__init__()
        self.structure = [obs_dim] + [hidden] * n_hidden_layers + [1]
        self.net = model.SimpleNNNSample(structure=self.structure, std=std, h=h, t=t,
                                         output_bias=True)
        self.t = t
        self.hidden = hidden
        self.force_max = force_max
        # FIXED (annealable) action-exploration std, decoupled from the readout sample
        # spread: the sample-spread sigma collapses and makes the (a-mu)/sigma^2 score
        # explode.  The NNN still supplies the mean mu (its forward-mirror credit); the
        # action noise is a plain fixed Gaussian, which is stable.
        self.sigma_explore = sigma_explore
        self.field = None

    @property
    def crossings(self):
        return self.net.gaussian_crossing

    @property
    def fcs(self):
        return self.net.fcs

    def rollout_step(self, obs, greedy=False):
        cap = Capture(self.net)
        try:
            o_mean = self.net(obs, stds=self.field)          # [N, 1] ensemble-mean readout
        finally:
            cap.remove()
        y = cap.y_samples                                    # [N, T, 1] per-sample forces
        mu = o_mean                                          # [N, 1]  NNN readout mean
        sigma = self.sigma_explore                           # fixed exploration std
        with torch.no_grad():
            a = mu if greedy else mu + sigma * torch.randn_like(mu)
        score = (a - mu) / (sigma ** 2)                      # [N, 1]  d log pi / d mu
        force = (self.force_max * a).clamp(-self.force_max, self.force_max)
        return StepData(obs=obs, p=mu.detach(), action=force.detach(), logp=None,
                        d=[t.clone() for t in cap.d],
                        z=[t.clone() for t in cap.z],
                        y_samples=y.clone(), value=None, v_samples=None,
                        score=score.detach())
