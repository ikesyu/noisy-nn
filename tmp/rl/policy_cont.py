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
                 sigma_explore=0.3, n_hidden_layers=2, noise_mode="external",
                 var_floor=0.01):
        super().__init__()
        # noise_mode "external": a ~ N(mu, sigma_explore^2) with the hand-annealed
        #   scalar sigma_explore (the §23.1 compromise; exploration is EXTERNAL).
        # noise_mode "internal" (§25): the executed action IS one of the T internal
        #   readout samples (§2.1) -- exploration noise = the NNN's own fluctuation, so
        #   the noise FIELD controls the exploration temperature.  The score uses the
        #   marginal variance var = Var_m(o) * (1 + 1/T) clamped to an ABSOLUTE floor
        #   var_floor: the (1+1/T) marginal term alone is only a RELATIVE floor (it
        #   vanishes with the spread), and without the absolute clamp the §23.1 score
        #   blow-up returns in small-spread states (measured: KL 1.46, clip 0.52,
        #   training stuck).  var_floor = 0.01 (std 0.1 in normalized-force units).
        self.noise_mode = noise_mode
        self.var_floor = var_floor
        # decomposition knobs (§25.6 knowledge 3 -- which sub-mechanism carries the
        # internal-noise win):
        #   draw_mode "sample": execute a real ensemble sample (structured body
        #     deviation + readout noise).  "gauss": execute mu + sqrt(var)*eps -- same
        #     per-state magnitude, NO structured component (isolates mechanism (ii)).
        #   var_override: use a CONSTANT variance in the score instead of the
        #     per-state measured one (isolates mechanism (i); execution unchanged).
        self.draw_mode = "sample"
        self.var_override = None
        # sigma_out: the READOUT unit's own noise-field entry (internal mode only).
        # Calibration probes showed the body-sigma field has almost no leverage on the
        # action temperature (readout ensemble spread saturates at ~0.2 for body sigma
        # 0.9-1.8, because crossing activities are bounded), so the field controls the
        # temperature at the readout unit instead: o^(m) <- o^(m) + sigma_out*xi^(m).
        # This is the same physical quantity as every other unit's sigma (the noise
        # field extended to the readout), NOT an external exploration schedule; the
        # trainer gates it per step by context.
        self.sigma_out = 0.0
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
        if self.noise_mode == "internal":
            if self.sigma_out > 0:
                y = y + self.sigma_out * torch.randn_like(y)   # readout-unit noise
            raw_var = y.var(dim=1) * (1.0 + 1.0 / self.t)    # [N, 1]
            var = raw_var.clamp_min(self.var_floor)
            with torch.no_grad():
                if greedy:
                    a = mu
                elif self.draw_mode == "gauss":              # magnitude-matched Gaussian
                    a = mu + var.sqrt() * torch.randn_like(mu)
                else:
                    m = int(torch.randint(self.t, (1,)).item())
                    a = y[:, m, :]                           # execute a REAL internal sample
                    # floor dither: when the ensemble spread falls below the absolute
                    # floor, top up the executed action's variance so that the
                    # behavioral distribution matches the score/ratio model N(mu, var)
                    extra = (var - raw_var).clamp_min(0.0)
                    a = a + extra.sqrt() * torch.randn_like(a)
            if self.var_override is not None:
                var = torch.full_like(var, self.var_override)
            score = (a - mu) / var                           # d log pi / d mu, marginal var
        else:
            sigma = self.sigma_explore                       # fixed exploration std
            with torch.no_grad():
                a = mu if greedy else mu + sigma * torch.randn_like(mu)
            score = (a - mu) / (sigma ** 2)                  # [N, 1]  d log pi / d mu
        force = (self.force_max * a).clamp(-self.force_max, self.force_max)
        return StepData(obs=obs, p=mu.detach(), action=force.detach(), logp=None,
                        d=[t.clone() for t in cap.d],
                        z=[t.clone() for t in cap.z],
                        y_samples=y.clone(), value=None, v_samples=None,
                        score=score.detach())
