"""tmp/rl.policy -- Bernoulli CartPole policy on a sample-level NNN (idea_rl.md §20.2).

The policy body is a `SimpleNNNSample` with structure [obs_dim, H, H, 1]; the single
linear readout is the action LOGIT. Running T internal noise samples gives per-sample
logits o^(m); the firing probability is p = sigmoid(mean_m o^(m)) and the executed
action is a ~ Bernoulli(p) (exactly one env action -- §2.1).

`rollout_step` runs ONE grad-enabled forward and returns both:
  * the detached per-sample internals (d[l], z[l], y_samples) captured by fncl.Capture,
    used by the forward-only credit estimators (tmp/rl/credit.py), and
  * a differentiable log pi(a|s) whose .backward() gives the autograd gold gradient.
Both come from the SAME noise realization, so gold and estimator are compared on
identical samples.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from . import constants  # noqa: F401  (bootstraps sys.path)
from nnn import model
from data_nce.fncl.network import Capture


@dataclass
class StepData:
    """Everything a single policy step exposes to the credit estimators."""
    obs: torch.Tensor          # [N, obs_dim]
    p: torch.Tensor            # [N, 1]  firing probability
    action: torch.Tensor       # [N, 1]  sampled {0,1}
    logp: torch.Tensor         # scalar (sum over batch) log pi(a|s), WITH autograd graph
    d: list                    # per hidden layer: [N, T, H]  clean pre-activation
    z: list                    # per hidden layer: [N, T, H]  crossing output
    y_samples: torch.Tensor    # [N, T, 1]  actor readout per-sample logit (pre-ensemble)
    value: torch.Tensor        # [N, 1]  value estimate V(s)
    v_samples: torch.Tensor    # [N, T, 1]  value readout per-sample (for the forward mirror)
    score: torch.Tensor = None # [N, D_out]  top-level output-space score (continuous policy);
                               # if set, the credit recursion uses it as err_out


class CartPolePolicy(nn.Module):
    def __init__(self, obs_dim: int = 4, hidden: int = constants.DEF_HIDDEN,
                 std: float = constants.DEF_STD, h: float = constants.DEF_H,
                 t: int = constants.DEF_T, n_hidden_layers: int = 2):
        super().__init__()
        self.structure = [obs_dim] + [hidden] * n_hidden_layers + [1]
        self.net = model.SimpleNNNSample(structure=self.structure, std=std, h=h, t=t,
                                         output_bias=True)
        self.t = t
        self.hidden = hidden
        # Per-unit noise field (§7): a list of one [H] std vector per hidden layer.
        # None -> uniform scalar std.  Set it to spatially allocate noise across units
        # (the field/option mechanism); the crossing noise layer broadcasts a [H] std.
        self.field = None
        # Value readout on the SAME shared hidden as the actor logit.  In the unified
        # critic (Task #1) its hidden credit flows through the same forward mirror as the
        # actor's; in the Step-B minimal critic it is fit by external linear TD instead.
        self.value_head = nn.Linear(hidden, 1)

    @property
    def crossings(self):
        return self.net.gaussian_crossing

    @property
    def fcs(self):
        return self.net.fcs

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        """Last-hidden ensemble-mean features [N, H] (detached), for the TD critic."""
        cap = Capture(self.net)
        try:
            with torch.no_grad():
                self.net(obs, stds=self.field)
            feat = cap.z[-1].mean(dim=1).clone()
        finally:
            cap.remove()
        return feat

    def rollout_step(self, obs: torch.Tensor, greedy: bool = False,
                     rng: torch.Generator | None = None) -> StepData:
        """One grad-enabled forward. `obs` is [N, obs_dim]."""
        cap = Capture(self.net)
        try:
            logit_mean = self.net(obs, stds=self.field)    # [N, 1] ensemble-mean logit
        finally:
            cap.remove()
        p = torch.sigmoid(logit_mean)                      # [N, 1]

        with torch.no_grad():
            if greedy:
                action = (p > 0.5).float()
            else:
                u = torch.rand(p.shape, generator=rng) if rng is not None else torch.rand_like(p)
                action = (u < p).float()

        eps = 1e-6
        logp = (action * torch.log(p + eps)
                + (1.0 - action) * torch.log(1.0 - p + eps)).sum()

        with torch.no_grad():
            v_samples = self.value_head(cap.z[-1])          # [N, T, 1]
            value = v_samples.mean(dim=1)                   # [N, 1]

        return StepData(obs=obs, p=p.detach(), action=action, logp=logp,
                        d=[t.clone() for t in cap.d],
                        z=[t.clone() for t in cap.z],
                        y_samples=cap.y_samples.clone(), value=value,
                        v_samples=v_samples.clone())

    def value_of(self, obs: torch.Tensor) -> float:
        """V(s) for a next-state bootstrap (no capture, no grad)."""
        cap = Capture(self.net)
        try:
            with torch.no_grad():
                self.net(obs, stds=self.field)
                v = self.value_head(cap.z[-1]).mean(dim=1)
        finally:
            cap.remove()
        return float(v.item())
