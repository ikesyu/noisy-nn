"""tmp/rl.critic -- external value MLP (standard backprop).

This is the deliberate integration compromise (方向1): the ACTOR credit stays NNN
(forward mirror, no transposed weights), while the CRITIC is an ordinary MLP trained by
backprop TD/GAE, which removes the critic collapse that blocked the internal value head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from nnn import model
from data_nce.fncl.network import Capture
from .policy import StepData


class ValueMLP(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NNNCritic(nn.Module):
    """Value function as an NNN, trained by the SAME forward mirror (cov_jac) as the actor
    -- no backprop anywhere.  V(s) = mean_T of a linear readout of the NNN body; the value
    error (V - GAE_return) is the top-level score fed to the cov_jac recursion (regression
    to the returns, which is exactly what cov_jac/FNCL was validated for).  With this the
    whole actor+critic is one forward-only NNN system (§6, §17.2-6, §20.17)."""

    def __init__(self, obs_dim, hidden=64, std=0.6, h=0.15, t=64, n_hidden_layers=2):
        super().__init__()
        structure = [obs_dim] + [hidden] * n_hidden_layers + [1]
        self.net = model.SimpleNNNSample(structure=structure, std=std, h=h, t=t,
                                         output_bias=True)
        self.t = t
        self.hidden = hidden

    @property
    def crossings(self):
        return self.net.gaussian_crossing

    @property
    def fcs(self):
        return self.net.fcs

    def value_step(self, obs):
        """Returns V(s) (float) and a StepData carrying the forward internals for cov_jac."""
        cap = Capture(self.net)
        try:
            v_mean = self.net(obs)                           # [N, 1] ensemble-mean value
        finally:
            cap.remove()
        step = StepData(obs=obs, p=None, action=None, logp=None,
                        d=[t.clone() for t in cap.d],
                        z=[t.clone() for t in cap.z],
                        y_samples=cap.y_samples.clone(), value=v_mean.detach(),
                        v_samples=None, score=None)
        return float(v_mean.item()), step
