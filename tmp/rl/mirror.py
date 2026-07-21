"""tmp/rl.mirror -- online weight-mirror state: EMA + Kolen-Pollack tracking (§20.4).

Online (N=1) the per-step covariance mirror is high variance, so we (a) EMA-smooth the
per-step `cov_weight` measurements across steps, and (b) after each actor update shift the
mirror by the KNOWN weight decrement (Kolen-Pollack PREDICT), so it tracks the moving
weights instead of lagging them.  Both together let the single-shot mirror of Step A be
used inside a learning loop (idea_rl.md §17.2-4).
"""
from __future__ import annotations

from data_nce.fncl.train import cov_weight


class MirrorState:
    def __init__(self, beta: float = 0.99, pool: bool = False, value_head: bool = False):
        self.beta = beta
        self.pool = pool
        self.value_head = value_head
        self.W_out = None       # actor readout mirror
        self.W_hidden = None    # body mirrors (per hidden layer l>=1)
        self.W_vout = None      # value readout mirror (unified critic, Task #1)

    def observe(self, policy, step):
        """EMA-update the mirrors from this pass's forward covariance."""
        z, d = step.z, step.d
        n_hidden = len(policy.crossings)
        meas_out = cov_weight(step.y_samples, z[-1], pool=self.pool)
        meas_hidden = {l: cov_weight(d[l], z[l - 1], pool=self.pool)
                       for l in range(1, n_hidden)}
        meas_v = cov_weight(step.v_samples, z[-1], pool=self.pool) if self.value_head else None
        if self.W_out is None:
            self.W_out, self.W_hidden, self.W_vout = meas_out, meas_hidden, meas_v
        else:
            b = self.beta
            self.W_out = b * self.W_out + (1.0 - b) * meas_out
            for l in meas_hidden:
                self.W_hidden[l] = b * self.W_hidden[l] + (1.0 - b) * meas_hidden[l]
            if self.value_head:
                self.W_vout = b * self.W_vout + (1.0 - b) * meas_v

    def kp_predict(self, shift):
        """Subtract the applied weight decrements (param -= decrement) so the mirror,
        which mirrors the real weight, moves with it. `shift` keys: 'out', 'vout',
        and hidden l."""
        if self.W_out is not None and "out" in shift:
            self.W_out = self.W_out - shift["out"]
        if self.W_vout is not None and "vout" in shift:
            self.W_vout = self.W_vout - shift["vout"]
        for l in list(self.W_hidden or {}):
            if l in shift:
                self.W_hidden[l] = self.W_hidden[l] - shift[l]
