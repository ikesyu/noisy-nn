"""tmp/rl.agents -- unified credit interface for the four Step-B conditions (§20.7).

Every agent exposes the SAME `logpi_grad(step)` -> {param_key: grad} estimate of
d log pi(a|s)/dW; the training loop (tmp/rl/train.py) applies the identical eligibility
trace + TD modulation + ManualOpt step to all of them.  Only the credit source differs:

    cov_jac        -- forward mirror (EMA + KP), the method
    true_transpose -- real transposed weights (upper bound; isolates mirror cost)
    node_pert      -- flat node-perturbation baseline (§18-C)
    backprop       -- autograd d log pi/dW (standard actor-critic reference)

This keeps the comparison about the ACTOR credit only; the critic is the same minimal
TD value head for every agent.
"""
from __future__ import annotations

from . import credit as C
from .mirror import MirrorState

KINDS = ("cov_jac", "true_transpose", "node_pert", "backprop")


class Agent:
    def __init__(self, policy, kind: str, mirror_beta: float = 0.99,
                 mirror_pool: bool = False):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind}")
        self.policy = policy
        self.kind = kind
        self.mirror = MirrorState(mirror_beta, mirror_pool) if kind == "cov_jac" else None

    def logpi_grad(self, step):
        if self.kind == "cov_jac":
            self.mirror.observe(self.policy, step)
            return C.recursion_from_weights(self.policy, step,
                                            self.mirror.W_out, self.mirror.W_hidden)
        if self.kind == "true_transpose":
            return C.true_transpose_grad(self.policy, step)
        if self.kind == "node_pert":
            return C.node_pert_grad(self.policy, step)
        if self.kind == "backprop":
            return C.gold_grad(self.policy, step)

    def post_actor_update(self, applied):
        """KP tracking for the mirror (cov_jac only). `applied[(l,'weight')]` is the
        decrement applied to fcs[l].weight."""
        if self.mirror is None:
            return
        n_hidden = len(self.policy.crossings)
        shift = {l: applied[(l, "weight")] for l in range(1, n_hidden)}
        shift["out"] = applied[(n_hidden, "weight")]
        self.mirror.kp_predict(shift)
