"""tmp/rl.credit -- forward-only policy-score credit, generated in the NNN noise path.

All estimators start from the SAME top-level output credit, the logit error

    err_out = a - p          (= d log pi(a|s) / d o,  the Bernoulli score at the logit)

and differ only in HOW that scalar is distributed to the hidden weights:

  cov_jac        : forward-covariance weight mirror (fncl.cov_weight) + KDE crossing
                   slope (fncl.kde_slope).  NO transposed-weight read.   <- the method
  true_transpose : same recursion with the REAL forward weights (mirror error = 0).
                   Upper bound; isolates mirror quality when compared with cov_jac.
  node_pert      : node-perturbation credit -- regress each unit's fluctuation DIRECTLY
                   on the per-sample output logit (fncl.covariance_credit), no mirror and
                   no cross-layer recursion (the §18-C baseline).

`gold_grad` returns the autograd d log pi / dW of the very same forward pass.

Every estimator returns a dict  {(layer_index, "weight"|"bias"): tensor}, sharing the
canonical parameter order of `param_keys`, so metrics.py can flatten them consistently.
Stage-1 (§20) is the ONLINE regime N = 1: call these on one state at a time.
"""
from __future__ import annotations

import torch

from . import constants  # noqa: F401
from data_nce.fncl.train import cov_weight, covariance_credit
from data_nce.fncl.network import kde_slope


def param_keys(policy):
    """Canonical ordering of the policy-body parameters (readout included)."""
    keys = []
    for l in range(len(policy.fcs)):
        keys.append((l, "weight"))
        keys.append((l, "bias"))
    return keys


# ---------------------------------------------------------------------------
# Autograd gold: d log pi(a|s) / dW of this forward pass.
# ---------------------------------------------------------------------------
def gold_grad(policy, step):
    for fc in policy.fcs:
        if fc.weight.grad is not None:
            fc.weight.grad = None
        if fc.bias is not None and fc.bias.grad is not None:
            fc.bias.grad = None
    step.logp.backward()
    grads = {}
    for l, fc in enumerate(policy.fcs):
        grads[(l, "weight")] = fc.weight.grad.detach().clone()
        grads[(l, "bias")] = (fc.bias.grad.detach().clone()
                              if fc.bias is not None else torch.zeros(fc.out_features))
    return grads


# ---------------------------------------------------------------------------
# Shared recursion.  Given the (possibly estimated) forward weights, distribute the
# output error err_out through the layers.  Only the WEIGHT SOURCE distinguishes
# cov_jac (mirror) from true_transpose (real W); the recursion is identical.
# ---------------------------------------------------------------------------
def slopes(policy, step):
    d = step.d
    n_hidden = len(policy.crossings)
    slope_full = [kde_slope(policy.crossings[l], d[l]) for l in range(n_hidden)]  # [N,T,H]
    slope_mean = [s.mean(dim=1) for s in slope_full]                              # [N,H]
    return slope_full, slope_mean


def _recurse_body(policy, step, err_out, W_out, W_hidden, slope_full, slope_mean):
    """Propagate an output-space signal `err_out` through the shared body via the mirror
    recursion (no transposed weights).  Returns the hidden-layer grads only; the readout
    grad is head-specific and added by the caller.  This body recursion is IDENTICAL for
    the actor score and the value error -- the unification (Task #1)."""
    z = step.z
    n_hidden = len(policy.crossings)
    N, T, _ = z[0].shape
    a_jac = [None] * n_hidden
    a_jac[-1] = err_out @ W_out                            # [N,1]@[1,H] -> [N,H]
    for l in range(n_hidden - 2, -1, -1):
        a_jac[l] = (a_jac[l + 1] * slope_mean[l + 1]) @ W_hidden[l + 1]   # [N, H_l]

    z_prev = [step.obs.unsqueeze(1).expand(N, T, step.obs.shape[1]), *z[:-1]]
    grads = {}
    for l in range(n_hidden):
        delta_hat = a_jac[l].unsqueeze(1) * slope_full[l]  # [N,T,H]
        grads[(l, "weight")] = torch.einsum("nto,nti->oi", delta_hat, z_prev[l]) / (N * T)
        grads[(l, "bias")] = delta_hat.mean(dim=(0, 1))
    return grads


def recursion_from_weights(policy, step, W_out, W_hidden,
                           slope_full=None, slope_mean=None):
    """Actor policy-score credit.  Top-level err is the output-space score:
    the Bernoulli score (a - p) by default, or `step.score` = Sigma^-1 (a - mu) for the
    continuous Gaussian-from-samples policy (§3.1)."""
    if slope_full is None:
        slope_full, slope_mean = slopes(policy, step)
    err_out = step.score if step.score is not None else (step.action - step.p)  # [N, D_out]
    grads = _recurse_body(policy, step, err_out, W_out, W_hidden, slope_full, slope_mean)
    n_hidden = len(policy.crossings)
    N = step.z[0].shape[0]
    z_last_mean = step.z[-1].mean(dim=1)                   # [N, H]
    grads[(n_hidden, "weight")] = (err_out.transpose(0, 1) @ z_last_mean) / N   # [1, H]
    grads[(n_hidden, "bias")] = err_out.mean(dim=0)        # [1]
    return grads


def value_grad(policy, step, W_hidden, W_vout, slope_full=None, slope_mean=None):
    """Value gradient dV/dW through the SAME forward mirror as the actor (Task #1).

    V = mean_T (W_v z[-1]); the top-level signal into the body is dV/dV = 1 propagated by
    the value-head mirror `W_vout`.  Returns body grads plus the value-head grads under
    keys ('v','weight'/'bias').  The caller scales by the TD error (semi-gradient TD)."""
    if slope_full is None:
        slope_full, slope_mean = slopes(policy, step)
    N = step.z[0].shape[0]
    ones = torch.ones(N, 1)
    grads = _recurse_body(policy, step, ones, W_vout, W_hidden, slope_full, slope_mean)
    z_last_mean = step.z[-1].mean(dim=1)                   # [N, H]
    grads[("v", "weight")] = z_last_mean.mean(dim=0, keepdim=True)   # dV/dW_v  [1, H]
    grads[("v", "bias")] = torch.ones(1)                            # dV/db_v
    return grads


def value_mirror(policy, step, pool=False):
    return cov_weight(step.v_samples, step.z[-1], pool=pool)


def sigma_grad(policy, step, W_out, W_hidden, slope_full=None, slope_mean=None):
    """Per-unit NOISE-FIELD eligibility psi_sigma (§10).  Shares the SAME forward credit
    delta = g * phi' as the weight eligibility (the recursion below is identical), but the
    LOCAL coordinate is -d/sigma instead of z_prev (§10.2):

        psi_W[i,j]   = delta_i * z_prev_j
        psi_sigma[k] = delta_k * (-d_k / sigma_k)

    The -d/sigma factor is the derivative dphi_bar/dsigma = -(d/sigma) phi_bar'(d), EXACT in
    the coupled crossing-width regime (h proportional to sigma, §17.2-7) and approximate for
    a fixed h.  Returns {layer: [H]}."""
    z, d = step.z, step.d
    n_hidden = len(policy.crossings)
    err_out = (step.action - step.p)                       # [N, 1]
    if slope_full is None:
        slope_full, slope_mean = slopes(policy, step)

    a_jac = [None] * n_hidden
    a_jac[-1] = err_out @ W_out
    for l in range(n_hidden - 2, -1, -1):
        a_jac[l] = (a_jac[l + 1] * slope_mean[l + 1]) @ W_hidden[l + 1]

    psi = {}
    for l in range(n_hidden):
        delta_hat = a_jac[l].unsqueeze(1) * slope_full[l]  # [N,T,H]  = g * phi'
        sigma_l = policy.field[l].detach().to(d[l].dtype).view(1, 1, -1)
        local = -d[l] / (sigma_l + 1e-6)                   # -d/sigma  [N,T,H]
        psi[l] = (delta_hat * local).mean(dim=(0, 1))      # [H]
    return psi


def sigma_grad_forward(policy, step, W_out, W_hidden, slope_full=None, slope_mean=None):
    """Per-unit noise-field eligibility WITHOUT the -d/sigma identity: estimate the local
    dz/dsigma from the crossing's OWN noise (a local grad re-run, exactly parallel to how
    `kde_slope` gets dz/dd), then weight by the recursion's unit credit a_jac.  No
    transposed weights, no readout -- the general, distribution-free form of psi_sigma.
    Returns {layer: [H]}."""
    n_hidden = len(policy.crossings)
    N, T, _ = step.z[0].shape
    err_out = (step.action - step.p)
    if slope_full is None:
        slope_full, slope_mean = slopes(policy, step)

    a_jac = [None] * n_hidden
    a_jac[-1] = err_out @ W_out
    for l in range(n_hidden - 2, -1, -1):
        a_jac[l] = (a_jac[l + 1] * slope_mean[l + 1]) @ W_hidden[l + 1]

    psi = {}
    for l in range(n_hidden):
        cl = policy.crossings[l]
        sig = policy.field[l].detach().clone().requires_grad_(True)
        d_l = step.d[l].detach()
        with torch.enable_grad():
            z = cl(d_l, std=sig)                            # crossing with sigma as leaf
            gout = a_jac[l].unsqueeze(1).expand_as(z)       # dL/dz broadcast over T
            z.backward(gout)                                # sig.grad = sum dL/dz * dz/dsig
        psi[l] = sig.grad.detach() / (N * T)                # ~ dL/dsigma per unit  [H]
    return psi



def mirror_weights(policy, step, pool=False):
    """Single-shot weight mirrors from THIS pass (used by Step A / cold start)."""
    z, d = step.z, step.d
    n_hidden = len(policy.crossings)
    W_out = cov_weight(step.y_samples, z[-1], pool=pool)          # [1, H]
    W_hidden = {l: cov_weight(d[l], z[l - 1], pool=pool) for l in range(1, n_hidden)}
    return W_out, W_hidden


def true_weights(policy, step):
    n_hidden = len(policy.crossings)
    return (policy.fcs[-1].weight.detach(),
            {l: policy.fcs[l].weight.detach() for l in range(1, n_hidden)})


def cov_jac_grad(policy, step):
    W_out, W_hidden = mirror_weights(policy, step)
    return recursion_from_weights(policy, step, W_out, W_hidden)


class MirrorEMA:
    """Persistent EMA weight mirror with Kolen-Pollack tracking (fix B, idea_rl.md §23.8).

    `cov_jac_grad` re-estimates the mirror from a SINGLE state's T samples at every step
    (high variance).  The supervised pipeline that validated cov_jac (CovJacTrainer,
    consolidation) instead keeps a PERSISTENT mirror: each measurement is blended in by
    EMA, and the applied weight updates are added so the mirror tracks the moving true
    weights (Kolen-Pollack).  This class ports that design to the RL loop: steady-state
    mirror variance drops to ~beta/(2-beta) of single-shot (beta=0.1 -> ~1/19).
    """

    def __init__(self, beta=0.1):
        self.beta = beta
        self.W_out = None
        self.W_hidden = None

    def grad(self, policy, step):
        """Blend this pass's measurement into the stored mirror, then run the credit
        recursion with the BLENDED mirror.  Same call shape as `cov_jac_grad`."""
        W_out, W_hidden = mirror_weights(policy, step)
        if self.W_out is None:
            self.W_out = W_out.clone()
            self.W_hidden = {l: w.clone() for l, w in W_hidden.items()}
        else:
            self.W_out += self.beta * (W_out - self.W_out)
            for l, w in W_hidden.items():
                self.W_hidden[l] += self.beta * (w - self.W_hidden[l])
        return recursion_from_weights(policy, step, self.W_out, self.W_hidden)

    def snapshot_weights(self, policy):
        """Call BEFORE applying weight updates; pairs with `kp_track`."""
        n_hidden = len(policy.crossings)
        self._prev = {l: policy.fcs[l].weight.detach().clone()
                      for l in range(1, n_hidden + 1)}

    def kp_track(self, policy):
        """Kolen-Pollack: add the just-applied update of each mirrored weight to the
        mirror, so it keeps tracking the moving true weights between measurements."""
        if self.W_out is None:
            return
        n_hidden = len(policy.crossings)
        for l in range(1, n_hidden):
            self.W_hidden[l] += policy.fcs[l].weight.detach() - self._prev[l]
        self.W_out += policy.fcs[n_hidden].weight.detach() - self._prev[n_hidden]


def true_transpose_grad(policy, step):
    W_out, W_hidden = true_weights(policy, step)
    return recursion_from_weights(policy, step, W_out, W_hidden)


# ---------------------------------------------------------------------------
# Node-perturbation baseline (§18-C): flat, no mirror, no recursion.
# ---------------------------------------------------------------------------
def node_pert_grad(policy, step):
    d, z = step.d, step.z
    n_hidden = len(policy.crossings)
    N, T, _ = z[0].shape
    err_out = (step.action - step.p)                       # [N,1]  score at the logit

    # Correlate EVERY unit (any layer) DIRECTLY with the output logit o^(m), which
    # fluctuates across the T internal noise draws -> dO/dz_i by single-variable
    # regression, no weight-mirror recursion.  Confounded by crosstalk (all units
    # fluctuate); variance grows with width -- the scaling weakness cov_jac avoids.
    o = step.y_samples.squeeze(-1)                         # [N, T]
    slope_full, _ = slopes(policy, step)

    z_prev = [step.obs.unsqueeze(1).expand(N, T, step.obs.shape[1]), *z[:-1]]
    grads = {}
    for l in range(n_hidden):
        g_bcast, _ = covariance_credit(z[l], o, credit="per_input")   # ~ dO/dz[l]  [N,1,H]
        delta_hat = err_out.unsqueeze(1) * g_bcast * slope_full[l]     # d log pi / dd[l]
        gW = torch.einsum("nto,nti->oi", delta_hat, z_prev[l]) / (N * T)
        gb = delta_hat.mean(dim=(0, 1))
        grads[(l, "weight")] = gW
        grads[(l, "bias")] = gb
    z_last_mean = z[-1].mean(dim=1)
    grads[(n_hidden, "weight")] = (err_out.transpose(0, 1) @ z_last_mean) / N
    grads[(n_hidden, "bias")] = err_out.mean(dim=0)
    return grads
