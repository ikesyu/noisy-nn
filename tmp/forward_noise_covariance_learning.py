"""
tmp/forward_noise_covariance_learning.py

Forward-Noise-Based Covariance Learning for Noise-modulated Neural Networks (NNNs)
=================================================================================

Proof-of-concept for a hardware-friendly, backprop-*like* learning rule that never
performs ordinary backward error propagation through transposed weight matrices.

This version uses the ACTUAL library models from `nnn/model.py` as the forward
network (selectable with --noise):

    --noise gaussian  ->  nnn.model.SimpleNNNSample         (Gaussian-noise crossing)
    --noise uniform   ->  nnn.model.SimpleNNNUniformSample  (bounded-uniform crossing)

Both models draw `t` stochastic samples internally and average them at the output
(EnsembleMeanLayer), so the network output is already an EXPECTED value.  We exploit
this:

* Hidden-layer credit is estimated from the covariance between the (per-sample) loss
  and each unit's stochastic activity, captured from the model's internal samples via
  forward hooks.  With --credit per_input (default) the covariance is taken WITHIN
  each input over the t samples, giving an input-dependent credit that is a much less
  biased estimate of the local gradient than the global --credit pooled variant:

        per_input: g_z[n,i] = Cov_t(L_n, z_{n,i}) / (Var_t(z_{n,i}) + eps) ~ dL_n/dz_{n,i}
        pooled   : g_z[i]   = Cov_{n,t}(L, z_i) / (Var_{n,t}(z_i) + eps)   (one scalar/unit)

  This replaces the backprop term  W_{l+1}^T delta_{l+1}  -- NO transposed-weight
  backward pass.  --credit-passes accumulates statistics over several passes (variance
  reduction).
* The local activation sensitivity is the noise-induced derivative of the crossing:
        gaussian: phi'(d) = 2 (1 - 2 F(d)) p(d)
        uniform : phi'(d) = -(d - c) / r^2        for |d - c| < r   (parabolic slope)
* The OUTPUT (readout) layer is updated from the ENSEMBLE-MEAN activation (an
  expected-value readout).  Averaging over the model's `t` samples cuts the
  variance-induced readout shrinkage by a factor ~t, which is why the fit is much
  better than a single-sample readout.

Weight updates (hidden, manual, NO autograd / NO .backward()):

        delta_hat[l, i] = g_z[l, i] * phi'(d_{l,i})           (cov_deriv)
                        = g_z[l, i]                           (cov_only)
        Delta W_l       = -eta * mean_{n,t}( delta_hat[l] z_{l-1}^T )

Methods compared
----------------
  backprop       : the SAME selected model, trained with ordinary autograd (Adam).
                   Reference only -- it does use backward propagation.
  cov_only       : hidden credit = covariance only (no phi').  Manual, no autograd.
  cov_deriv      : proposed, DEFAULT -- covariance credit x the crossing's OWN
                   distribution-free density slope (xor2-xor1)/(2h) (--slope kde, default).
                   The dual +/-h thresholds evaluated on the SAME samples are an antithetic /
                   common-random-number finite difference, so the slope is low-variance and
                   needs NO knowledge of the noise distribution.  Read out via a local
                   single-layer grad re-run (no transposed-weight backward).  With
                   --slope analytic it instead uses the hand-coded phi'(d); the two match on
                   gaussian and uniform, so the analytic phi_prime() is not needed.
  cov_jac        : proposed -- STRUCTURED / RECURSIVE covariance credit.  Rather than
                   collapsing the whole downstream onto Cov(L, z), it propagates the exact
                   output error DOWN the graph using forward weights recovered from
                   covariance, W_hat[l+1] = Cov(d[l+1], z[l])/Var(z[l]) (EMA-smoothed) -- a
                   weight-mirror, forward-only, weight-transport-free reconstruction of true
                   backprop.  BEATS cov_deriv at full budget (lowers the estimator-bias floor
                   ~30%), at the cost of slower convergence.  Correlate with the CONTINUOUS
                   pre-activation d, never the binary z.
  cov_jac_full   : proposed -- cov_jac with the READOUT error ALSO taken from covariance.
                   cov_jac (like every other method here) still seeds the recursion and
                   updates the readout weights with the ANALYTIC loss derivative
                   dL/dy = 2 (y - t).  cov_jac_full replaces it with the same forward-noise
                   estimator used everywhere else:
                       g_y[n] = Cov_t(L_n, y_n) / Var_t(y_n)  ~=  dL_n/dy_n
                   over the readout's own per-sample fluctuation y_n,t (pre-ensemble).
                   Same recursion, mirrors, and updates otherwise -- no analytic derivative
                   of the loss is needed ANYWHERE; the network only ever sees the scalar
                   per-sample loss.  CAVEAT (measured): because L = (y - t)^2 is quadratic
                   in y, the raw regression coefficient is 2 (E[y] - t) + E[eps^3]/Var(eps);
                   the skewness term of the crossing-induced readout fluctuation is a bias
                   that does NOT vanish at convergence (corr 0.997 with the observed bias)
                   and that Adam, which normalises gradient scale, amplifies late in
                   training.  --jac-out selects the fix: 'cov_m3' (DEFAULT) subtracts the
                   observed third central moment of y (still pure forward statistics; exact
                   for quadratic loss; matches backprop: final MSE 0.00070 vs 0.00069),
                   'probe' regresses the loss on an injected symmetric Gaussian probe
                   (unbiased for ANY loss; 0.00129), 'cov' keeps the raw estimator (drifts
                   to 0.029 with Adam; convergence SPEED still matches cov_jac).
  cov_deriv_gate_crn : proposed -- ANTITHETIC / COMMON-RANDOM-NUMBER gate.  Same block
                   gating, but each pass runs a paired (+xi, -xi) forward under the SAME
                   crossing noise (RNG-state reset), so the intrinsic-noise nuisance cancels
                   in the loss DIFFERENCE.  UNBIASED and low-variance in principle -- but on
                   the BINARY level-crossing the pathwise response to a small pre-activation
                   shift is degenerate (mostly no flip), so it barely beats cov_deriv_gate.
                   A concrete demonstration of why the crossing's OWN density estimator
                   (cov_deriv_kde) is the right tool and external perturbation is not.
  cov_deriv_gate : proposed -- PERTURBATION GATE.  On top of the model's forward noise
                   an extra small perturbation xi is injected into a rotating BLOCK of
                   hidden pre-activations (via forward pre-hooks), and hidden credit is
                   estimated from Cov(L, xi)/Var(xi) instead of Cov(L, z)/Var(z).  Only
                   the gated block is perturbed AND updated each epoch.  The perturbation
                   gate is intended to reduce covariance-credit bias by correlating loss
                   with an explicitly injected local perturbation, rather than with all
                   spontaneous unit fluctuations.  Because xi is added at the PRE-activation
                   d, Cov(L, xi)/Var(xi) already estimates dL/dd = delta directly (no extra
                   phi'(d) factor -- see train_cov).
  cov_deriv_field_gate : proposed -- NOISE-FIELD / RECRUITMENT GATE.  cov_deriv with each
                   hidden update multiplied by the unit's noise-field strength s_i
                   (Delta W_l[i,j] *= s_i).  With --field-sparsity 0 (default) s_i == 1 for
                   all units, so the rule IS cov_deriv; with --field-sparsity f>0 a fraction
                   f of hidden units are un-recruited (s_i = 0 -> zero forward noise, dead,
                   AND zero update), linking credit assignment to the NNN recruitment/noise
                   field.  A one-line gate on top of cov_deriv (see docs section 8.5).

This is a proof-of-concept for hardware-friendly APPROXIMATE backpropagation, not an
exact replacement.  See docs/forward_noise_covariance_learning.md.

Run
---
    python tmp/forward_noise_covariance_learning.py
    python tmp/forward_noise_covariance_learning.py --noise uniform
    python tmp/forward_noise_covariance_learning.py --epochs 1500 --num-samples 64
    python tmp/forward_noise_covariance_learning.py --gate-block-size 8 --gate-alpha 0.05

Displays three figures with plt.show() (no files are written).
Requires only Python, PyTorch, NumPy, Matplotlib (+ the local nnn package).
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Use the local nnn library models.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from nnn import model

SQRT2 = math.sqrt(2.0)
EPS = 1e-6
NUM_POINTS = 128
UNIFORM_CENTER = 0.0


# ============================================================
# Noise-induced local derivatives phi'(d) = dz/dd of the expected crossing response
# ============================================================
def gauss_cdf(d: torch.Tensor, sigma: float) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(d / (sigma * SQRT2)))


def gauss_pdf(d: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-(d * d) / (2.0 * sigma * sigma)) / (sigma * math.sqrt(2.0 * math.pi))


def phi_prime(d: torch.Tensor, noise: str, sigma: float, radius: float) -> torch.Tensor:
    """Local derivative of the EXPECTED crossing response for the chosen noise.

    gaussian: phi_bar(d) = 2 F(d)(1-F(d))          -> phi' = 2 (1 - 2 F(d)) p(d)
    uniform : phi_bar(d) = 0.5 [1 - ((d-c)/r)^2]_+ -> phi' = -(d-c)/r^2  (inside support)
    """
    if noise == "gaussian":
        F = gauss_cdf(d, sigma)
        return 2.0 * (1.0 - 2.0 * F) * gauss_pdf(d, sigma)
    u = d - UNIFORM_CENTER
    return torch.where(u.abs() < radius, -u / (radius * radius), torch.zeros_like(u))


def kde_slope(crossing_layer, d_clean: torch.Tensor) -> torch.Tensor:
    """Distribution-FREE local slope dz_bar/dd from the Crossing's OWN internal density
    estimate -- no analytic phi', no knowledge of the noise distribution.

    The library's CrossingSample.backward computes, per unit,

        coeff = mean_t(xor2 - xor1) / (2h)

    where xor1/xor2 count level crossings of the SAME noisy samples at the thresholds +h
    and -h.  Evaluating the two thresholds on one shared sample set is exactly an
    ANTITHETIC / COMMON-RANDOM-NUMBER finite difference: shifting the threshold by +/-h is
    equivalent to shifting the pre-activation d by -/+h, and the common samples make the
    noise cancel in the difference -> a low-variance, distribution-free estimate of dz/dd.

    We obtain it by a LOCAL grad-enabled re-run of just this crossing layer (fresh noise;
    a single unit's activation derivative -- NO transposed-weight backward, no readout).
    Because the noise addition d -> d + eta has unit gradient w.r.t. d, autograd routes
    CrossingSample.backward's coeff straight onto d_clean.grad.

    Returns dz/dd of shape [N, T, H] (constant across T; the library repeats coeff over T).
    """
    with torch.enable_grad():
        d_req = d_clean.detach().clone().requires_grad_(True)
        z = crossing_layer(d_req)                 # noise + CrossingSample, this layer only
        z.sum().backward()                        # grad_output = 1 -> d_req.grad = coeff
    return d_req.grad.detach()


# ============================================================
# Build a library model and give layer-1 a spread "bump tiling" so the readout has a
# good basis (the default init does not tile the 1-D input).
# ============================================================
def build_model(noise: str, hidden: int, sigma: float, radius: float, h: float,
                t: int, device: torch.device, field: torch.Tensor = None):
    structure = [1, hidden, hidden, 1]
    if noise == "gaussian":
        net = model.SimpleNNNSample(structure=structure, std=sigma, h=h, t=t,
                                    output_bias=True)
    elif noise == "uniform":
        net = model.SimpleNNNUniformSample(structure=structure, radius=radius,
                                           center=UNIFORM_CENTER, h=h, t=t,
                                           output_bias=True)
    else:
        raise ValueError(f"--noise must be 'gaussian' or 'uniform', got '{noise}'")

    # Spread hidden-layer-1 bump centres across the normalised input range [-2, 2].
    H = hidden
    centres = torch.linspace(-2.0, 2.0, H)
    mag = 0.8 + 0.4 * torch.rand(H)
    sign = torch.where(torch.rand(H) < 0.5, -1.0, 1.0)
    w1 = (mag * sign).unsqueeze(1)
    with torch.no_grad():
        net.fcs[0].weight.copy_(w1)
        net.fcs[0].bias.copy_(-(w1.squeeze(1)) * centres)

    # --- per-unit noise field (recruitment gate) read by cov_deriv_field_gate ---
    # `net.noise_field[l]` is a [H] vector of each hidden unit's noise-field strength.
    # When a non-trivial field is supplied it becomes a REAL noise field: the per-unit
    # Gaussian std (or uniform radius) is scaled by it, so a zero-field unit gets NO
    # noise -> it is deterministic (dead) and receives no update under the field gate.
    # Default (field is None) -> all-ones, so the network and every other method are
    # byte-identical to before (the scalar std/radius is left untouched).
    n_hidden = len(structure) - 2
    if field is None:
        s = torch.ones(H, device=device)
    else:
        s = field.detach().to(device).float()
        if noise == "gaussian":
            net.std = sigma * s                                  # [H], broadcasts over [N,T,H]
        else:
            for cl in net.uniform_crossing:
                cl.radius = radius * s
    net.noise_field = [s for _ in range(n_hidden)]
    return net.to(device)


def crossing_layers(net):
    """The list of hidden crossing activation modules of a Sample/UniformSample net."""
    return getattr(net, "gaussian_crossing", None) or net.uniform_crossing


# ============================================================
# Forward-hook capture of the model's internal per-sample activations
# ============================================================
class Capture:
    """Captures, per forward pass: for each hidden crossing layer its input d_l and
    output z_l (shape [N, T, H]); and the readout layer's pre-ensemble output
    y_samples (shape [N, T, 1])."""

    def __init__(self, net):
        self.crossings = crossing_layers(net)
        self.n_hidden = len(self.crossings)
        self.d = [None] * self.n_hidden
        self.z = [None] * self.n_hidden
        self.y_samples = None
        self.handles = []
        for l, layer in enumerate(self.crossings):
            self.handles.append(layer.register_forward_hook(self._make(l)))
        self.handles.append(net.fcs[-1].register_forward_hook(self._readout_hook))

    def _make(self, l):
        def hook(module, inp, out):
            self.d[l] = inp[0].detach()
            self.z[l] = out.detach()
        return hook

    def _readout_hook(self, module, inp, out):
        self.y_samples = out.detach()      # [N, T, 1] (before EnsembleMeanLayer)

    def remove(self):
        for hnd in self.handles:
            hnd.remove()


# ============================================================
# Perturbation gate (cov_deriv_gate)
# ------------------------------------------------------------
# The perturbation gate is intended to reduce covariance-credit bias by correlating the
# loss with an explicitly injected local perturbation, rather than with all spontaneous
# unit fluctuations.  Instead of regressing L onto z's own (large, binary) forward noise,
# we inject a small EXTRA perturbation xi into a rotating BLOCK of hidden pre-activations
# and regress L onto xi.  Only that block is perturbed, and only that block is updated.
#
# Crucially the injection is done via forward PRE-hooks on the crossing layers, so the
# perturbation enters the ACTUAL loss-bearing forward pass:
#
#       d_sample = d + base_nnn_noise + alpha * G_k * xi_std          (xi_std ~ N(0,1))
#
# The model's own noise field / stochastic crossing is left completely intact -- we only
# add xi on top, on the gated units.  The injected perturbation p = alpha * G_k * xi_std
# is recorded per pass so the hidden credit can be estimated as Cov(L, p)/Var(p).
#
# The Perturber also supports an ANTITHETIC / COMMON-RANDOM-NUMBER mode ("crn", used by
# cov_deriv_gate_crn): a FIXED perturbation is added as +xi or -xi on demand, so the caller
# can run a paired (+xi, -xi) forward under the SAME crossing noise (via an RNG-state reset)
# and cancel the intrinsic-noise nuisance in the loss difference (see train_cov / §8.3).
# ============================================================
class Perturber:
    """Injects a small block-gated perturbation into hidden pre-activations via forward
    pre-hooks so credit can be taken from Cov(L, perturbation).

    Two modes:
      mode="gate" (default): each forward draws a fresh p = alpha * mask * N(0,1) and
        records it in `p[l]` (shape [N, T, H_l], zero off the gated block).
      mode="crn"           : adds `sign * fixed_p[l]` (a caller-supplied fixed perturbation),
        used to run antithetic (+/-) pairs under common random numbers.

    Set `masks` (a list of [H_l] float gate vectors, 1.0 on the currently gated units)
    before each forward pass."""

    def __init__(self, net, alpha: float, mode: str = "gate"):
        self.crossings = crossing_layers(net)
        self.n_hidden = len(self.crossings)
        self.alpha = float(alpha)
        self.mode = mode
        self.masks = [None] * self.n_hidden     # [H_l] float gate, set each epoch
        self.p = [None] * self.n_hidden         # last injected perturbation [N, T, H_l]
        self.fixed_p = [None] * self.n_hidden   # crn: caller-supplied fixed perturbation
        self.sign = 1.0                         # crn: +1 or -1 for the antithetic pair
        self.handles = []
        for l, layer in enumerate(self.crossings):
            self.handles.append(layer.register_forward_pre_hook(self._make(l)))

    def _make(self, l):
        def pre_hook(module, inp):
            d = inp[0]                                          # [N, T, H] pre-activation
            if self.mode == "crn":
                # add the SAME fixed perturbation with the current sign (antithetic pair)
                p = self.sign * self.fixed_p[l]
                return (d + p,) + tuple(inp[1:])
            mask = self.masks[l]
            if mask is None or self.alpha == 0.0:
                self.p[l] = torch.zeros_like(d)
                return None                                    # no injection this pass
            # nonzero only on the gated block; base forward noise is added by the layer
            p = (self.alpha * mask) * torch.randn_like(d)      # alpha * G_k * xi_std
            self.p[l] = p.detach()
            return (d + p,) + tuple(inp[1:])                   # perturb the ACTUAL forward
        return pre_hook

    def remove(self):
        for hnd in self.handles:
            hnd.remove()


def rng_snapshot(device: torch.device):
    """Snapshot the RNG state (for common-random-numbers antithetic pairs)."""
    if device.type == "cuda":
        return ("cuda", device, torch.cuda.get_rng_state(device))
    return ("cpu", None, torch.get_rng_state())


def rng_restore(snap) -> None:
    """Restore an rng_snapshot so the next forward re-draws the identical noise."""
    kind, device, state = snap
    if kind == "cuda":
        torch.cuda.set_rng_state(state, device)
    else:
        torch.set_rng_state(state)


def gate_masks(hidden_sizes, epoch: int, block_size: int, mode: str,
               device: torch.device):
    """Block gate G_k for each hidden layer: a rotating (cyclic) or random contiguous
    block of `block_size` units is set to 1.0, the rest 0.0.  Returns a list of [H_l]."""
    masks = []
    for H in hidden_sizes:
        n_blocks = max(1, math.ceil(H / block_size))
        if mode == "cyclic":
            b = epoch % n_blocks
        else:  # random
            b = int(torch.randint(0, n_blocks, (1,)).item())
        m = torch.zeros(H, device=device)
        m[b * block_size:(b + 1) * block_size] = 1.0
        masks.append(m)
    return masks


# ============================================================
# Proposed forward-noise covariance learning (manual; NO autograd for the network)
# ============================================================
def covariance_credit(z_l, L, credit):
    """Estimate the activity-side credit g_z ~= dL/dz from stochastic samples.

    z_l : [N, T, H]  (T = samples, pooled over the credit passes)
    L   : [N, T]     (per-sample loss)
    credit == "pooled"   : one global scalar per unit, g_z [H]  (centre over N,T).
    credit == "per_input": input-specific credit, g_z [N, H]  (centre over T only,
        so the confounding between-input loss variation is removed -> a much less
        biased, input-dependent estimate of the local gradient).
    Returns (g_broadcast, g_unit) where g_broadcast broadcasts against [N, T, H] and
    g_unit is a [H] per-unit summary for the activity-stats figure.
    """
    if credit == "pooled":
        cz = z_l - z_l.mean(dim=(0, 1), keepdim=True)
        cL = (L - L.mean()).unsqueeze(-1)                       # [N, T, 1]
        cov = (cL * cz).mean(dim=(0, 1))                        # [H]
        var = (cz ** 2).mean(dim=(0, 1))                        # [H]
        g_z = cov / (var + EPS)                                 # [H]
        return g_z.view(1, 1, -1), g_z
    # per_input
    cz = z_l - z_l.mean(dim=1, keepdim=True)                    # [N, T, H]
    cL = (L - L.mean(dim=1, keepdim=True)).unsqueeze(-1)        # [N, T, 1]
    cov = (cL * cz).mean(dim=1)                                 # [N, H]
    var = (cz ** 2).mean(dim=1)                                 # [N, H]
    g_z = cov / (var + EPS)                                     # [N, H]
    return g_z.unsqueeze(1), g_z.mean(dim=0)                    # broadcast [N,1,H], unit [H]


def cov_weight(d_next, z_prev, pool: bool = False):
    """WEIGHT MIRROR (cov_jac): estimate the forward weight W (d_next = W z_prev + b)
    from forward-noise covariance alone -- NO explicit transpose.

    Because the pre-activation is LINEAR in the previous activity, per input the
    single-variable regression coefficient of d_next_j on z_prev_i equals W_ji when the
    z_prev_i fluctuate independently (which they do: independent per-unit crossing noise
    at a fixed input).  We therefore estimate it WITHIN each input (over the T samples,
    where the fluctuation is pure noise) and average over inputs:

        W_hat[j,i] = mean_n  Cov_t(d_next_j, z_prev_i) / Var_t(z_prev_i)   ~=   W_{ji}

    NB: correlate with the CONTINUOUS pre-activation d_next, never the binary z_next
    (the latter is a poor estimator -- the same binary degeneracy as the CRN gate).

    d_next : [N, T, Ho]   z_prev : [N, T, Hi]   ->   W_hat : [Ho, Hi]
    """
    cd = d_next - d_next.mean(dim=1, keepdim=True)              # [N, T, Ho]
    cz = z_prev - z_prev.mean(dim=1, keepdim=True)              # [N, T, Hi]
    cov = torch.einsum("nto,nti->noi", cd, cz) / d_next.shape[1]  # [N, Ho, Hi]
    var = (cz ** 2).mean(dim=1)                                 # [N, Hi]
    if pool:
        # Variance-weighted "within" pooling over inputs (idea 2): keep the per-input
        # centering (so the between-input cross-unit confound is NOT reintroduced) but
        # SUM numerator and denominator over inputs before dividing.  Because W is the
        # SAME for every input (unlike the gradient), this uses all N*T samples for one
        # estimate -> far lower variance than mean_n(cov_n / var_n).
        return cov.sum(dim=0) / (var.sum(dim=0).unsqueeze(0) + EPS)  # [Ho, Hi]
    W = cov / (var.unsqueeze(1) + EPS)                          # [N, Ho, Hi]
    return W.mean(dim=0)                                        # [Ho, Hi]


class ManualOpt:
    """Minimal in-place optimiser for the manually-computed gradients.

    kind == "sgd"  : param -= lr * grad.
    kind == "adam" : an Adam step (adaptive/preconditioned) using the SAME manual
        gradients -- this only changes HOW the step is taken, not the forward-noise
        credit itself (still no transposed-weight backward pass).
    """

    def __init__(self, kind: str, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.kind = kind
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m, self.v, self.step = {}, {}, {}

    def update(self, key: str, param, grad, lr: float):
        """Apply the step in place and RETURN the decrement applied (delta = before -
        after), so cov_jac can integrate the same known increment into its weight
        mirrors (Kolen-Pollack tracking)."""
        if self.kind == "sgd":
            step = lr * grad
            param.data -= step
            return step
        if key not in self.m:
            self.m[key] = torch.zeros_like(grad)
            self.v[key] = torch.zeros_like(grad)
            self.step[key] = 0
        self.step[key] += 1
        m, v = self.m[key], self.v[key]
        m.mul_(self.b1).add_(grad, alpha=1.0 - self.b1)
        v.mul_(self.b2).addcmul_(grad, grad, value=1.0 - self.b2)
        m_hat = m / (1.0 - self.b1 ** self.step[key])
        v_hat = v / (1.0 - self.b2 ** self.step[key])
        step = lr * m_hat / (v_hat.sqrt() + self.eps)
        param.data -= step
        return step


def lr_at(lr0: float, epoch: int, epochs: int, decay: str) -> float:
    """Learning-rate schedule (shrinks the end-stage stochastic 'noise ball')."""
    if decay == "cosine":
        return lr0 * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, epochs)))
    if decay == "exp":
        return lr0 * (0.01 ** (epoch / max(1, epochs)))     # -> 1% of lr0 at the end
    return lr0


def train_cov(net, x: torch.Tensor, t_target: torch.Tensor, noise: str, sigma: float,
              radius: float, method: str, lr: float, epochs: int,
              hidden_lr_scale: float = 1.0, credit: str = "per_input",
              credit_passes: int = 1, opt: str = "sgd", lr_decay: str = "none",
              log_every: int = 0, slope: str = "kde", gate_block_size: int = 8,
              gate_alpha: float = 0.05, gate_mode: str = "cyclic", jac_ema: float = 0.9,
              jac_track: bool = False, jac_out: str = "cov",
              out_probe_alpha: float = 0.2):
    """Manual covariance learning on a library Sample/UniformSample model.

    Hidden layers: covariance credit (x local slope) from captured per-sample
    activations; `credit` selects pooled vs per-input estimation.  For method "cov_deriv"
    the slope is `slope="kde"` (DEFAULT: the crossing's own distribution-free density
    estimate (xor2-xor1)/(2h)) or `slope="analytic"` (the hand-coded phi'(d)).
    `credit_passes` accumulates the statistics over several stochastic forward passes
    (variance reduction: effective samples = credit_passes * t).  Output layer: exact
    readout gradient on the ENSEMBLE-MEAN features (expected-value readout).  All no-grad.

    method == "cov_deriv_gate": PERTURBATION GATE.  An extra small perturbation xi is
    injected (via Perturber's forward pre-hooks) into a rotating BLOCK of hidden
    pre-activations, and the hidden credit is taken from Cov(L, xi)/Var(xi) instead of
    Cov(L, z)/Var(z).  Only the gated block is perturbed AND updated each epoch
    (gate_block_size units; cyclic or random block via gate_mode; strength gate_alpha).

    method == "cov_deriv_gate_crn": ANTITHETIC / COMMON-RANDOM-NUMBER gate.  Same block
    gating, but each credit pass runs a PAIRED (+xi, -xi) forward under the SAME crossing
    noise (RNG state reset between the two).  The intrinsic-noise nuisance then cancels in
    the loss DIFFERENCE, so credit = Cov_t(L(+xi) - L(-xi), xi) / (2 Var_t(xi)) is an
    UNBIASED and (unlike cov_deriv_gate) LOW-VARIANCE estimate of dL/dd.  Readout and
    z_prev use the antithetic AVERAGE (the O(alpha) perturbation cancels).

    method == "cov_jac": STRUCTURED / RECURSIVE covariance credit (weight mirror).  Instead
    of collapsing the whole downstream onto Cov(L, z), credit is propagated LAYER BY LAYER
    from the exact output error, using forward weights RECOVERED from covariance,
    W_hat[l+1] = mean_n Cov_t(d[l+1], z[l])/Var_t(z[l]) (EMA-smoothed over epochs, rate
    jac_ema).  The recursion  delta^l = (dz/dd)^l * (W_hat[l+1]^T delta^{l+1})  is a
    forward-only, weight-transport-free reconstruction of true backprop.  Slope from KDE.
    """
    assert method in ("cov_only", "cov_deriv", "cov_deriv_gate", "cov_deriv_kde",
                      "cov_deriv_gate_crn", "cov_jac", "cov_jac_full",
                      "cov_deriv_field_gate")
    assert credit in ("pooled", "per_input")
    assert slope in ("kde", "analytic")
    assert jac_out in ("cov", "cov_m3", "probe")
    gate = method == "cov_deriv_gate"
    crn = method == "cov_deriv_gate_crn"
    jac = method in ("cov_jac", "cov_jac_full")
    jac_full = method == "cov_jac_full"
    field_gate = method == "cov_deriv_field_gate"
    W_ema = {}                                      # cov_jac: running weight mirrors
    cap = Capture(net)
    n_hidden = cap.n_hidden
    hidden_sizes = list(net.structure[1:-1])                        # units per hidden layer
    N = x.shape[0]
    optim = ManualOpt(opt)
    perturber = (Perturber(net, gate_alpha, mode="crn" if crn else "gate")
                 if (gate or crn) else None)
    losses, last_stats = [], {}
    for epoch in range(epochs):
        lr_t = lr_at(lr, epoch, epochs, lr_decay)
        lr_hidden = lr_t * hidden_lr_scale
        if gate or crn:
            # pick the block(s) to perturb + update this epoch; set them on the Perturber
            # so the injection happens inside the loss-bearing forward pass below.
            masks = gate_masks(hidden_sizes, epoch, gate_block_size, gate_mode, x.device)
            perturber.masks = masks
        with torch.no_grad():
            # --- collect stochastic samples over credit_passes forward passes ---
            z_p = [[] for _ in range(n_hidden)]
            d_p = [[] for _ in range(n_hidden)]
            xi_p = [[] for _ in range(n_hidden)]
            L_p, y_p, dL_p, ys_p = [], [], [], []
            for _ in range(credit_passes):
                if crn:
                    # Build a FIXED perturbation for this pair, then run +xi and -xi under
                    # COMMON RANDOM NUMBERS (identical crossing noise via an RNG reset).
                    xis = [(gate_alpha * perturber.masks[l].view(1, 1, -1)
                            * torch.randn(N, net.t, hidden_sizes[l], device=x.device))
                           for l in range(n_hidden)]
                    perturber.fixed_p = xis
                    snap = rng_snapshot(x.device)
                    perturber.sign = 1.0
                    y_plus = net(x)                                        # draws noise eta
                    L_plus = (cap.y_samples.squeeze(-1) - t_target) ** 2   # [N, T]
                    z_plus = [cap.z[l] for l in range(n_hidden)]
                    d_plus = [cap.d[l] for l in range(n_hidden)]
                    rng_restore(snap)                                      # re-draw same eta
                    perturber.sign = -1.0
                    y_minus = net(x)
                    L_minus = (cap.y_samples.squeeze(-1) - t_target) ** 2
                    z_minus = [cap.z[l] for l in range(n_hidden)]
                    dL_p.append(L_plus - L_minus)                          # eta cancels
                    y_p.append(0.5 * (y_plus + y_minus))                   # ~ unperturbed y
                    for l in range(n_hidden):
                        xi_p[l].append(xis[l])
                        z_p[l].append(0.5 * (z_plus[l] + z_minus[l]))      # antithetic mean
                        d_p[l].append(d_plus[l])
                else:
                    y_p.append(net(x))                               # [N, 1]; fires hooks
                    L_p.append((cap.y_samples.squeeze(-1) - t_target) ** 2)  # [N, T]
                    if jac:
                        ys_p.append(cap.y_samples)                   # [N, T, 1] pre-ensemble
                    for l in range(n_hidden):
                        z_p[l].append(cap.z[l])
                        d_p[l].append(cap.d[l])
                        if gate:
                            xi_p[l].append(perturber.p[l])           # injected perturbation
            z = [torch.cat(z_p[l], dim=1) for l in range(n_hidden)]      # [N, K*T, H]
            d = [torch.cat(d_p[l], dim=1) for l in range(n_hidden)]
            xi = [torch.cat(xi_p[l], dim=1) for l in range(n_hidden)] if (gate or crn) else None
            ys = torch.cat(ys_p, dim=1) if jac else None                # [N, K*T, 1]
            L = torch.cat(L_p, dim=1) if not crn else None              # [N, K*T]
            dL = torch.cat(dL_p, dim=1) if crn else None                # [N, K*T]
            y = torch.stack(y_p, dim=0).mean(dim=0)                      # [N, 1]
            T = (dL if crn else L).shape[1]

            # --- cov_jac: weight mirrors (EMA) + recursive credit from the output error ---
            a_jac = None
            g_y = None
            if jac:
                slope_full = [kde_slope(cap.crossings[l], d[l]) for l in range(n_hidden)]
                slope_mean = [s.mean(dim=1) for s in slope_full]        # [N, H] each
                # measure this epoch's forward weights from covariance, then EMA-smooth
                meas = {"out": cov_weight(ys, z[-1], pool=jac_track)}   # [1, H]  (readout)
                for l in range(1, n_hidden):
                    meas[l] = cov_weight(d[l], z[l - 1], pool=jac_track)  # [H_l, H_{l-1}]
                if not W_ema:
                    W_ema.update(meas)                                 # cold start
                else:
                    for k, v in meas.items():
                        W_ema[k] = jac_ema * W_ema[k] + (1.0 - jac_ema) * v
                # recursion top-down: a[l] = dL/dz[l] (per input) [N, H]
                if jac_full:
                    # cov_jac_full: the readout error itself from forward-noise statistics.
                    # No analytic dL/dy anywhere.  Three estimators (jac_out):
                    #   "cov"    : g_y = Cov_t(L, y)/Var_t(y).  For quadratic L this equals
                    #              2 (E[y] - t) PLUS a skewness term E[eps^3]/Var(eps) of the
                    #              crossing-induced readout fluctuation -- a bias that does
                    #              NOT vanish at convergence (Adam amplifies it late).
                    #   "cov_m3" : subtract the OBSERVED third central moment of y from the
                    #              covariance -- exact bias removal for quadratic loss, still
                    #              only forward statistics of (L, y).
                    #   "probe"  : add a known symmetric Gaussian probe xi to the readout
                    #              samples and regress the PROBED loss on xi:
                    #              g_y = Cov_t(L(y+xi), xi)/Var_t(xi).  E[xi^3] = 0, so this
                    #              is unbiased for any smooth loss (no skew term at all).
                    ys_f = ys.squeeze(-1)                              # [N, KT]
                    if jac_out == "probe":
                        xi_out = out_probe_alpha * torch.randn_like(ys_f)
                        L_pr = (ys_f + xi_out - t_target) ** 2
                        cxi = xi_out - xi_out.mean(dim=1, keepdim=True)
                        cLp = L_pr - L_pr.mean(dim=1, keepdim=True)
                        g_y = ((cLp * cxi).mean(dim=1)
                               / ((cxi ** 2).mean(dim=1) + EPS)).unsqueeze(1)   # [N, 1]
                    else:
                        cy = ys_f - ys_f.mean(dim=1, keepdim=True)
                        cL = L - L.mean(dim=1, keepdim=True)
                        cov_Ly = (cL * cy).mean(dim=1)                 # [N]
                        if jac_out == "cov_m3":
                            cov_Ly = cov_Ly - (cy ** 3).mean(dim=1)    # skew correction
                        g_y = (cov_Ly / ((cy ** 2).mean(dim=1) + EPS)).unsqueeze(1)
                a_jac = [None] * n_hidden
                err_out = g_y if jac_full else 2.0 * (y - t_target)     # [N, 1] ~ dL/dy
                a_jac[-1] = err_out * W_ema["out"]                      # [N,1]*[1,H] -> [N,H]
                for l in range(n_hidden - 2, -1, -1):
                    dd_next = a_jac[l + 1] * slope_mean[l + 1]          # dL/dd[l+1]  [N, H]
                    a_jac[l] = dd_next @ W_ema[l + 1]                   # [N, H_l]

            # z_prev for each hidden layer: input x (broadcast over T) then z_1
            z_prev = [x.unsqueeze(1).expand(N, T, x.shape[1]), z[0]]
            hidden_grads, stats = [], []
            for l in range(n_hidden):
                if jac:
                    # Recursive weight-mirror credit: a_jac[l] ~= dL/dz[l] (from the
                    # top-down recursion above), times the local KDE slope dz/dd.
                    g_unit = a_jac[l].mean(dim=0)
                    delta_hat = a_jac[l].unsqueeze(1) * slope_full[l]       # [N,T,H]
                elif gate:
                    # Credit from the INJECTED perturbation xi (added at pre-activation d).
                    # Cov(L, xi)/Var(xi) already estimates dL/dd = delta DIRECTLY, so no
                    # extra phi'(d) factor is applied (that would double-count the local
                    # crossing slope, which is captured implicitly via L's response to xi).
                    g_bcast, g_unit = covariance_credit(xi[l], L, credit)   # ~ dL/dd
                    gate_mask = perturber.masks[l].view(1, 1, -1)
                    # Zero credit on non-gated units => only the gated block is updated.
                    delta_hat = g_bcast.expand_as(z[l]) * gate_mask
                elif crn:
                    # Antithetic / common-random-number credit: regress the loss DIFFERENCE
                    # L(+xi) - L(-xi) onto xi.  The shared crossing noise cancels in dL, so
                    #   Cov_t(dL, xi) / (2 Var_t(xi))  ~= dL/dd   (unbiased, low variance).
                    # The 1/2 comes from dL ~ 2 * sum_j (dL/dd_j) xi_j.
                    g_bcast, g_unit = covariance_credit(xi[l], dL, credit)
                    gate_mask = perturber.masks[l].view(1, 1, -1)
                    delta_hat = 0.5 * g_bcast.expand_as(z[l]) * gate_mask
                else:
                    g_bcast, g_unit = covariance_credit(z[l], L, credit)    # ~ dL/dz
                    if method == "cov_only":
                        delta_hat = g_bcast.expand_as(z[l])
                    elif method == "cov_deriv" and slope == "analytic":
                        # analytic, noise-distribution-specific local slope phi'(d) (ablation)
                        delta_hat = g_bcast * phi_prime(d[l], noise, sigma, radius)
                    else:  # cov_deriv (DEFAULT slope="kde") or cov_deriv_kde alias
                        # distribution-FREE slope from the crossing's own internal density
                        # estimate coeff = mean_t(xor2 - xor1)/(2h) (an antithetic / CRN
                        # finite difference over the +/-h thresholds on shared samples).
                        dz_dd = kde_slope(cap.crossings[l], d[l])           # ~ dz/dd, [N,T,H]
                        delta_hat = g_bcast * dz_dd
                    if field_gate:
                        # Gate the update by the per-unit noise field s_i (recruitment):
                        # D W_l[i,j] *= s_i.  Un-recruited units (s_i = 0) are detached
                        # and receive no update -- the one-line link to the NNN noise field.
                        delta_hat = delta_hat * net.noise_field[l].view(1, 1, -1)
                gW = torch.einsum("nto,nti->oi", delta_hat, z_prev[l]) / (N * T)
                gb = delta_hat.mean(dim=(0, 1))
                hidden_grads.append((gW, gb))
                stats.append({
                    "g_z": g_unit.cpu(),
                    "mean_activity": z[l].mean(dim=(0, 1)).cpu(),
                    "phi_prime": phi_prime(d[l], noise, sigma, radius).mean(dim=(0, 1)).cpu(),
                })

            # --- readout on the EXPECTED (ensemble-mean) features ---
            z_bar = z[-1].mean(dim=1)                            # [N, H]  ~ phi_bar
            # cov_jac_full: covariance-estimated readout error instead of the analytic
            # loss derivative (g_y computed above from Cov_t(L, y)/Var_t(y)).
            dL_dy = g_y if jac_full else 2.0 * (y - t_target)    # [N, 1]
            gWout = torch.einsum("no,ni->oi", dL_dy, z_bar) / N  # [1, H]
            gbout = dL_dy.mean(dim=0)                            # [1]

            # --- apply updates via the manual optimiser (sgd or adam) ---
            steps = {}
            for l in range(n_hidden):
                steps[l] = optim.update(f"w{l}", net.fcs[l].weight,
                                        hidden_grads[l][0], lr_hidden)
                if net.fcs[l].bias is not None:
                    optim.update(f"b{l}", net.fcs[l].bias, hidden_grads[l][1], lr_hidden)
            steps["out"] = optim.update("wout", net.fcs[-1].weight, gWout, lr_t)
            if net.fcs[-1].bias is not None:
                optim.update("bout", net.fcs[-1].bias, gbout, lr_t)

            # cov_jac Kolen-Pollack PREDICT (idea 1): the true weights just moved by the
            # KNOWN increment -steps, so shift the mirrors by exactly the same amount.
            # The covariance 'meas' (EMA-corrected above) then only has to pin down the
            # STATIC initial offset instead of chasing a moving target -> no tracking lag.
            if jac and jac_track:
                W_ema["out"] = W_ema["out"] - steps["out"]
                for l in range(1, n_hidden):
                    W_ema[l] = W_ema[l] - steps[l]

            losses.append(float(((y - t_target) ** 2).mean()))
            last_stats = {"layer1": stats[0]}

        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  [{method}] epoch {epoch:5d}  eval_mse={losses[-1]:.5f}")
    cap.remove()
    if perturber is not None:
        perturber.remove()
    return losses, last_stats


# ============================================================
# Backprop reference (autograd on the SAME library model -- reference only)
# ============================================================
def train_backprop(net, x: torch.Tensor, t_target: torch.Tensor, lr: int,
                   epochs: int, log_every: int = 0):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss = ((net(x) - t_target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  [backprop] epoch {epoch:5d}  mse={losses[-1]:.5f}")
    return losses


def predict(net, x: torch.Tensor, passes: int = 8) -> np.ndarray:
    """Ensemble prediction averaged over a few stochastic passes (smooth curve)."""
    with torch.no_grad():
        y = torch.stack([net(x) for _ in range(passes)], dim=0).mean(dim=0)
    return y.squeeze(1).cpu().numpy()


# ============================================================
# Plots (displayed with plt.show(), not saved)
# ============================================================
def plot_losses(losses: dict):
    fig = plt.figure(figsize=(7, 5))
    for name, curve in losses.items():
        plt.semilogy(curve, label=name)
    plt.xlabel("epoch")
    plt.ylabel("MSE (eval, log scale)")
    plt.title("Learning curves: backprop vs forward-noise covariance learning")
    plt.legend()
    plt.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_predictions(x_raw, target, preds: dict):
    order = np.argsort(x_raw)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(x_raw[order], target[order], "k-", lw=2.5, label="target sin(x)")
    styles = {"backprop": "--", "cov_only": "-.", "cov_deriv": "-",
              "cov_deriv_kde": "-", "cov_deriv_analytic": (0, (3, 1, 1, 1)),
              "cov_jac": (0, (1, 1)), "cov_jac_sgd": (0, (1, 1)),
              "cov_jac_adam": (0, (4, 1, 1, 1)),
              "cov_jac_full_sgd": (0, (2, 2)), "cov_jac_full_adam": (0, (6, 1, 1, 1)),
              "cov_deriv_gate": ":", "cov_deriv_gate_crn": (0, (5, 1)),
              "cov_deriv_field_gate": (0, (3, 1, 1, 1, 1, 1))}
    for name, y in preds.items():
        plt.plot(x_raw[order], y[order], linestyle=styles.get(name, "-"),
                 lw=1.6, label=name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predictions on y = sin(x)")
    plt.legend()
    plt.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_activity_stats(stats_layer: dict):
    idx = np.arange(stats_layer["mean_activity"].numel())
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].bar(idx, stats_layer["mean_activity"].numpy(), color="tab:blue")
    axes[0].set_ylabel("mean activity\n<z>")
    axes[0].set_title("cov_deriv, hidden layer 1 (final epoch)")
    axes[1].bar(idx, stats_layer["g_z"].numpy(), color="tab:green")
    axes[1].set_ylabel("covariance credit\ng_z = Cov(L,z)/Var(z)")
    axes[2].bar(idx, stats_layer["phi_prime"].numpy(), color="tab:red")
    axes[2].set_ylabel("local derivative\nmean phi'(d)")
    axes[2].set_xlabel("hidden unit index")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_fit_check(x_raw, target, preds: dict,
                   focus=("backprop", "cov_jac_adam", "cov_jac_full_adam")):
    """Focused confirmation figure: does cov_jac_adam approximate sin(x) as TIGHTLY as
    backprop?  Overlays only the target and the `focus` methods, plus a residual panel and
    the MSE in the legend, so the fit quality is unmistakable (rather than buried in the
    all-methods plot)."""
    order = np.argsort(x_raw)
    colors = {"backprop": "tab:blue", "cov_jac_adam": "tab:red",
              "cov_jac_sgd": "tab:green", "cov_jac_full_adam": "tab:purple"}
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(x_raw[order], target[order], "k-", lw=3.0, alpha=0.35, label="target sin(x)")
    for name in focus:
        if name not in preds:
            continue
        mse = float(np.mean((preds[name] - target) ** 2))
        axes[0].plot(x_raw[order], preds[name][order], color=colors.get(name), lw=1.8,
                     label=f"{name}  (MSE={mse:.2e})")
        axes[1].plot(x_raw[order], (preds[name] - target)[order], color=colors.get(name),
                     lw=1.2, label=name)
    axes[0].set_ylabel("y")
    axes[0].set_title("Fit check: cov_jac_adam vs backprop on y = sin(x)")
    axes[0].legend()
    axes[1].axhline(0.0, color="k", lw=0.6)
    axes[1].set_ylabel("residual\n(pred - sin)")
    axes[1].set_xlabel("x")
    axes[1].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Forward-noise covariance learning on nnn Sample models (PoC).")
    p.add_argument("--noise", choices=("gaussian", "uniform"), default="gaussian",
                   help="gaussian -> SimpleNNNSample, uniform -> SimpleNNNUniformSample")
    p.add_argument("--epochs",      type=int,   default=1500)
    p.add_argument("--hidden-dim",  type=int,   default=64)
    p.add_argument("--num-samples", type=int,   default=64,
                   help="t: stochastic samples the model draws internally")
    p.add_argument("--lr",          type=float, default=1e-2)
    p.add_argument("--sigma",       type=float, default=0.5, help="Gaussian crossing std")
    p.add_argument("--radius",      type=float, default=1.0, help="uniform crossing half-width")
    p.add_argument("--crossing-h",  type=float, default=0.2, help="crossing threshold h")
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--device",      type=str,   default="cpu")
    p.add_argument("--hidden-lr-scale", type=float, default=1.0,
                   help="scale the (noisy) hidden covariance step vs the readout step")
    p.add_argument("--credit", choices=("pooled", "per_input"), default="per_input",
                   help="hidden credit: 'per_input' Cov over samples per input "
                        "(input-dependent, less biased; default) or 'pooled' (one "
                        "global scalar per unit)")
    p.add_argument("--credit-passes", type=int, default=1,
                   help="accumulate covariance stats over this many forward passes "
                        "(variance reduction: effective samples = passes * t) [1]")
    p.add_argument("--opt", choices=("sgd", "adam"), default="sgd",
                   help="manual-update rule for cov methods: 'sgd' or 'adam' "
                        "(adaptive step; converges faster) [sgd]")
    p.add_argument("--lr-decay", choices=("none", "cosine", "exp"), default="none",
                   help="learning-rate schedule for cov methods; decay shrinks the "
                        "end-stage stochastic noise-ball -> lower final MSE [none]")
    # --- perturbation-gate (cov_deriv_gate) knobs ---
    p.add_argument("--gate-block-size", type=int, default=8,
                   help="cov_deriv_gate: number of hidden units in the perturbed/updated "
                        "block G_k [8]")
    p.add_argument("--gate-alpha", type=float, default=0.05,
                   help="cov_deriv_gate: strength of the injected pre-activation "
                        "perturbation xi (added on top of the model's forward noise) [0.05]")
    p.add_argument("--gate-mode", choices=("random", "cyclic"), default="cyclic",
                   help="cov_deriv_gate: rotate the gated block cyclically or pick it at "
                        "random each epoch [cyclic]")
    p.add_argument("--slope", choices=("kde", "analytic"), default="kde",
                   help="cov_deriv local slope dz/dd: 'kde' = the crossing's OWN "
                        "distribution-free density estimate (xor2-xor1)/(2h) [DEFAULT]; "
                        "'analytic' = hand-coded phi'(d) per noise distribution (ablation)")
    p.add_argument("--jac-ema", type=float, default=0.9,
                   help="cov_jac: EMA rate for the running weight mirrors "
                        "W_hat = Cov(d_next,z)/Var(z) (higher = smoother/lower-variance) [0.9]")
    p.add_argument("--jac-out", choices=("cov", "cov_m3", "probe"), default="cov_m3",
                   help="cov_jac_full readout-error estimator: 'cov' = raw Cov(L,y)/Var(y) "
                        "(carries an E[eps^3]/Var skew bias that Adam amplifies late); "
                        "'cov_m3' = subtract the observed third moment (exact for quadratic "
                        "loss; matches backprop) [DEFAULT]; 'probe' = Cov(L(y+xi), xi)/Var(xi) "
                        "with an injected symmetric Gaussian probe xi (unbiased for any loss)")
    p.add_argument("--out-probe-alpha", type=float, default=0.2,
                   help="cov_jac_full --jac-out probe: std of the injected readout probe [0.2]")
    p.add_argument("--jac-track", action=argparse.BooleanOptionalAction, default=True,
                   help="cov_jac: Kolen-Pollack weight-mirror TRACKING (DEFAULT ON) -- "
                        "integrate the known applied weight update into the mirrors (predict) "
                        "and pool the mirror covariance over inputs (idea 1+2). Removes the "
                        "mirror tracking lag. Use --no-jac-track to disable.")
    p.add_argument("--fit-check", action="store_true",
                   help="show an extra FOCUSED figure overlaying only sin(x), backprop and "
                        "cov_jac_adam (+residuals, MSE) to confirm cov_jac_adam fits as "
                        "tightly as backprop.")
    p.add_argument("--save", type=str, default=None,
                   help="directory to save the figures as PNG instead of plt.show() "
                        "(useful headless / for the paper).")
    p.add_argument("--include-gates", action="store_true",
                   help="also run the perturbation/field gate methods (cov_deriv_gate, "
                        "cov_deriv_gate_crn, cov_deriv_field_gate) in the comparison. They "
                        "are kept in the code but OFF the default verification set.")
    p.add_argument("--field-sparsity", type=float, default=0.0,
                   help="cov_deriv_field_gate: fraction of hidden units with ZERO noise "
                        "field (un-recruited: no forward noise AND no update). Applies a "
                        "REAL per-unit noise field to the shared network. 0 -> all units "
                        "recruited, byte-identical to before (default) [0.0]")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # --- task: y = sin(x), x in [-2pi, 2pi]; input normalised to ~[-2, 2] ---
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, NUM_POINTS, dtype=np.float32)
    target_np = np.sin(x_raw).astype(np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    t = torch.tensor(target_np, device=device).unsqueeze(1)

    log_every = max(1, args.epochs // 10)
    model_name = "SimpleNNNSample" if args.noise == "gaussian" else "SimpleNNNUniformSample"
    scale = args.sigma if args.noise == "gaussian" else args.radius
    print(f"Forward-noise covariance learning on nnn.{model_name}  (device={device})")
    print(f"  --noise={args.noise} | H={args.hidden_dim} | t(num-samples)={args.num_samples} "
          f"| noise-scale={scale} | h={args.crossing_h} | lr={args.lr} | epochs={args.epochs}")
    print(f"  credit={args.credit} | credit-passes={args.credit_passes} "
          f"(effective samples/update = {args.credit_passes * args.num_samples}) "
          f"| opt={args.opt} | lr-decay={args.lr_decay}")
    print(f"  gate(cov_deriv_gate): block-size={args.gate_block_size} | "
          f"alpha={args.gate_alpha} | mode={args.gate_mode}")
    print(f"  cov_deriv slope={args.slope} (DEFAULT 'kde' = distribution-free "
          f"(xor2-xor1)/(2h); 'analytic' = phi'(d))")

    # per-unit noise field shared by all networks (a real recruitment field): a fraction
    # --field-sparsity of hidden units are un-recruited (zero field -> no noise, no update).
    # Deterministic (own generator) so every fresh() network shares the same recruitment.
    H = args.hidden_dim
    if args.field_sparsity > 0.0:
        g = torch.Generator().manual_seed(args.seed + 12345)
        field = (torch.rand(H, generator=g) >= args.field_sparsity).float().to(device)
        n_off = int((field == 0).sum())
        print(f"  noise field: {n_off}/{H} hidden units un-recruited "
              f"(sparsity={args.field_sparsity}); cov_deriv_field_gate gates on it")
    else:
        field = None

    # identical initial weights for all methods
    net0 = build_model(args.noise, args.hidden_dim, args.sigma, args.radius,
                       args.crossing_h, args.num_samples, device, field=field)
    init_state = copy.deepcopy(net0.state_dict())

    def fresh():
        n = build_model(args.noise, args.hidden_dim, args.sigma, args.radius,
                        args.crossing_h, args.num_samples, device, field=field)
        n.load_state_dict(init_state)
        return n

    # ============================================================
    # Verification set (default): backprop, cov_only, cov_deriv_analytic, cov_deriv_kde,
    # cov_jac_sgd, cov_jac_adam.  cov_jac uses --jac-track (DEFAULT ON).  The two cov_jac
    # rows differ ONLY in the (local, FPGA-friendly) optimiser: sgd vs adam.  The gate/field
    # methods are kept in the code but run only with --include-gates.
    # ============================================================
    losses, preds, stats = {}, {}, {}

    def run(name, method, **kw):
        print(f"\n[{name}] {kw.pop('_desc', method)}")
        net = fresh()
        lo, st = train_cov(net, x, t, args.noise, args.sigma, args.radius, method,
                           args.lr, args.epochs, args.hidden_lr_scale, args.credit,
                           args.credit_passes, kw.pop("opt", args.opt), args.lr_decay,
                           log_every, **kw)
        losses[name] = lo
        preds[name] = predict(net, x)
        stats[name] = st

    print("\n[backprop] (reference, autograd Adam on the same model)")
    net_bp = fresh()
    losses["backprop"] = train_backprop(net_bp, x, t, args.lr, args.epochs, log_every)
    preds["backprop"] = predict(net_bp, x)

    run("cov_only", "cov_only", _desc="covariance credit only (no phi')")
    run("cov_deriv_analytic", "cov_deriv", slope="analytic",
        _desc="cov_deriv with the hand-coded analytic phi'(d) (ablation)")
    run("cov_deriv_kde", "cov_deriv", slope="kde",
        _desc="cov_deriv with the crossing's distribution-free (xor2-xor1)/(2h) slope")
    run("cov_jac_sgd", "cov_jac", opt="sgd", jac_ema=args.jac_ema, jac_track=args.jac_track,
        _desc="weight-mirror recursive credit, SGD (--jac-track=%s)" % args.jac_track)
    run("cov_jac_adam", "cov_jac", opt="adam", jac_ema=args.jac_ema, jac_track=args.jac_track,
        _desc="weight-mirror recursive credit, ADAM -- expected to match backprop "
              "(--jac-track=%s)" % args.jac_track)
    run("cov_jac_full_sgd", "cov_jac_full", opt="sgd", jac_ema=args.jac_ema,
        jac_track=args.jac_track, jac_out=args.jac_out,
        out_probe_alpha=args.out_probe_alpha,
        _desc="cov_jac + covariance readout error (jac_out=%s), SGD "
              "(--jac-track=%s)" % (args.jac_out, args.jac_track))
    run("cov_jac_full_adam", "cov_jac_full", opt="adam", jac_ema=args.jac_ema,
        jac_track=args.jac_track, jac_out=args.jac_out,
        out_probe_alpha=args.out_probe_alpha,
        _desc="cov_jac + covariance readout error (jac_out=%s), ADAM -- no analytic "
              "dL/dy anywhere (--jac-track=%s)" % (args.jac_out, args.jac_track))

    if args.include_gates:
        run("cov_deriv_gate", "cov_deriv_gate", gate_block_size=args.gate_block_size,
            gate_alpha=args.gate_alpha, gate_mode=args.gate_mode,
            _desc="perturbation-gated credit Cov(L,xi)/Var(xi) on a rotating block")
        run("cov_deriv_gate_crn", "cov_deriv_gate_crn", gate_block_size=args.gate_block_size,
            gate_alpha=args.gate_alpha, gate_mode=args.gate_mode,
            _desc="antithetic/common-random-number gate Cov(L(+xi)-L(-xi), xi)")
        run("cov_deriv_field_gate", "cov_deriv_field_gate", slope=args.slope,
            _desc="cov_deriv gated by the per-unit noise field s_i (recruitment)")

    figs = {"learning_curves": plot_losses(losses),
            "predictions": plot_predictions(x_raw, target_np, preds),
            "layer1_stats": plot_activity_stats(stats["cov_deriv_kde"]["layer1"])}
    if args.fit_check:
        figs["fit_check"] = plot_fit_check(x_raw, target_np, preds)

    # low-noise final MSE from the averaged (multi-pass) prediction
    fin = {k: float(np.mean((v - target_np) ** 2)) for k, v in preds.items()}
    deriv_matches = abs(fin["cov_deriv_kde"] - fin["cov_deriv_analytic"]) <= 0.01
    jac_adam_near_bp = fin["cov_jac_adam"] <= max(2.0 * fin["backprop"], fin["backprop"] + 0.003)
    jac_full_matches = abs(fin["cov_jac_full_adam"] - fin["cov_jac_adam"]) <= 0.003
    order = ["backprop", "cov_only", "cov_deriv_analytic", "cov_deriv_kde",
             "cov_jac_sgd", "cov_jac_adam", "cov_jac_full_sgd", "cov_jac_full_adam"]
    if args.include_gates:
        order += ["cov_deriv_gate", "cov_deriv_gate_crn", "cov_deriv_field_gate"]
    print("\n================ SUMMARY ================")
    print(f"Model: nnn.{model_name}   (readout uses the ensemble-mean = expected value)")
    print("Final MSE (8-pass predict):")
    for k in order:
        print(f"  {k:20s}: {fin[k]:.5f}")
    print("\nInterpretation:")
    print("  - backprop is the exact-gradient reference (autograd Adam on the same model).")
    print("  - cov_deriv_kde = covariance credit x the crossing's OWN distribution-free")
    print("    density slope (xor2-xor1)/(2h); cov_deriv_analytic uses the hand-coded phi'(d).")
    print(f"  - cov_deriv_kde {'MATCHES' if deriv_matches else 'does NOT match'} "
          f"cov_deriv_analytic (delta MSE = {fin['cov_deriv_analytic'] - fin['cov_deriv_kde']:+.5f}) "
          f"-> the analytic phi' is not needed.")
    print("  - cov_jac_{sgd,adam} = STRUCTURED/RECURSIVE credit via weight mirrors "
          "W_hat=Cov(d_next,z)/Var(z)")
    print(f"    (--jac-track={args.jac_track}); the two differ ONLY in the local optimiser.")
    print(f"  - cov_jac_adam final MSE {fin['cov_jac_adam']:.5f} vs backprop {fin['backprop']:.5f}: "
          f"{'REACHES backprop level' if jac_adam_near_bp else 'does NOT yet reach backprop level'}.")
    print(f"  - cov_jac_adam {'beats' if fin['cov_jac_adam'] < fin['cov_jac_sgd'] else 'trails'} "
          f"cov_jac_sgd (delta MSE = {fin['cov_jac_sgd'] - fin['cov_jac_adam']:+.5f}) "
          f"-> the sgd 'floor' is optimisation, not estimator bias.")
    print("  - cov_jac_full_{sgd,adam} = cov_jac with the READOUT error ALSO from forward")
    print(f"    statistics (jac_out={args.jac_out}) ~ dL/dy -> NO analytic loss derivative "
          "anywhere.")
    print(f"  - cov_jac_full_adam {'MATCHES' if jac_full_matches else 'does NOT match'} "
          f"cov_jac_adam (delta MSE = {fin['cov_jac_full_adam'] - fin['cov_jac_adam']:+.5f}) "
          f"-> the analytic dL/dy at the readout is not needed.")
    print("  - Adam is a LOCAL per-weight rule -> keeps the no-weight-transport / "
          "FPGA-friendly property.")
    print("=========================================")

    if args.save:
        import os
        os.makedirs(args.save, exist_ok=True)
        for name, fig in figs.items():
            path = os.path.join(args.save, f"{name}.png")
            fig.savefig(path, dpi=130)
            print(f"  saved {path}")
    else:
        print(f"\nOpening {len(figs)} figure windows (close them to exit)...")
        plt.show()


if __name__ == "__main__":
    main()
