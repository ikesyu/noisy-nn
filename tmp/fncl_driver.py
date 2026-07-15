"""fncl_driver.py — tmp 側実験の実験ドライバ (旧 fncl_lib_tmp.py の後継).

共通・最小のライブラリ部分は nnn パッケージへ移した (nnn.stats: Capture /
kde_slope / phi_prime / crossing_layers、nnn.credit: covariance_credit /
cov_weight / ManualOpt / lr_at)。このモジュールに残るのは tmp 実験に固有の部分:

  - sin(x) 回帰タスクとモデル構築 (1 次元 bump タイリング初期化 + ノイズ場)
  - 外部摂動ゲート (Appendix B 用: Perturber / gate_masks / RNG スナップショット)
  - 学習則ドライバ train_cov (method 文字列で全手法を切り替える) と backprop 参照
  - 確認用プロットと実験ヘルパ (argparse / タスク / 保存)

利便のため nnn 側のライブラリ関数もここから re-export する (旧 fncl_lib_tmp と
同じ `import fncl_driver as fncl` の使い方ができる)。

論文の図表再現には data_nce/ を使うこと。このモジュールは tmp 専用であり、
data_nce/ とは独立している (data_nce/fncl/ は自前の実装を持つ)。

ヘッドレス実行するスクリプトは、このモジュールを import する前に
matplotlib.use("Agg") を呼ぶこと。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Use the local nnn library.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from nnn import model, noise as nnn_noise  # noqa: E402
from nnn.credit import (EPS, ManualOpt, cov_weight, covariance_credit,  # noqa: E402,F401
                        lr_at)
from nnn.stats import (Capture, crossing_layers, kde_slope,  # noqa: E402,F401
                       phi_prime)

NUM_POINTS = 128
UNIFORM_CENTER = 0.0


def noise_funcs(noise: str, sigma: float, radius: float):
    """(pdf, cdf) closures of the injected noise, for nnn.stats.phi_prime."""
    if noise == "gaussian":
        return (nnn_noise.gaussian_pdf_torch(0.0, sigma),
                nnn_noise.gaussian_cdf_torch(0.0, sigma))
    if noise == "uniform":
        return (nnn_noise.uniform_pdf_torch(UNIFORM_CENTER, radius),
                nnn_noise.uniform_cdf_torch(UNIFORM_CENTER, radius))
    raise ValueError(f"--noise must be 'gaussian' or 'uniform', got '{noise}'")


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
# Forward-noise covariance learning driver (manual; NO autograd for the network).
# The estimators themselves live in nnn.credit / nnn.stats; this dispatches the
# method variants and runs the update loop.
# ============================================================
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
    estimate (xor2-xor1)/(2h)) or `slope="analytic"` (the closed-form phi'(d)).
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
    pdf, cdf = noise_funcs(noise, sigma, radius)    # for the analytic phi'(d)
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
                        delta_hat = g_bcast * phi_prime(d[l], pdf, cdf)
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
                    "phi_prime": phi_prime(d[l], pdf, cdf).mean(dim=(0, 1)).cpu(),
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
# 実験ヘルパの最小集合 (data_nce/fncl_common.py と同等; tmp 側で自己完結させる)
# ============================================================
def add_common_args(p: argparse.ArgumentParser, *, epochs: int = 1500,
                    hidden_dim: int = 64, num_samples: int = 64,
                    seeds: str = "0,1,2") -> argparse.ArgumentParser:
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--hidden-dim", type=int, default=hidden_dim)
    p.add_argument("--num-samples", type=int, default=num_samples,
                   help="T: モデルが内部で引く確率サンプル数")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--sigma", type=float, default=0.5, help="ガウスノイズ std")
    p.add_argument("--radius", type=float, default=1.0, help="一様ノイズ半幅")
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--credit", choices=("pooled", "per_input"), default="per_input")
    p.add_argument("--credit-passes", type=int, default=1)
    p.add_argument("--jac-ema", type=float, default=0.9)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seeds", type=str, default=seeds, help="カンマ区切り seed リスト")
    p.add_argument("--out", type=str, default=None, help="出力ディレクトリ")
    p.add_argument("--quick", action="store_true", help="縮小設定での動作確認")
    return p


def finalize_args(args: argparse.Namespace, default_out: str) -> argparse.Namespace:
    if args.quick:
        args.epochs = min(args.epochs, 60)
        args.num_samples = min(args.num_samples, 16)
        args.hidden_dim = min(args.hidden_dim, 16)
        args.seeds = str(args.seeds).split(",")[0]
    args.seed_list = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    args.out_dir = Path(args.out or default_out)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return args


def config_dict(args: argparse.Namespace) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def make_task(device: torch.device):
    """回帰タスク y = sin(x), x in [-2pi, 2pi] (入力は ~[-2, 2] に正規化)."""
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, NUM_POINTS, dtype=np.float32)
    target = np.sin(x_raw).astype(np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    t = torch.tensor(target, device=device).unsqueeze(1)
    return x_raw, target, x, t


def model_factory(noise: str, args: argparse.Namespace, device: torch.device,
                  field: torch.Tensor = None):
    """同一初期重みのネットワークを繰り返し生成する fresh() を返す."""
    net0 = build_model(noise, args.hidden_dim, args.sigma, args.radius,
                       args.crossing_h, args.num_samples, device, field=field)
    init_state = copy.deepcopy(net0.state_dict())

    def fresh():
        n = build_model(noise, args.hidden_dim, args.sigma, args.radius,
                        args.crossing_h, args.num_samples, device, field=field)
        n.load_state_dict(init_state)
        return n

    return fresh


def run_method(spec: dict, fresh, x, t, noise: str, args, log_every: int = 0):
    """spec に従い 1 手法を学習して (losses, pred[np.ndarray], net) を返す.

    spec 例: {"kind": "backprop"} または
             {"method": "cov_jac", "opt": "adam", "jac_track": True, ...}
    method/opt/credit/jac_ema 以外のキー (slope, jac_track, gate_* など) は
    そのまま train_cov に渡される。
    """
    net = fresh()
    spec = dict(spec)
    if spec.pop("kind", None) == "backprop":
        losses = train_backprop(net, x, t, args.lr, args.epochs, log_every)
    else:
        method = spec.pop("method")
        opt = spec.pop("opt", "sgd")
        credit = spec.pop("credit", args.credit)
        jac_ema = spec.pop("jac_ema", args.jac_ema)
        losses, _ = train_cov(
            net, x, t, noise, args.sigma, args.radius, method, args.lr, args.epochs,
            credit=credit, credit_passes=args.credit_passes, opt=opt,
            lr_decay="none", log_every=log_every, jac_ema=jac_ema, **spec)
    pred = predict(net, x)
    return losses, pred, net


def write_text(path: Path, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")
    print(f"  saved {path}")


def save_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    print(f"  saved {path}")


def savefig(fig, path: Path) -> None:
    """PNG (確認用) と PDF (論文用) を常に併せて保存する."""
    path = Path(path)
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {path} (+ .pdf)")
