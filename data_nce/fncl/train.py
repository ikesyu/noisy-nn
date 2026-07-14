"""fncl.train — forward 統計のみによる学習則 (論文 §4) と backprop 参照.

- covariance_credit: スカラー共分散 credit Cov(L, z)/Var(z) (§4.2),
- cov_weight: 共分散 weight mirror W_hat = Cov(d_next, z)/Var(z) (§4.3.1),
- train_cov: cov_only / cov_deriv / cov_jac / cov_jac_full (+ gate 変種) の
  手動更新学習ループ (autograd 不使用; 再帰的誤差伝播は §4.3.2),
- ManualOpt: 局所 SGD/Adam, train_backprop: autograd 参照, predict: 推論.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .constants import EPS
from .network import Capture, kde_slope, phi_prime
from .perturb import Perturber, gate_masks, rng_restore, rng_snapshot


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
