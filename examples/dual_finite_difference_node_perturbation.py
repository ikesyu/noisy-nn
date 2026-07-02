"""
examples/dual_finite_difference_node_perturbation.py

Proof-of-concept for the theta-P duality in Noise-modulated Neural Networks (NNN),
verified via fully perturbation-based learning (no backprop through the activation):

    y = f(x ; theta, P)

where theta are the shared weight matrices and P is the per-unit noise field
(Gaussian std per neuron, per hidden layer).

BOTH theta and P are updated without backpropagating through the crossing activation:

  * theta-step : finite-difference node perturbation on hidden pre-activations
  * P-step     : finite-difference (SPSA) perturbation on noise-field parameters rho

Three conditions are compared:
  A. theta-only  : P fixed; theta updated by node perturbation
  B. field-only  : theta fixed (after optional short pre-training); P updated by SPSA
  C. dual        : theta-step then P-step each epoch

Task: one network, two noise fields P_A / P_B
    P_A active → approximate  sin(x)
    P_B active → approximate  sin(2x)

Noise field parameterisation:
    rho_A_l, rho_B_l  in  R^H   (one per hidden layer)
    P_A_l = softplus(rho_A_l),  P_B_l = softplus(rho_B_l)

Theta-step (hidden layers):
    For each hidden layer l, sample delta_l ~ N(0, I) of the same shape as d_l.
    Evaluate loss at d_l +/- eps_theta * delta_l.
    Update:
        grad_W_l  ~ (s * delta_l)^T @ z_{l-1} / N
        grad_b_l  ~ mean(s * delta_l)
    where s = (L_plus - L_minus) / (2 * eps_theta).

    Note: hidden layers are updated by finite-difference node perturbation;
          output layer uses direct readout gradient (analytical MSE gradient)
          for numerical stability.

P-step:
    Symmetric SPSA on rho:
        s = (L(P + eps*delta) - L(P - eps*delta)) / (2 * eps)
        rho -= lr_p * s * delta

Regularisation (in P-step):
    L_reg = lambda_sparse  * mean(P)
          + lambda_overlap * dot(normalise(P_A), normalise(P_B))   [per layer, averaged]

Three matplotlib figures are shown interactively after training completes:
    1. loss curves for all three conditions
    2. final predictions (all conditions)
    3. final noise fields P_A / P_B per condition and layer

Run:
    python examples/dual_finite_difference_node_perturbation.py
    python examples/dual_finite_difference_node_perturbation.py \\
        --epochs 2000 --hidden-dim 64 --lr-theta 1e-2 --seed 0
"""

from __future__ import annotations

import argparse
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Existing NNN API ---
from nnn.activation import CrossingAnalytic          # analytic crossing (used below)
from nnn.noise import gaussian_pdf_torch, gaussian_cdf_torch   # Gaussian PDF/CDF factories


# ============================================================
# Analytic crossing activation (wraps existing nnn API)
# ============================================================

def analytic_crossing(
    d: torch.Tensor,
    sigma: torch.Tensor,
    sigma_min: float = 1e-6,
) -> torch.Tensor:
    """
    phi_bar(d; sigma) = 2 * F(d; sigma) * (1 - F(d; sigma))
    F(d; sigma)       = 0.5 * (1 + erf(d / (sigma_eff * sqrt(2))))
    sigma_eff         = clamp(sigma, min=sigma_min)

    Uses nnn.activation.CrossingAnalytic + nnn.noise.{gaussian_pdf_torch, gaussian_cdf_torch}.
    sigma can be a scalar or a [H]-dim tensor (per-unit noise field).
    """
    sigma_eff = sigma.clamp(min=sigma_min)
    pdf = gaussian_pdf_torch(0.0, sigma_eff)
    cdf = gaussian_cdf_torch(0.0, sigma_eff)
    return CrossingAnalytic.apply(d, pdf, cdf)


# ============================================================
# Network: two hidden layers, manual-update-friendly
# ============================================================

class NNNfd(nn.Module):
    """
    Two-hidden-layer NNN for finite-difference node perturbation experiments.
    Structure: [1, H, H, 1]
    Activation: analytic_crossing with per-unit sigma (noise field) per hidden layer.
    All weight updates are performed manually (no optimizer).
    """

    def __init__(self, hidden_dim: int = 64, sigma_min: float = 1e-6):
        super().__init__()
        H = hidden_dim
        self.hidden_dim = H
        self.sigma_min  = sigma_min
        # fcs[0]: R^1  -> R^H  (hidden layer 1)
        # fcs[1]: R^H  -> R^H  (hidden layer 2)
        # fcs[2]: R^H  -> R^1  (output layer)
        self.fcs = nn.ModuleList([
            nn.Linear(1, H),
            nn.Linear(H, H),
            nn.Linear(H, 1),
        ])

    def forward_clean(
        self,
        x:    torch.Tensor,          # [N, 1]
        stds: list[torch.Tensor],    # [sigma_l1 [H], sigma_l2 [H]]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Clean forward (no perturbation).
        Returns (y [N,1], inputs_to_each_layer)
        where inputs_to_each_layer = [z0 [N,1], z1 [N,H], z2 [N,H]]
        and z_l is the activation fed into fcs[l].
        """
        z0 = x                                                     # input to fcs[0]
        d1 = self.fcs[0](z0)
        z1 = analytic_crossing(d1, stds[0], self.sigma_min)        # input to fcs[1]
        d2 = self.fcs[1](z1)
        z2 = analytic_crossing(d2, stds[1], self.sigma_min)        # input to fcs[2]
        y  = self.fcs[2](z2)
        return y, [z0, z1, z2]

    def forward_perturbed(
        self,
        x:      torch.Tensor,
        stds:   list[torch.Tensor],
        deltas: list[torch.Tensor],  # per-hidden-layer perturbation vectors [N, H]
        eps:    float,
    ) -> torch.Tensor:
        """
        Forward with eps * delta_l added to each hidden pre-activation d_l.
        Used to compute L_plus and L_minus for node perturbation gradient estimate.
        """
        z0 = x
        d1 = self.fcs[0](z0) + eps * deltas[0]
        z1 = analytic_crossing(d1, stds[0], self.sigma_min)
        d2 = self.fcs[1](z1) + eps * deltas[1]
        z2 = analytic_crossing(d2, stds[1], self.sigma_min)
        y  = self.fcs[2](z2)
        return y


# ============================================================
# Noise-field helpers
# ============================================================

def get_stds(rho: list[torch.Tensor]) -> list[torch.Tensor]:
    """rho (list of [H] tensors) -> P (list of softplus-[H] tensors)."""
    return [F.softplus(r) for r in rho]


def overlap_val(P_A: list[torch.Tensor], P_B: list[torch.Tensor]) -> torch.Tensor:
    """
    Mean cosine similarity of P_A and P_B across layers.
    normalize(P) = P / (norm(P) + 1e-8)
    overlap      = mean over layers of dot(normalize(P_A_l), normalize(P_B_l))
    """
    n = len(P_A)
    total = sum(
        ((pa / (pa.norm() + 1e-8)) * (pb / (pb.norm() + 1e-8))).sum()
        for pa, pb in zip(P_A, P_B)
    )
    return total / n


def reg_loss(
    P_A: list[torch.Tensor], P_B: list[torch.Tensor],
    lambda_sparse: float, lambda_overlap: float,
) -> torch.Tensor:
    """Sparsity + overlap regularisation (scalar tensor)."""
    n       = len(P_A)
    sparse  = sum(p.mean() for p in P_A + P_B) / (2 * n)
    overlap = overlap_val(P_A, P_B)
    return lambda_sparse * sparse + lambda_overlap * overlap


def init_rho(
    n_hidden: int,
    H:        int,
    device:   torch.device,
    bias:     float = 0.5,
    noise:    float = 0.1,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Initialize rho_A (front-half biased) and rho_B (back-half biased).
    Both share some overlap so the overlap regularisation has something to push against.

    softplus(bias) ~ 0.97  vs  softplus(0) ~ 0.69  → partial separation.
    """
    HALF  = H // 2
    rho_A = []
    rho_B = []
    for _ in range(n_hidden):
        a = noise * torch.randn(H, device=device)
        a[:HALF] += bias
        rho_A.append(a)

        b = noise * torch.randn(H, device=device)
        b[HALF:] += bias
        rho_B.append(b)
    return rho_A, rho_B


# ============================================================
# Theta-step: finite-difference node perturbation
# ============================================================

def theta_step(
    net:       NNNfd,
    x:         torch.Tensor,
    y_A:       torch.Tensor,
    y_B:       torch.Tensor,
    rho_A:     list[torch.Tensor],
    rho_B:     list[torch.Tensor],
    eps_theta: float,
    lr_theta:  float,
) -> tuple[float, float]:
    """
    Update theta using finite-difference node perturbation on pre-activations.

    Hidden layers: finite-difference node perturbation gradient estimate.
        For each hidden layer l:
            delta_l ~ N(0, I)  (shape [N, H])
            s       = (L(d_l + eps*delta_l) - L(d_l - eps*delta_l)) / (2*eps)
            grad_W  ~ (s * delta_l)^T @ z_{l-1} / N
            grad_b  ~ mean(s * delta_l)

    Output layer: direct readout gradient for numerical stability.
        dL/dy_pred = 2 * (y_pred - y_target) / N
        grad_W_out ~ (dL/dy_pred)^T @ z_last
        grad_b_out ~ mean(dL/dy_pred)

    Returns (loss_A, loss_B) evaluated from clean forward before the update.
    """
    N   = x.shape[0]
    H   = net.hidden_dim
    dev = x.device
    P_A = get_stds(rho_A)
    P_B = get_stds(rho_B)

    with torch.no_grad():
        # ---- Task A: finite-difference signal for hidden layers ----
        delta_A = [torch.randn(N, H, device=dev) for _ in range(2)]

        y_pred_A, z_A = net.forward_clean(x, P_A)          # clean; z_A stores inputs per layer

        y_Ap = net.forward_perturbed(x, P_A, delta_A, +eps_theta)
        y_Am = net.forward_perturbed(x, P_A, delta_A, -eps_theta)
        s_A  = (F.mse_loss(y_Ap, y_A) - F.mse_loss(y_Am, y_A)) / (2.0 * eps_theta)  # scalar

        # ---- Task B ----
        delta_B = [torch.randn(N, H, device=dev) for _ in range(2)]

        y_pred_B, z_B = net.forward_clean(x, P_B)

        y_Bp = net.forward_perturbed(x, P_B, delta_B, +eps_theta)
        y_Bm = net.forward_perturbed(x, P_B, delta_B, -eps_theta)
        s_B  = (F.mse_loss(y_Bp, y_B) - F.mse_loss(y_Bm, y_B)) / (2.0 * eps_theta)

        # ---- Hidden layer weight updates (summed over A and B tasks) ----
        # grad_W_l ~ (s * delta_l)^T @ z_{l-1} / N   shape: [H_out, H_in]
        # grad_b_l ~ mean(s * delta_l, dim=0)          shape: [H_out]
        for l in range(2):
            z_in_A  = z_A[l]              # input to hidden layer l under P_A  [N, H_in]
            z_in_B  = z_B[l]              # input to hidden layer l under P_B

            sig_dA  = s_A * delta_A[l]   # [N, H]
            sig_dB  = s_B * delta_B[l]

            grad_W  = (sig_dA.T @ z_in_A + sig_dB.T @ z_in_B) / N   # [H_out, H_in]
            grad_b  = (sig_dA + sig_dB).mean(0)                        # [H_out]

            net.fcs[l].weight -= lr_theta * grad_W
            net.fcs[l].bias   -= lr_theta * grad_b

        # ---- Output layer: direct readout gradient ----
        # hidden layers are updated by finite-difference node perturbation;
        # output layer uses direct readout gradient for stability
        z2_A = z_A[2]   # last hidden activation under P_A  [N, H]
        z2_B = z_B[2]

        dL_A = 2.0 * (y_pred_A - y_A) / N   # [N, 1]
        dL_B = 2.0 * (y_pred_B - y_B) / N

        net.fcs[2].weight -= lr_theta * (dL_A.T @ z2_A + dL_B.T @ z2_B)   # [1, H]
        net.fcs[2].bias   -= lr_theta * (dL_A + dL_B).mean(0)               # [1]

        lA = F.mse_loss(y_pred_A, y_A).item()
        lB = F.mse_loss(y_pred_B, y_B).item()

    return lA, lB


# ============================================================
# P-step: finite-difference noise-field perturbation (SPSA)
# ============================================================

def p_step(
    net:            NNNfd,
    x:              torch.Tensor,
    y_A:            torch.Tensor,
    y_B:            torch.Tensor,
    rho_A:          list[torch.Tensor],
    rho_B:          list[torch.Tensor],
    eps_p:          float,
    lr_p:           float,
    lambda_sparse:  float,
    lambda_overlap: float,
) -> None:
    """
    In-place SPSA update of rho_A and rho_B (sequential: A then B).

    For each rho:
        delta    ~ N(0, I)    (same shape as rho)
        L_plus   = task_loss(P + eps*delta) + reg(P + eps*delta, P_other)
        L_minus  = task_loss(P - eps*delta) + reg(P - eps*delta, P_other)
        grad_est = (L_plus - L_minus) / (2*eps) * delta
        rho     -= lr_p * grad_est
    """
    n_hidden = len(rho_A)

    with torch.no_grad():
        # ---- Update rho_A (rho_B fixed) ----
        delta_A = [torch.randn_like(rho_A[l]) for l in range(n_hidden)]
        P_B     = get_stds(rho_B)

        rA_p = [rho_A[l] + eps_p * delta_A[l] for l in range(n_hidden)]
        rA_m = [rho_A[l] - eps_p * delta_A[l] for l in range(n_hidden)]
        P_Ap = [F.softplus(r) for r in rA_p]
        P_Am = [F.softplus(r) for r in rA_m]

        yp, _ = net.forward_clean(x, P_Ap)
        ym, _ = net.forward_clean(x, P_Am)
        Lp = F.mse_loss(yp, y_A) + reg_loss(P_Ap, P_B, lambda_sparse, lambda_overlap)
        Lm = F.mse_loss(ym, y_A) + reg_loss(P_Am, P_B, lambda_sparse, lambda_overlap)

        s_A = (Lp - Lm).item() / (2.0 * eps_p)
        for l in range(n_hidden):
            rho_A[l] -= lr_p * s_A * delta_A[l]

        # ---- Update rho_B (updated rho_A now fixed) ----
        delta_B = [torch.randn_like(rho_B[l]) for l in range(n_hidden)]
        P_A     = get_stds(rho_A)

        rB_p = [rho_B[l] + eps_p * delta_B[l] for l in range(n_hidden)]
        rB_m = [rho_B[l] - eps_p * delta_B[l] for l in range(n_hidden)]
        P_Bp = [F.softplus(r) for r in rB_p]
        P_Bm = [F.softplus(r) for r in rB_m]

        yp, _ = net.forward_clean(x, P_Bp)
        ym, _ = net.forward_clean(x, P_Bm)
        Lp = F.mse_loss(yp, y_B) + reg_loss(P_A, P_Bp, lambda_sparse, lambda_overlap)
        Lm = F.mse_loss(ym, y_B) + reg_loss(P_A, P_Bm, lambda_sparse, lambda_overlap)

        s_B = (Lp - Lm).item() / (2.0 * eps_p)
        for l in range(n_hidden):
            rho_B[l] -= lr_p * s_B * delta_B[l]


# ============================================================
# Logging
# ============================================================

LOG_EVERY = 50   # record metrics every this many epochs


def log_entry(
    net:   NNNfd,
    x:     torch.Tensor,
    y_A:   torch.Tensor,
    y_B:   torch.Tensor,
    rho_A: list[torch.Tensor],
    rho_B: list[torch.Tensor],
) -> dict:
    P_A = get_stds(rho_A)
    P_B = get_stds(rho_B)
    with torch.no_grad():
        yA, _ = net.forward_clean(x, P_A)
        yB, _ = net.forward_clean(x, P_B)
        lA = F.mse_loss(yA, y_A).item()
        lB = F.mse_loss(yB, y_B).item()
        n  = len(P_A)
        mA = sum(p.mean().item() for p in P_A) / n
        mB = sum(p.mean().item() for p in P_B) / n
        ov = overlap_val(P_A, P_B).item()
    return {'loss_A': lA, 'loss_B': lB, 'total': lA + lB,
            'mean_P_A': mA, 'mean_P_B': mB, 'overlap': ov}


# ============================================================
# Experiment runners
# ============================================================

def _print_progress(tag: str, epoch: int, e: dict) -> None:
    print(f"  [{tag}] ep {epoch:5d}  "
          f"total={e['total']:.4e}  A={e['loss_A']:.4e}  B={e['loss_B']:.4e}  "
          f"ovlp={e['overlap']:.3f}")


def run_theta_only(
    net:   NNNfd,
    x:     torch.Tensor, y_A: torch.Tensor, y_B: torch.Tensor,
    rho_A: list[torch.Tensor], rho_B: list[torch.Tensor],
    args:  argparse.Namespace,
) -> list[dict]:
    """theta updated by node perturbation; rho fixed."""
    logs: list[dict] = []
    for epoch in range(args.epochs):
        theta_step(net, x, y_A, y_B, rho_A, rho_B, args.eps_theta, args.lr_theta)
        if epoch % LOG_EVERY == 0 or epoch == args.epochs - 1:
            e = log_entry(net, x, y_A, y_B, rho_A, rho_B)
            logs.append(e)
            if epoch % 500 == 0 or epoch == args.epochs - 1:
                _print_progress('theta-only', epoch, e)
    return logs


def run_field_only(
    net:   NNNfd,
    x:     torch.Tensor, y_A: torch.Tensor, y_B: torch.Tensor,
    rho_A: list[torch.Tensor], rho_B: list[torch.Tensor],
    args:  argparse.Namespace,
) -> list[dict]:
    """rho updated by SPSA; theta frozen (caller is responsible for not updating it)."""
    logs: list[dict] = []
    for epoch in range(args.epochs):
        p_step(net, x, y_A, y_B, rho_A, rho_B,
               args.eps_p, args.lr_p, args.lambda_sparse, args.lambda_overlap)
        if epoch % LOG_EVERY == 0 or epoch == args.epochs - 1:
            e = log_entry(net, x, y_A, y_B, rho_A, rho_B)
            logs.append(e)
            if epoch % 500 == 0 or epoch == args.epochs - 1:
                _print_progress('field-only', epoch, e)
    return logs


def run_dual(
    net:   NNNfd,
    x:     torch.Tensor, y_A: torch.Tensor, y_B: torch.Tensor,
    rho_A: list[torch.Tensor], rho_B: list[torch.Tensor],
    args:  argparse.Namespace,
) -> list[dict]:
    """theta-step then P-step each epoch."""
    logs: list[dict] = []
    for epoch in range(args.epochs):
        theta_step(net, x, y_A, y_B, rho_A, rho_B, args.eps_theta, args.lr_theta)
        p_step(net, x, y_A, y_B, rho_A, rho_B,
               args.eps_p, args.lr_p, args.lambda_sparse, args.lambda_overlap)
        if epoch % LOG_EVERY == 0 or epoch == args.epochs - 1:
            e = log_entry(net, x, y_A, y_B, rho_A, rho_B)
            logs.append(e)
            if epoch % 500 == 0 or epoch == args.epochs - 1:
                _print_progress('dual      ', epoch, e)
    return logs


# ============================================================
# Plotting
# ============================================================

def _ep(logs: list[dict]) -> list[int]:
    return [i * LOG_EVERY for i in range(len(logs))]


def save_loss_curves(
    logs_theta: list[dict], logs_field: list[dict], logs_dual: list[dict],
    args: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    palette = [('theta-only', logs_theta, 'steelblue'),
               ('field-only', logs_field, 'darkorange'),
               ('dual',       logs_dual,  'seagreen')]

    for label, logs, color in palette:
        ep = _ep(logs)
        axes[0].plot(ep, [l['total']   for l in logs], label=label, color=color, lw=1.5)
        axes[1].plot(ep, [l['mean_P_A'] for l in logs], label=f'{label} P_A', color=color, ls='-',  lw=1.2)
        axes[1].plot(ep, [l['mean_P_B'] for l in logs], label=f'{label} P_B', color=color, ls='--', lw=1.2)
        axes[2].plot(ep, [l['overlap'] for l in logs], label=label, color=color, lw=1.5)

    axes[0].set(title='Total loss  (L_A + L_B)', xlabel='epoch', ylabel='MSE')
    axes[0].set_yscale('log'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set(title='Mean noise-field std  E[P]', xlabel='epoch', ylabel='mean σ')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    axes[2].set(title='Overlap(P_A, P_B)  [cosine sim]', xlabel='epoch', ylabel='overlap')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.suptitle(
        f'NNN theta-P duality via finite-difference node perturbation  '
        f'(epochs={args.epochs}, H={args.hidden_dim}, '
        f'lr_θ={args.lr_theta}, lr_P={args.lr_p}, '
        f'ε_θ={args.eps_theta}, ε_P={args.eps_p})',
        fontsize=8,
    )
    plt.tight_layout()
    plt.show()


def save_predictions(
    net_theta: NNNfd, rA_theta: list, rB_theta: list,
    net_field: NNNfd, rA_field: list, rB_field: list,
    net_dual:  NNNfd, rA_dual:  list, rB_dual:  list,
    x: torch.Tensor, x_np: np.ndarray, yA_np: np.ndarray, yB_np: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    conditions = [
        (net_theta, rA_theta, rB_theta, 'theta-only', 'steelblue',  0.6, 1.2),
        (net_field, rA_field, rB_field, 'field-only', 'darkorange', 0.6, 1.2),
        (net_dual,  rA_dual,  rB_dual,  'dual',       'seagreen',   1.0, 2.5),
    ]
    for net, rA, rB, label, color, alpha, lw in conditions:
        with torch.no_grad():
            predA, _ = net.forward_clean(x, get_stds(rA))
            predB, _ = net.forward_clean(x, get_stds(rB))
        axes[0].plot(x_np.ravel(), predA.cpu().numpy().ravel(),
                     color=color, lw=lw, alpha=alpha, label=label)
        axes[1].plot(x_np.ravel(), predB.cpu().numpy().ravel(),
                     color=color, lw=lw, alpha=alpha, label=label)

    for ax, target, title in [
        (axes[0], yA_np, 'P_A active → sin(x)'),
        (axes[1], yB_np, 'P_B active → sin(2x)'),
    ]:
        ax.plot(x_np.ravel(), target.ravel(), 'k-', lw=2, label='target')
        ax.set(title=title, xlabel='x', ylabel='y')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle('Final predictions  (dual = bold line)', fontsize=10)
    plt.tight_layout()
    plt.show()


def save_fields(
    rA_theta: list, rB_theta: list,
    rA_field: list, rB_field: list,
    rA_dual:  list, rB_dual:  list,
) -> None:
    n_layers = 2
    fig, axes = plt.subplots(n_layers, 3, figsize=(15, 5))
    conditions = [
        (rA_theta, rB_theta, 'theta-only'),
        (rA_field, rB_field, 'field-only'),
        (rA_dual,  rB_dual,  'dual'),
    ]
    for col, (rA, rB, label) in enumerate(conditions):
        for l in range(n_layers):
            ax = axes[l][col]
            PA_l = F.softplus(rA[l]).detach().cpu().numpy()
            PB_l = F.softplus(rB[l]).detach().cpu().numpy()
            idx  = np.arange(len(PA_l))
            ax.bar(idx - 0.2, PA_l, width=0.4, color='steelblue',  alpha=0.8, label='P_A')
            ax.bar(idx + 0.2, PB_l, width=0.4, color='darkorange', alpha=0.8, label='P_B')
            ax.set_title(f'{label}  —  layer {l + 1}', fontsize=9)
            ax.set_xlabel('Neuron index'); ax.set_ylabel('σ'); ax.legend(fontsize=7)
            ax.grid(alpha=0.2, axis='y')

    fig.suptitle('Final noise fields P_A and P_B per condition and layer', fontsize=10)
    plt.tight_layout()
    plt.show()


# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='NNN theta-P duality via finite-difference node perturbation'
    )
    p.add_argument('--epochs',                     type=int,   default=2000)
    p.add_argument('--hidden-dim',                 type=int,   default=64)
    p.add_argument('--lr-theta',                   type=float, default=1e-2)
    p.add_argument('--lr-p',                       type=float, default=1e-2)
    p.add_argument('--eps-theta',                  type=float, default=1e-3)
    p.add_argument('--eps-p',                      type=float, default=1e-2)
    p.add_argument('--lambda-sparse',              type=float, default=1e-3)
    p.add_argument('--lambda-overlap',             type=float, default=1e-2)
    p.add_argument('--field-only-pretrain-epochs', type=int,   default=500)
    p.add_argument('--seed',                       type=int,   default=0)
    p.add_argument('--device',                     type=str,   default='cpu')
    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('NNN theta-P duality — finite-difference node perturbation')
    print(f'  Device      : {device}')
    print(f'  Hidden dim  : {args.hidden_dim}')
    print(f'  Epochs      : {args.epochs}')
    print('  Update rule : hidden layers via finite-difference node perturbation;')
    print('                output layer via direct readout gradient (analytical MSE).\n')

    # --- Dataset ---
    N      = 500
    x_np   = np.linspace(-2 * math.pi, 2 * math.pi, N).reshape(-1, 1)
    yA_np  = np.sin(x_np)
    yB_np  = np.sin(2 * x_np)
    x   = torch.tensor(x_np,  dtype=torch.float32, device=device)
    y_A = torch.tensor(yA_np, dtype=torch.float32, device=device)
    y_B = torch.tensor(yB_np, dtype=torch.float32, device=device)

    H = args.hidden_dim

    # Shared initial weight state (all conditions start from the same theta_0)
    _proto     = NNNfd(hidden_dim=H).to(device)
    init_state = deepcopy(_proto.state_dict())

    # Shared initial rho (same across conditions; front/back biased to avoid exact overlap)
    rho_A_init, rho_B_init = init_rho(n_hidden=2, H=H, device=device)

    # ----------------------------------------------------------------
    # A. theta-only
    # ----------------------------------------------------------------
    print('=== A. theta-only  (theta updated, P fixed) ===')
    net_theta = NNNfd(hidden_dim=H).to(device)
    net_theta.load_state_dict(deepcopy(init_state))
    rA_theta = [r.clone() for r in rho_A_init]
    rB_theta = [r.clone() for r in rho_B_init]
    logs_theta = run_theta_only(net_theta, x, y_A, y_B, rA_theta, rB_theta, args)

    # ----------------------------------------------------------------
    # B. field-only  (theta pre-trained by theta-only then frozen)
    # ----------------------------------------------------------------
    print(f'\n=== B. field-only  '
          f'(theta pre-trained for {args.field_only_pretrain_epochs} epochs, then frozen) ===')
    net_pre = NNNfd(hidden_dim=H).to(device)
    net_pre.load_state_dict(deepcopy(init_state))
    rho_A_tmp = [r.clone() for r in rho_A_init]
    rho_B_tmp = [r.clone() for r in rho_B_init]
    for ep in range(args.field_only_pretrain_epochs):
        theta_step(net_pre, x, y_A, y_B, rho_A_tmp, rho_B_tmp, args.eps_theta, args.lr_theta)
    with torch.no_grad():
        pre_e = log_entry(net_pre, x, y_A, y_B, rho_A_tmp, rho_B_tmp)
    print(f'  Pre-training done: total={pre_e["total"]:.4e}  — theta now frozen')

    net_field = net_pre   # theta frozen; no theta_step called below
    rA_field  = [r.clone() for r in rho_A_init]
    rB_field  = [r.clone() for r in rho_B_init]
    logs_field = run_field_only(net_field, x, y_A, y_B, rA_field, rB_field, args)

    # ----------------------------------------------------------------
    # C. dual
    # ----------------------------------------------------------------
    print('\n=== C. dual  (theta-step + P-step each epoch) ===')
    net_dual = NNNfd(hidden_dim=H).to(device)
    net_dual.load_state_dict(deepcopy(init_state))
    rA_dual = [r.clone() for r in rho_A_init]
    rB_dual = [r.clone() for r in rho_B_init]
    logs_dual = run_dual(net_dual, x, y_A, y_B, rA_dual, rB_dual, args)

    # ----------------------------------------------------------------
    # Console summary
    # ----------------------------------------------------------------
    print('\n' + '=' * 72)
    print('Final metrics (last logged epoch):')
    for name, logs in [('theta-only', logs_theta),
                       ('field-only', logs_field),
                       ('dual',       logs_dual)]:
        e = logs[-1]
        print(f'  {name:12s}  total={e["total"]:.5f}  '
              f'A={e["loss_A"]:.5f}  B={e["loss_B"]:.5f}  '
              f'mean_PA={e["mean_P_A"]:.4f}  mean_PB={e["mean_P_B"]:.4f}  '
              f'overlap={e["overlap"]:.4f}')
    print('=' * 72)
    print('\nInterpretation:')
    print('  theta-only  tests whether fixed fields can support learning by weight adaptation.')
    print('  field-only  tests whether changing only the noise field can select useful')
    print('              subnetworks under fixed weights.')
    print('  dual        tests whether weights form local mappings while noise fields')
    print('              select and separate functional regions.')
    print('\nNote: hidden layers updated by finite-difference node perturbation.')
    print('      Output layer updated by direct readout gradient for numerical stability.\n')

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    save_loss_curves(logs_theta, logs_field, logs_dual, args)
    save_predictions(
        net_theta, rA_theta, rB_theta,
        net_field, rA_field, rB_field,
        net_dual,  rA_dual,  rB_dual,
        x, x_np, yA_np, yB_np,
    )
    save_fields(
        rA_theta, rB_theta,
        rA_field, rB_field,
        rA_dual,  rB_dual,
    )


if __name__ == '__main__':
    main()
