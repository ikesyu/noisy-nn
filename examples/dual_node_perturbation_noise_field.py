"""
Dual verification experiment: weight θ vs noise-field P in NNN.

In a Noise-modulated Neural Network (NNN):

    y = f(x ; θ, P)

where θ are the shared linear weight matrices and P is the per-unit
Gaussian noise-std vector (noise field) that determines which neurons
receive stochastic activation.

The experiment trains ONE network on TWO tasks simultaneously using two
separate noise fields P_A and P_B:

    P_A active → approximate  sin(x)
    P_B active → approximate  sin(2x)

Three conditions are compared:

    A. theta-only  : P_A, P_B fixed at random init; θ updated by Adam backprop
    B. field-only  : θ fixed (copied from theta-only after n_pretrain epochs);
                     P_A, P_B updated by symmetric-perturbation SPSA
    C. dual        : θ updated by Adam (θ-step), then P_A/P_B updated by SPSA
                     (P-step) — both optimised jointly each epoch

The P parameterisation ensures positivity:

    P[l] = softplus(rho[l])     rho ∈ R^H

The SPSA gradient estimate for rho (per layer) is:

    g_est = (L(rho + ε·δ) - L(rho - ε·δ)) / (2ε)  ×  δ

where δ ∼ N(0, I) and L includes task MSE plus regularisation:

    reg = λ_sparse · mean(P) + λ_overlap · cosine_sim(P_A, P_B)

Two matplotlib figures are shown interactively after training completes:
    1. loss curves for all three conditions
    2. final predictions of the dual condition

Run:
    python examples/dual_node_perturbation_noise_field.py
    python examples/dual_node_perturbation_noise_field.py \\
        --epochs 2000 --lr-theta 3e-3 --lr-p 5e-3 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from nnn import model

# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='NNN theta-P duality verification via node perturbation'
    )
    p.add_argument('--epochs',         type=int,   default=1500,
                   help='Training epochs per condition  [1500]')
    p.add_argument('--lr-theta',       type=float, default=1e-2,
                   help='Adam learning rate for theta-step  [1e-2]')
    p.add_argument('--lr-p',           type=float, default=5e-3,
                   help='SPSA step size for P-step  [5e-3]')
    p.add_argument('--eps-p',          type=float, default=0.05,
                   help='SPSA perturbation magnitude for rho  [0.05]')
    p.add_argument('--lambda-sparse',  type=float, default=1e-3,
                   help='Sparsity regularisation weight  [1e-3]')
    p.add_argument('--lambda-overlap', type=float, default=1e-2,
                   help='Overlap regularisation weight  [1e-2]')
    p.add_argument('--seed',           type=int,   default=42,
                   help='Random seed  [42]')
    p.add_argument('--n-pretrain',     type=int,   default=300,
                   help='Theta-only epochs before field-only starts  [300]')
    return p.parse_args()


# ============================================================
# Helper functions
# ============================================================

def get_stds(rho: torch.Tensor) -> list[torch.Tensor]:
    """rho [n_layers, H] → list of n_layers softplus-H tensors."""
    return [F.softplus(rho[i]) for i in range(rho.shape[0])]


def task_losses(
    net, x: torch.Tensor,
    rho_A: torch.Tensor, rho_B: torch.Tensor,
    y_A: torch.Tensor,   y_B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    loss_A = F.mse_loss(net(x, stds=get_stds(rho_A)), y_A)
    loss_B = F.mse_loss(net(x, stds=get_stds(rho_B)), y_B)
    return loss_A, loss_B


def reg_loss(
    rho_A: torch.Tensor, rho_B: torch.Tensor,
    lambda_sparse: float, lambda_overlap: float,
) -> torch.Tensor:
    """Sparsity + overlap regularisation (scalar tensor)."""
    n = rho_A.shape[0]
    P_A = [F.softplus(rho_A[i]) for i in range(n)]
    P_B = [F.softplus(rho_B[i]) for i in range(n)]

    sparse = sum(p.mean() for p in P_A + P_B) / (2 * n)

    ov_terms = [
        ((pa / (pa.norm() + 1e-8)) * (pb / (pb.norm() + 1e-8))).mean()
        for pa, pb in zip(P_A, P_B)
    ]
    overlap = sum(ov_terms) / n

    return lambda_sparse * sparse + lambda_overlap * overlap


def overlap_scalar(rho_A: torch.Tensor, rho_B: torch.Tensor) -> float:
    """Mean cosine similarity of P_A and P_B across layers (metric only)."""
    n = rho_A.shape[0]
    total = 0.0
    with torch.no_grad():
        for i in range(n):
            pa = F.softplus(rho_A[i])
            pb = F.softplus(rho_B[i])
            na = pa / (pa.norm() + 1e-8)
            nb = pb / (pb.norm() + 1e-8)
            total += (na * nb).mean().item()
    return total / n


def p_step(
    net,
    rho_A: torch.Tensor, rho_B: torch.Tensor,
    x: torch.Tensor, y_A: torch.Tensor, y_B: torch.Tensor,
    lr_p: float, eps_p: float,
    lambda_sparse: float, lambda_overlap: float,
) -> None:
    """In-place SPSA update of rho_A and rho_B. No autograd through θ."""
    with torch.no_grad():
        # --- update rho_A (task A loss + reg, rho_B fixed) ---
        d = torch.randn_like(rho_A)
        rA_p = rho_A + eps_p * d
        rA_m = rho_A - eps_p * d

        Lp = (F.mse_loss(net(x, stds=get_stds(rA_p)), y_A)
              + reg_loss(rA_p, rho_B, lambda_sparse, lambda_overlap))
        Lm = (F.mse_loss(net(x, stds=get_stds(rA_m)), y_A)
              + reg_loss(rA_m, rho_B, lambda_sparse, lambda_overlap))

        rho_A -= lr_p * ((Lp - Lm) / (2.0 * eps_p)) * d

        # --- update rho_B (task B loss + reg, updated rho_A fixed) ---
        d = torch.randn_like(rho_B)
        rB_p = rho_B + eps_p * d
        rB_m = rho_B - eps_p * d

        Lp = (F.mse_loss(net(x, stds=get_stds(rB_p)), y_B)
              + reg_loss(rho_A, rB_p, lambda_sparse, lambda_overlap))
        Lm = (F.mse_loss(net(x, stds=get_stds(rB_m)), y_B)
              + reg_loss(rho_A, rB_m, lambda_sparse, lambda_overlap))

        rho_B -= lr_p * ((Lp - Lm) / (2.0 * eps_p)) * d


def log_entry(
    net, x: torch.Tensor,
    rho_A: torch.Tensor, rho_B: torch.Tensor,
    y_A: torch.Tensor, y_B: torch.Tensor,
) -> dict:
    net.eval()
    with torch.no_grad():
        lA, lB = task_losses(net, x, rho_A, rho_B, y_A, y_B)
        n = rho_A.shape[0]
        mA = sum(F.softplus(rho_A[i]).mean().item() for i in range(n)) / n
        mB = sum(F.softplus(rho_B[i]).mean().item() for i in range(n)) / n
    return {
        'loss_A':   lA.item(),
        'loss_B':   lB.item(),
        'total':    lA.item() + lB.item(),
        'mean_P_A': mA,
        'mean_P_B': mB,
        'overlap':  overlap_scalar(rho_A, rho_B),
    }


# ============================================================
# Experiment runners
# ============================================================

LOG_EVERY = 50   # record a log entry every this many epochs


def run_theta_only(
    net, rho_A: torch.Tensor, rho_B: torch.Tensor,
    x: torch.Tensor, y_A: torch.Tensor, y_B: torch.Tensor,
    args: argparse.Namespace,
) -> list[dict]:
    """θ updated by Adam; ρ fixed throughout."""
    opt = optim.Adam(net.parameters(), lr=args.lr_theta)
    logs: list[dict] = []

    for epoch in range(args.epochs):
        net.train()
        opt.zero_grad()
        lA, lB = task_losses(net, x, rho_A, rho_B, y_A, y_B)
        (lA + lB).backward()
        opt.step()

        if epoch % LOG_EVERY == 0 or epoch == args.epochs - 1:
            e = log_entry(net, x, rho_A, rho_B, y_A, y_B)
            logs.append(e)
            if epoch % 300 == 0 or epoch == args.epochs - 1:
                print(f"  [theta-only]  epoch {epoch:5d}  "
                      f"total={e['total']:.4e}  "
                      f"A={e['loss_A']:.4e}  B={e['loss_B']:.4e}  "
                      f"ovlp={e['overlap']:.3f}")
    return logs


def run_field_only(
    net, rho_A: torch.Tensor, rho_B: torch.Tensor,
    x: torch.Tensor, y_A: torch.Tensor, y_B: torch.Tensor,
    args: argparse.Namespace,
) -> list[dict]:
    """ρ updated by SPSA; θ frozen (caller is responsible for not passing optimizer)."""
    logs: list[dict] = []

    for epoch in range(args.epochs):
        net.eval()
        p_step(net, rho_A, rho_B, x, y_A, y_B,
               args.lr_p, args.eps_p, args.lambda_sparse, args.lambda_overlap)

        if epoch % LOG_EVERY == 0 or epoch == args.epochs - 1:
            e = log_entry(net, x, rho_A, rho_B, y_A, y_B)
            logs.append(e)
            if epoch % 300 == 0 or epoch == args.epochs - 1:
                print(f"  [field-only]  epoch {epoch:5d}  "
                      f"total={e['total']:.4e}  "
                      f"A={e['loss_A']:.4e}  B={e['loss_B']:.4e}  "
                      f"ovlp={e['overlap']:.3f}")
    return logs


def run_dual(
    net, rho_A: torch.Tensor, rho_B: torch.Tensor,
    x: torch.Tensor, y_A: torch.Tensor, y_B: torch.Tensor,
    args: argparse.Namespace,
) -> list[dict]:
    """θ-step (Adam) then P-step (SPSA) alternated every epoch."""
    opt = optim.Adam(net.parameters(), lr=args.lr_theta)
    logs: list[dict] = []

    for epoch in range(args.epochs):
        # θ-step
        net.train()
        opt.zero_grad()
        lA, lB = task_losses(net, x, rho_A, rho_B, y_A, y_B)
        (lA + lB).backward()
        opt.step()

        # P-step
        net.eval()
        p_step(net, rho_A, rho_B, x, y_A, y_B,
               args.lr_p, args.eps_p, args.lambda_sparse, args.lambda_overlap)

        if epoch % LOG_EVERY == 0 or epoch == args.epochs - 1:
            e = log_entry(net, x, rho_A, rho_B, y_A, y_B)
            logs.append(e)
            if epoch % 300 == 0 or epoch == args.epochs - 1:
                print(f"  [dual]        epoch {epoch:5d}  "
                      f"total={e['total']:.4e}  "
                      f"A={e['loss_A']:.4e}  B={e['loss_B']:.4e}  "
                      f"ovlp={e['overlap']:.3f}")
    return logs


# ============================================================
# Plotting
# ============================================================

def epochs_axis(logs: list[dict]) -> list[int]:
    return [i * LOG_EVERY for i in range(len(logs))]


def plot_loss_curves(
    logs_theta: list[dict],
    logs_field: list[dict],
    logs_dual:  list[dict],
    args: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    styles = [
        (logs_theta, 'theta-only', 'steelblue'),
        (logs_field, 'field-only', 'darkorange'),
        (logs_dual,  'dual',       'seagreen'),
    ]

    # Panel 1: total loss
    for logs, label, color in styles:
        ep = epochs_axis(logs)
        axes[0].plot(ep, [l['total'] for l in logs], label=label, color=color, lw=1.5)
    axes[0].set_title('Total loss  (L_A + L_B)')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('MSE')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Panel 2: mean noise-field magnitude (theta-only and dual)
    for logs, label, color in styles[:1] + styles[2:]:
        ep = epochs_axis(logs)
        axes[1].plot(ep, [l['mean_P_A'] for l in logs],
                     label=f'{label} P_A', color=color, ls='-',  lw=1.2)
        axes[1].plot(ep, [l['mean_P_B'] for l in logs],
                     label=f'{label} P_B', color=color, ls='--', lw=1.2)
    # field-only
    ep = epochs_axis(logs_field)
    axes[1].plot(ep, [l['mean_P_A'] for l in logs_field],
                 label='field-only P_A', color='darkorange', ls='-',  lw=1.2)
    axes[1].plot(ep, [l['mean_P_B'] for l in logs_field],
                 label='field-only P_B', color='darkorange', ls='--', lw=1.2)
    axes[1].set_title('Mean noise-field std  E[P]')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('mean std')
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    # Panel 3: overlap
    for logs, label, color in styles:
        ep = epochs_axis(logs)
        axes[2].plot(ep, [l['overlap'] for l in logs], label=label, color=color, lw=1.5)
    axes[2].set_title('Overlap(P_A, P_B)  [cosine similarity / layer]')
    axes[2].set_xlabel('epoch')
    axes[2].set_ylabel('overlap')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f'NNN θ–P duality  (epochs={args.epochs}, lr_θ={args.lr_theta}, '
        f'lr_P={args.lr_p}, ε_P={args.eps_p}, '
        f'λ_sparse={args.lambda_sparse}, λ_overlap={args.lambda_overlap})',
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()


def plot_predictions(
    net, rho_A: torch.Tensor, rho_B: torch.Tensor,
    x_np: np.ndarray, y_A_np: np.ndarray, y_B_np: np.ndarray,
) -> None:
    x_t = torch.tensor(x_np, dtype=torch.float32)
    net.eval()
    with torch.no_grad():
        pred_A = net(x_t, stds=get_stds(rho_A)).numpy().ravel()
        pred_B = net(x_t, stds=get_stds(rho_B)).numpy().ravel()

    x_plot = x_np.ravel()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x_plot, y_A_np.ravel(), 'k-', lw=2, label='target  sin(x)')
    axes[0].plot(x_plot, pred_A, '--', color='seagreen', lw=1.8,
                 label='network (P_A active)')
    axes[0].set_title('P_A active  →  sin(x)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(x_plot, y_B_np.ravel(), 'k-', lw=2, label='target  sin(2x)')
    axes[1].plot(x_plot, pred_B, '--', color='seagreen', lw=1.8,
                 label='network (P_B active)')
    axes[1].set_title('P_B active  →  sin(2x)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle('Final predictions — dual condition  (θ + P jointly optimised)', fontsize=10)
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Dataset ---
    N = 500
    x_np  = np.linspace(-2 * np.pi, 2 * np.pi, N).reshape(-1, 1)
    yA_np = np.sin(x_np)
    yB_np = np.sin(2 * x_np)
    x  = torch.tensor(x_np,  dtype=torch.float32)
    y_A = torch.tensor(yA_np, dtype=torch.float32)
    y_B = torch.tensor(yB_np, dtype=torch.float32)

    # --- Architecture ---
    STRUCTURE = [1, 64, 64, 1]
    n_hidden  = len(STRUCTURE) - 2
    H         = STRUCTURE[1]

    # --- Shared initial state (all conditions start from the same θ₀) ---
    # _net_proto = model.SimpleNNNAnalytic(structure=STRUCTURE, output_bias=False)
    _net_proto = model.SimpleNNNStatistic(structure=STRUCTURE, output_bias=False)
    init_state = deepcopy(_net_proto.state_dict())

    # --- Initial rho (same across conditions) ---
    # softplus(0.1 * randn) ≈ 0.693 ± small for all units initially
    rho_A_init = torch.randn(n_hidden, H) * 0.1
    rho_B_init = torch.randn(n_hidden, H) * 0.1

    # ----------------------------------------------------------------
    # A. theta-only
    # ----------------------------------------------------------------
    print('\n=== A. theta-only  (θ updated, P fixed) ===')
    # net_theta = model.SimpleNNNAnalytic(structure=STRUCTURE, output_bias=False)
    net_theta = model.SimpleNNNStatistic(structure=STRUCTURE, output_bias=False)
    net_theta.load_state_dict(deepcopy(init_state))
    rho_A_theta = rho_A_init.clone()
    rho_B_theta = rho_B_init.clone()
    logs_theta = run_theta_only(
        net_theta, rho_A_theta, rho_B_theta, x, y_A, y_B, args
    )

    # ----------------------------------------------------------------
    # B. field-only  (θ pre-trained by theta-only, then frozen; ρ updated)
    # ----------------------------------------------------------------
    print(f'\n=== B. field-only  (θ from {args.n_pretrain}-epoch pre-training, frozen) ===')

    # Pre-train θ for n_pretrain epochs with the same initial state
    # net_pre = model.SimpleNNNAnalytic(structure=STRUCTURE, output_bias=False)
    net_pre = model.SimpleNNNStatistic(structure=STRUCTURE, output_bias=False)
    net_pre.load_state_dict(deepcopy(init_state))
    opt_pre = optim.Adam(net_pre.parameters(), lr=args.lr_theta)
    rho_A_tmp = rho_A_init.clone()
    rho_B_tmp = rho_B_init.clone()
    net_pre.train()
    for ep in range(args.n_pretrain):
        opt_pre.zero_grad()
        lA, lB = task_losses(net_pre, x, rho_A_tmp, rho_B_tmp, y_A, y_B)
        (lA + lB).backward()
        opt_pre.step()
    net_pre.eval()
    with torch.no_grad():
        lA0, lB0 = task_losses(net_pre, x, rho_A_tmp, rho_B_tmp, y_A, y_B)
    print(f'  Pre-training done: total={lA0.item() + lB0.item():.4e} '
          f'(A={lA0.item():.4e}  B={lB0.item():.4e})  — θ now frozen')

    net_field = net_pre   # θ frozen; we never call an optimizer on it
    rho_A_field = rho_A_init.clone()
    rho_B_field = rho_B_init.clone()
    logs_field = run_field_only(
        net_field, rho_A_field, rho_B_field, x, y_A, y_B, args
    )

    # ----------------------------------------------------------------
    # C. dual  (θ-step + P-step each epoch, both from scratch)
    # ----------------------------------------------------------------
    print('\n=== C. dual  (θ-step + P-step each epoch) ===')
    # net_dual = model.SimpleNNNAnalytic(structure=STRUCTURE, output_bias=False)
    net_dual = model.SimpleNNNStatistic(structure=STRUCTURE, output_bias=False)
    net_dual.load_state_dict(deepcopy(init_state))
    rho_A_dual = rho_A_init.clone()
    rho_B_dual = rho_B_init.clone()
    logs_dual = run_dual(
        net_dual, rho_A_dual, rho_B_dual, x, y_A, y_B, args
    )

    # ----------------------------------------------------------------
    # Final metrics
    # ----------------------------------------------------------------
    print('\n' + '=' * 64)
    print('Final losses and overlap (last logged epoch):')
    for name, logs in [('theta-only', logs_theta),
                       ('field-only', logs_field),
                       ('dual',       logs_dual)]:
        e = logs[-1]
        print(f'  {name:12s}  total={e["total"]:.5f}  '
              f'A={e["loss_A"]:.5f}  B={e["loss_B"]:.5f}  '
              f'overlap={e["overlap"]:.4f}')
    print('=' * 64)

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    plot_loss_curves(logs_theta, logs_field, logs_dual, args)
    plot_predictions(net_dual, rho_A_dual, rho_B_dual, x_np, yA_np, yB_np)

    # ----------------------------------------------------------------
    # Result interpretation
    # ----------------------------------------------------------------
    print("""
Result interpretation
─────────────────────
theta-only  θ is updated with P fixed at its (random) initial values.
            If both tasks are learned, the fixed P already provides
            enough differentiation.  If not, P is the bottleneck.

field-only  θ is frozen after short pre-training.  The SPSA P-step
            adjusts which neurons receive noise.  Decreasing total loss
            shows that the noise field itself carries meaningful
            learning signal.  Watch for overlap(P_A, P_B) decreasing
            as the two fields specialise to different sub-networks.

dual        θ and P are jointly optimised.  The θ-step builds local
            input-output maps; the P-step selects which map to activate
            via noise routing.  If dual outperforms theta-only in total
            loss, the joint (θ, P) optimisation exploits the NNN duality.
            Decreasing overlap with lower loss is the key signature.

""")


if __name__ == '__main__':
    main()
