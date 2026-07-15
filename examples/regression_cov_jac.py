"""
Forward-noise covariance learning on y = sin(x): backprop vs cov_jac vs cov_jac_full.

A Noise-modulated Neural Network (NNN) is trained by backpropagation only at the
cost of "weight transport": the error must travel back through the TRANSPOSED
forward weights.  This example shows two learning rules that avoid it, built
entirely from `nnn.stats` and `nnn.credit`:

    backprop      reference.  autograd + Adam through the crossing activation.

    cov_jac       Both factors of the layer Jacobian are ESTIMATED from the noise
                  the forward pass already carries.
                    - weight mirror     W_hat = Cov_t(d_next, z) / Var_t(z)
                                        (nnn.credit.cov_weight)
                    - local slope       dz/dd from the crossing's own crossing
                                        counts (nnn.stats.kde_slope), which needs
                                        no knowledge of the noise distribution
                  The error then recurses exactly as in backprop,
                    delta^l = (W_hat[l+1]^T delta^{l+1}) * dz/dd^l,
                  but no transposed weight is ever read and no backward data path
                  exists.  The readout error 2(y - t) is still analytic.

    cov_jac_full  cov_jac with the readout error ALSO estimated from forward
                  statistics, g_y = (Cov_t(L, y) - m3) / Var_t(y), so no analytic
                  loss derivative appears anywhere.  The m3 term removes the
                  skewness bias of the readout fluctuation, which does not vanish
                  with more samples and which Adam would otherwise amplify late in
                  training.

All three start from the same initial weights.  The covariance rules run entirely
under no_grad; their only use of autograd is the single-layer re-run inside
`kde_slope`, which touches one activation and never the weights.

Two matplotlib figures are shown interactively (nothing is written to disk):
    1. learning curves and the recovered weight mirror W_hat vs the true W
    2. the fit to sin(x) and the residual of each method

Run:
    python examples/regression_cov_jac.py
    python examples/regression_cov_jac.py --epochs 300 --hidden-dim 32 --quick
    python examples/regression_cov_jac.py --noise uniform

Reference: docs/nce_draft.md sections 4.3 and 4.4.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from nnn import model
from nnn.credit import EPS, ManualOpt, cov_weight
from nnn.stats import Capture, kde_slope

NUM_POINTS = 128
UNIFORM_CENTER = 0.0
METHODS = ("backprop", "cov_jac", "cov_jac_full")
COLORS = {"backprop": "tab:blue", "cov_jac": "tab:red", "cov_jac_full": "tab:purple"}


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="backprop vs cov_jac vs cov_jac_full on y = sin(x)")
    p.add_argument('--noise',      choices=('gaussian', 'uniform'), default='gaussian',
                   help='Injected noise distribution  [gaussian]')
    p.add_argument('--epochs',     type=int,   default=1500,
                   help='Training epochs per method  [1500]')
    p.add_argument('--hidden-dim', type=int,   default=64,
                   help='Units per hidden layer  [64]')
    p.add_argument('--num-samples', type=int,  default=64,
                   help='T: stochastic samples drawn per input  [64]')
    p.add_argument('--lr',         type=float, default=1e-2,
                   help='Adam learning rate  [1e-2]')
    p.add_argument('--sigma',      type=float, default=0.5,
                   help='Gaussian noise std  [0.5]')
    p.add_argument('--radius',     type=float, default=1.0,
                   help='Uniform noise half-width  [1.0]')
    p.add_argument('--crossing-h', type=float, default=0.2,
                   help='Crossing threshold offset h  [0.2]')
    p.add_argument('--jac-ema',    type=float, default=0.9,
                   help='EMA rate of the weight mirrors  [0.9]')
    p.add_argument('--seed',       type=int,   default=0,
                   help='Random seed  [0]')
    p.add_argument('--quick',      action='store_true',
                   help='Small settings for a fast check')
    args = p.parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 150)
        args.hidden_dim = min(args.hidden_dim, 16)
        args.num_samples = min(args.num_samples, 16)
    return args


# ============================================================
# Task and model
# ============================================================

def make_task(device: torch.device):
    """y = sin(x) on x in [-2pi, 2pi]; the network input is x / pi."""
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, NUM_POINTS, dtype=np.float32)
    target = np.sin(x_raw).astype(np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    t = torch.tensor(target, device=device).unsqueeze(1)
    return x_raw, target, x, t


def build_model(args: argparse.Namespace, device: torch.device):
    """A [1, H, H, 1] sample-level NNN whose first layer tiles the input range.

    The default init does not spread the layer-1 bumps over the 1-D input, which
    leaves the readout without a usable basis, so the centres are tiled here.
    """
    structure = [1, args.hidden_dim, args.hidden_dim, 1]
    if args.noise == 'gaussian':
        net = model.SimpleNNNSample(structure=structure, std=args.sigma,
                                    h=args.crossing_h, t=args.num_samples,
                                    output_bias=True)
    else:
        net = model.SimpleNNNUniformSample(structure=structure, radius=args.radius,
                                           center=UNIFORM_CENTER, h=args.crossing_h,
                                           t=args.num_samples, output_bias=True)
    H = args.hidden_dim
    centres = torch.linspace(-2.0, 2.0, H)
    mag = 0.8 + 0.4 * torch.rand(H)
    sign = torch.where(torch.rand(H) < 0.5, -1.0, 1.0)
    w1 = (mag * sign).unsqueeze(1)
    with torch.no_grad():
        net.fcs[0].weight.copy_(w1)
        net.fcs[0].bias.copy_(-(w1.squeeze(1)) * centres)
    return net.to(device)


def predict(net, x: torch.Tensor, passes: int = 8) -> np.ndarray:
    """Ensemble prediction averaged over a few stochastic passes."""
    with torch.no_grad():
        y = torch.stack([net(x) for _ in range(passes)], dim=0).mean(dim=0)
    return y.squeeze(1).cpu().numpy()


# ============================================================
# Reference: backprop (the weight-transport baseline)
# ============================================================

def train_backprop(net, x: torch.Tensor, t: torch.Tensor,
                   args: argparse.Namespace) -> list[float]:
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    losses = []
    for epoch in range(args.epochs):
        opt.zero_grad()
        loss = ((net(x) - t) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if epoch % 300 == 0 or epoch == args.epochs - 1:
            print(f"  [backprop]     epoch {epoch:5d}  mse={losses[-1]:.5f}")
    return losses


# ============================================================
# Proposed: cov_jac / cov_jac_full (forward statistics only, no autograd)
# ============================================================

def readout_error(y, ys, L, t, jac_full: bool):
    """dL/dy at the readout: analytic 2(y - t), or estimated from forward noise.

    cov_jac_full regresses the per-sample loss on the readout fluctuation.  For a
    quadratic loss the population coefficient is 2(E[y] - t) + E[eps^3]/Var(eps),
    so the observed third central moment is subtracted to cancel the skew term.
    """
    if not jac_full:
        return 2.0 * (y - t)
    ys_f = ys.squeeze(-1)                                 # [N, T]
    cy = ys_f - ys_f.mean(dim=1, keepdim=True)
    cL = L - L.mean(dim=1, keepdim=True)
    cov_Ly = (cL * cy).mean(dim=1) - (cy ** 3).mean(dim=1)
    return (cov_Ly / ((cy ** 2).mean(dim=1) + EPS)).unsqueeze(1)


def train_cov_jac(net, x: torch.Tensor, t: torch.Tensor, method: str,
                  args: argparse.Namespace) -> tuple[list[float], dict]:
    """Weight-mirror credit with a local Adam step.  No transposed weights.

    Each epoch measures the forward weights from this pass's covariance, smooths
    them with an EMA, recurses the error down through the mirrors, and updates.
    After the step the mirrors are shifted by the same known increment, so they
    only have to pin down the static offset instead of chasing a moving target
    (Kolen-Pollack tracking).
    """
    jac_full = method == 'cov_jac_full'
    cap = Capture(net)
    n_hidden = cap.n_hidden
    N = x.shape[0]
    opt = ManualOpt('adam')
    W_ema: dict = {}
    losses = []

    for epoch in range(args.epochs):
        with torch.no_grad():
            y = net(x)                                    # [N, 1]; fires the hooks
            ys = cap.y_samples                            # [N, T, 1] pre-ensemble
            L = (ys.squeeze(-1) - t) ** 2                 # [N, T] per-sample loss
            z = [cap.z[l] for l in range(n_hidden)]       # [N, T, H]
            d = [cap.d[l] for l in range(n_hidden)]
            T = L.shape[1]

            # local slope dz/dd, from the crossing's own samples (distribution-free)
            slope = [kde_slope(cap.crossings[l], d[l]) for l in range(n_hidden)]
            slope_mean = [s.mean(dim=1) for s in slope]   # [N, H]

            # weight mirrors: measure from covariance, then EMA-smooth
            meas = {'out': cov_weight(ys, z[-1], pool=True)}
            for l in range(1, n_hidden):
                meas[l] = cov_weight(d[l], z[l - 1], pool=True)
            if not W_ema:
                W_ema.update(meas)
            else:
                for k, v in meas.items():
                    W_ema[k] = args.jac_ema * W_ema[k] + (1.0 - args.jac_ema) * v

            # recurse the error down the mirrors: a[l] ~= dL/dz[l]
            err = readout_error(y, ys, L, t, jac_full)    # [N, 1]
            a = [None] * n_hidden
            a[-1] = err * W_ema['out']
            for l in range(n_hidden - 2, -1, -1):
                a[l] = (a[l + 1] * slope_mean[l + 1]) @ W_ema[l + 1]

            # hidden gradients: the same outer product backprop would form
            z_prev = [x.unsqueeze(1).expand(N, T, x.shape[1])] + z[:-1]
            grads = []
            for l in range(n_hidden):
                delta = a[l].unsqueeze(1) * slope[l]      # [N, T, H]
                grads.append((torch.einsum('nto,nti->oi', delta, z_prev[l]) / (N * T),
                              delta.mean(dim=(0, 1))))

            # readout gradient on the ensemble-mean features (exact and local)
            z_bar = z[-1].mean(dim=1)                     # [N, H]
            gWout = torch.einsum('no,ni->oi', err, z_bar) / N
            gbout = err.mean(dim=0)

            steps = {}
            for l in range(n_hidden):
                steps[l] = opt.update(f'w{l}', net.fcs[l].weight, grads[l][0], args.lr)
                opt.update(f'b{l}', net.fcs[l].bias, grads[l][1], args.lr)
            steps['out'] = opt.update('wout', net.fcs[-1].weight, gWout, args.lr)
            opt.update('bout', net.fcs[-1].bias, gbout, args.lr)

            # the true weights just moved by a known amount: move the mirrors too
            W_ema['out'] = W_ema['out'] - steps['out']
            for l in range(1, n_hidden):
                W_ema[l] = W_ema[l] - steps[l]

            losses.append(float(((y - t) ** 2).mean()))

        if epoch % 300 == 0 or epoch == args.epochs - 1:
            print(f"  [{method:12s}] epoch {epoch:5d}  mse={losses[-1]:.5f}")

    cap.remove()
    mirror = {'W_hat': W_ema[1].cpu().numpy().ravel(),
              'W_true': net.fcs[1].weight.detach().cpu().numpy().ravel()}
    return losses, mirror


# ============================================================
# Plotting
# ============================================================

def plot_curves(losses: dict, mirror: dict, args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for name in METHODS:
        axes[0].semilogy(losses[name], color=COLORS[name], lw=1.4, label=name)
    axes[0].set_title('Learning curves (training MSE)')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('MSE (log scale)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # the mirror is the whole point of cov_jac: show that it recovers W
    w_hat, w_true = mirror['W_hat'], mirror['W_true']
    r = float(np.corrcoef(w_hat, w_true)[0, 1])
    lim = 1.05 * max(np.abs(w_hat).max(), np.abs(w_true).max())
    axes[1].plot([-lim, lim], [-lim, lim], 'k--', lw=1.0, alpha=0.6, label='y = x')
    axes[1].scatter(w_true, w_hat, s=6, alpha=0.35, color='tab:red')
    axes[1].set_title(f'Weight mirror of hidden layer 2  (Pearson r = {r:.4f})')
    axes[1].set_xlabel('true forward weight  W')
    axes[1].set_ylabel('recovered  W_hat = Cov(d, z) / Var(z)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f'Reconstructing backprop from forward fluctuations  '
        f'(noise={args.noise}, H={args.hidden_dim}, T={args.num_samples}, '
        f'epochs={args.epochs}, lr={args.lr})', fontsize=9)
    fig.tight_layout()


def plot_fit(x_raw, target, preds: dict) -> None:
    order = np.argsort(x_raw)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})
    axes[0].plot(x_raw[order], target[order], 'k-', lw=3.0, alpha=0.3,
                 label='target sin(x)')
    for name in METHODS:
        mse = float(np.mean((preds[name] - target) ** 2))
        axes[0].plot(x_raw[order], preds[name][order], color=COLORS[name], lw=1.6,
                     label=f'{name}  (MSE={mse:.2e})')
        axes[1].plot(x_raw[order], (preds[name] - target)[order], color=COLORS[name],
                     lw=1.1, label=name)
    axes[0].set_ylabel('y')
    axes[0].set_title('Fit to sin(x) after training (8-pass ensemble prediction)')
    axes[0].legend()
    axes[1].axhline(0.0, color='k', lw=0.6)
    axes[1].set_ylabel('residual')
    axes[1].set_xlabel('x')
    axes[1].legend(loc='upper right', fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    device = torch.device('cpu')

    print('Forward-noise covariance learning on y = sin(x)')
    print(f'  noise={args.noise}  H={args.hidden_dim}  T={args.num_samples}  '
          f'epochs={args.epochs}  lr={args.lr}  seed={args.seed}\n')

    x_raw, target, x, t = make_task(device)

    # all methods start from the same initial weights
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    init_state = copy.deepcopy(build_model(args, device).state_dict())

    def fresh():
        net = build_model(args, device)
        net.load_state_dict(init_state)
        return net

    losses, preds, mirror = {}, {}, None
    for name in METHODS:
        torch.manual_seed(args.seed)          # same noise stream for every method
        np.random.seed(args.seed)
        net = fresh()
        if name == 'backprop':
            losses[name] = train_backprop(net, x, t, args)
        else:
            losses[name], m = train_cov_jac(net, x, t, name, args)
            if name == 'cov_jac':
                mirror = m
        preds[name] = predict(net, x)

    print('\n' + '=' * 60)
    print('Final MSE (8-pass ensemble prediction):')
    for name in METHODS:
        print(f'  {name:14s}  {np.mean((preds[name] - target) ** 2):.5f}')
    print('=' * 60)
    print("""
cov_jac and cov_jac_full should land on the backprop curve.  Both build the
error recursion from forward statistics only: the weight mirror recovers W from
Cov(d, z)/Var(z) (see the scatter panel) and the local slope comes from the
crossing's own counts.  cov_jac_full additionally estimates the readout error,
so no analytic loss derivative is used anywhere.
""")

    plot_curves(losses, mirror, args)
    plot_fit(x_raw, target, preds)
    plt.show()


if __name__ == '__main__':
    main()
