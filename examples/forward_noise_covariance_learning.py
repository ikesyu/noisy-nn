"""
examples/forward_noise_covariance_learning.py

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
  backprop  : the SAME selected model, trained with ordinary autograd (Adam).
              Reference only -- it does use backward propagation.
  cov_only  : hidden credit = covariance only (no phi').  Manual, no autograd.
  cov_deriv : proposed -- covariance credit x noise-induced crossing derivative.

This is a proof-of-concept for hardware-friendly APPROXIMATE backpropagation, not an
exact replacement.  See docs/forward_noise_covariance_learning.md.

Run
---
    python examples/forward_noise_covariance_learning.py
    python examples/forward_noise_covariance_learning.py --noise uniform
    python examples/forward_noise_covariance_learning.py --epochs 1500 --num-samples 64

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


# ============================================================
# Build a library model and give layer-1 a spread "bump tiling" so the readout has a
# good basis (the default init does not tile the 1-D input).
# ============================================================
def build_model(noise: str, hidden: int, sigma: float, radius: float, h: float,
                t: int, device: torch.device):
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
        if self.kind == "sgd":
            param.data -= lr * grad
            return
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
        param.data -= lr * m_hat / (v_hat.sqrt() + self.eps)


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
              log_every: int = 0):
    """Manual covariance learning on a library Sample/UniformSample model.

    Hidden layers: covariance credit (+ optional phi') from captured per-sample
    activations; `credit` selects pooled vs per-input estimation.  `credit_passes`
    accumulates the statistics over several stochastic forward passes (variance
    reduction: effective samples = credit_passes * t).  Output layer: exact readout
    gradient on the ENSEMBLE-MEAN features (expected-value readout).  All no-grad.
    """
    assert method in ("cov_only", "cov_deriv")
    assert credit in ("pooled", "per_input")
    cap = Capture(net)
    n_hidden = cap.n_hidden
    N = x.shape[0]
    optim = ManualOpt(opt)
    losses, last_stats = [], {}
    for epoch in range(epochs):
        lr_t = lr_at(lr, epoch, epochs, lr_decay)
        lr_hidden = lr_t * hidden_lr_scale
        with torch.no_grad():
            # --- collect stochastic samples over credit_passes forward passes ---
            z_p = [[] for _ in range(n_hidden)]
            d_p = [[] for _ in range(n_hidden)]
            L_p, y_p = [], []
            for _ in range(credit_passes):
                y_p.append(net(x))                               # [N, 1]; fires hooks
                L_p.append((cap.y_samples.squeeze(-1) - t_target) ** 2)  # [N, T]
                for l in range(n_hidden):
                    z_p[l].append(cap.z[l])
                    d_p[l].append(cap.d[l])
            z = [torch.cat(z_p[l], dim=1) for l in range(n_hidden)]      # [N, K*T, H]
            d = [torch.cat(d_p[l], dim=1) for l in range(n_hidden)]
            L = torch.cat(L_p, dim=1)                                    # [N, K*T]
            y = torch.stack(y_p, dim=0).mean(dim=0)                      # [N, 1]
            T = L.shape[1]

            # z_prev for each hidden layer: input x (broadcast over T) then z_1
            z_prev = [x.unsqueeze(1).expand(N, T, x.shape[1]), z[0]]
            hidden_grads, stats = [], []
            for l in range(n_hidden):
                g_bcast, g_unit = covariance_credit(z[l], L, credit)     # ~ dL/dz
                if method == "cov_deriv":
                    delta_hat = g_bcast * phi_prime(d[l], noise, sigma, radius)
                else:
                    delta_hat = g_bcast.expand_as(z[l])
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
            dL_dy = 2.0 * (y - t_target)                         # [N, 1]
            gWout = torch.einsum("no,ni->oi", dL_dy, z_bar) / N  # [1, H]
            gbout = dL_dy.mean(dim=0)                            # [1]

            # --- apply updates via the manual optimiser (sgd or adam) ---
            for l in range(n_hidden):
                optim.update(f"w{l}", net.fcs[l].weight, hidden_grads[l][0], lr_hidden)
                if net.fcs[l].bias is not None:
                    optim.update(f"b{l}", net.fcs[l].bias, hidden_grads[l][1], lr_hidden)
            optim.update("wout", net.fcs[-1].weight, gWout, lr_t)
            if net.fcs[-1].bias is not None:
                optim.update("bout", net.fcs[-1].bias, gbout, lr_t)

            losses.append(float(((y - t_target) ** 2).mean()))
            last_stats = {"layer1": stats[0]}

        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  [{method}] epoch {epoch:5d}  eval_mse={losses[-1]:.5f}")
    cap.remove()
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
    styles = {"backprop": "--", "cov_only": "-.", "cov_deriv": "-"}
    for name, y in preds.items():
        plt.plot(x_raw[order], y[order], styles.get(name, "-"), lw=1.6, label=name)
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

    # identical initial weights for all three methods
    net0 = build_model(args.noise, args.hidden_dim, args.sigma, args.radius,
                       args.crossing_h, args.num_samples, device)
    init_state = copy.deepcopy(net0.state_dict())

    def fresh():
        n = build_model(args.noise, args.hidden_dim, args.sigma, args.radius,
                        args.crossing_h, args.num_samples, device)
        n.load_state_dict(init_state)
        return n

    print("\n[backprop] (reference, autograd Adam on the same model)")
    net_bp = fresh()
    bp_losses = train_backprop(net_bp, x, t, args.lr, args.epochs, log_every)

    print("\n[cov_only] covariance credit only (no phi')")
    net_co = fresh()
    co_losses, _ = train_cov(net_co, x, t, args.noise, args.sigma, args.radius,
                             "cov_only", args.lr, args.epochs, args.hidden_lr_scale,
                             args.credit, args.credit_passes, args.opt, args.lr_decay,
                             log_every)

    print("\n[cov_deriv] covariance credit x noise-induced derivative (proposed)")
    net_cd = fresh()
    cd_losses, cd_stats = train_cov(net_cd, x, t, args.noise, args.sigma, args.radius,
                                    "cov_deriv", args.lr, args.epochs, args.hidden_lr_scale,
                                    args.credit, args.credit_passes, args.opt,
                                    args.lr_decay, log_every)

    preds = {
        "backprop": predict(net_bp, x),
        "cov_only": predict(net_co, x),
        "cov_deriv": predict(net_cd, x),
    }

    plot_losses({"backprop": bp_losses, "cov_only": co_losses, "cov_deriv": cd_losses})
    plot_predictions(x_raw, target_np, preds)
    plot_activity_stats(cd_stats["layer1"])

    # low-noise final MSE from the averaged (multi-pass) prediction
    def final_mse(p):
        return float(np.mean((p - target_np) ** 2))
    fin = {k: final_mse(v) for k, v in preds.items()}
    improved = fin["cov_deriv"] < fin["cov_only"]
    print("\n================ SUMMARY ================")
    print(f"Model: nnn.{model_name}   (readout uses the ensemble-mean = expected value)")
    print("Final MSE:")
    for k in ("backprop", "cov_only", "cov_deriv"):
        print(f"  {k:10s}: {fin[k]:.5f}")
    print("\nInterpretation:")
    print("  - backprop is the exact-gradient reference (autograd on the same model).")
    print("  - cov_only  uses covariance credit ALONE for the hidden layers.")
    print("  - cov_deriv adds the noise-induced crossing derivative phi'(d).")
    print(f"  - cov_deriv {'IMPROVED over' if improved else 'did NOT improve over'} "
          f"cov_only (delta MSE = {fin['cov_only'] - fin['cov_deriv']:+.5f}).")
    print("=========================================")

    print("\nOpening 3 figure windows (close them to exit)...")
    plt.show()


if __name__ == "__main__":
    main()
