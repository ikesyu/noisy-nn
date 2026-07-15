"""
duality_sigma_grad.py — PoC for "direction A" of docs/idea_duality.md.

CLAIM.  The weights theta and the noise field P are not two systems driven by two
different learning machines.  ONE forward-covariance credit g_hat feeds both: pair
it with z_prev and you get the gradient of theta, pair it with -d/sigma and you get
the gradient of P.  No SPSA, no extra perturbation phase, no new statistic.

The identity behind it holds for any scale family p(xi) = (1/sigma) q(xi/sigma):

    dF/dsigma = -(d/sigma) p(d)   =>   d phi_bar/d sigma = -(d/sigma) phi_bar'(d)

and phi_bar'(d) is exactly what nnn.stats.kde_slope already estimates
distribution-free from the crossing's own counts.  So the field gradient is the
SAME three-factor product as the weight gradient, with z_prev swapped for -d/sigma:

    dL/dW[l]_kj    ~ < g_hat[l]_k * phi_T'(d[l]_k) * z[l-1]_j >
    dL/dsigma[l]_k ~ < g_hat[l]_k * phi_T'(d[l]_k) * (-d[l]_k / sigma[l]_k) >

TASK.  One shared theta ([1, H, H, 1]) and two noise fields:
    P_A active -> approximate sin(x)
    P_B active -> approximate sin(2x)
theta is trained by cov_jac (weight-mirror credit) in EVERY condition, so the
comparison isolates the P-step:

    theta_only       P frozen at its initial value                    (control)
    spsa             P by symmetric SPSA on the task loss             (baseline)
    sigma_cov_deriv  P by the sigma gradient, but with g_hat from the SCALAR
                     regression Cov(L, z)/Var(z)                      (H2)
    sigma_grad       P by the sigma gradient with cov_jac credit      (proposal)

Regularisation (identical wherever P moves) is an explicit function of the field
alone, so its gradient is local and needs no network backward:

    reg = lambda_sparse * mean(P) + lambda_overlap * cos(P_A, P_B)

WHAT IT FOUND (1000 epochs, H=32, T=32, seed 0; see docs/idea_duality.md 9 and 10)

  * The identity holds on the crossing AS IMPLEMENTED, and --couple-h makes it
    essentially exact.  Tying h_k = c*sigma_k turns zbar into a function of d/sigma
    alone, so d zbar/d sigma = -(d/sigma) d zbar/d d holds for any c:
        gaussian, fixed h=0.1    r = 0.9987   est/num = 0.922   <- 8% bias, stuck
        gaussian, couple_h=0.2   r = 0.9987   est/num = 0.996   <- bias gone
    Uniform keeps r ~ 0.83-0.90: its phi_bar' is discontinuous at the support edge,
    so the O(h^2) error piles up on the kink and coupling cannot remove it.

  * SPSA reads the SAME direction; the difference is variance alone.  Averaging K
    independent SPSA draws and taking the cosine against the sigma gradient:
        K =   1  ->  0.010     K =  64  ->  0.347
        K =   4  ->  0.143     K = 256  ->  0.641
    This fits cos(K) = 1/sqrt(1 + v/K) with v = 388 (residual rms 0.031), so the
    cosine converges on 1.  SPSA needs ~3310 extra forward passes to reach cos=0.9.
    The sigma gradient reads that direction off the T samples inference already
    drew: 0 extra passes.  This is the strongest and most setting-independent
    claim of direction A -- stronger than any loss comparison.

  * With both controls in place, the sigma gradient wins on loss too.  Final total
    MSE, spsa vs sigma_grad:
        fixed h=0.2, Adam            0.0755   0.1313   <- sigma loses
        fixed h=0.1, Adam            0.0615   0.0649   <- tie
        couple_h=0.2, Adam           0.0690   0.0723   <- tie
        couple_h=0.2, matched 0.01   0.0886   0.0703   <- sigma wins by 21%
        uniform, couple_h=0.2, matched 0.02
                                     0.0666   0.0459   <- sigma wins by 31%
    The two controls do different jobs.  Coupling h fixes the ESTIMATOR (sigma_grad
    0.1313 -> 0.0723).  Matching the step removes a CONFOUND: sigma_grad barely
    moves (0.0723 -> 0.0703) while spsa gets worse (0.0690 -> 0.0886), because
    forcing a near-random direction (cos ~ 0.01) to actually walk costs it.  SPSA
    looked good only because it was hardly moving the field.

  * H3 holds everywhere: sigma_grad reaches the lowest overlap of any rule, and the
    layer-1 fields come out nearly disjoint.  H2 holds: cov_jac credit beats
    cov_deriv credit by 39% (gaussian) / 30% (uniform).

Run
---
    python tmp/duality_sigma_grad.py --check --couple-h 0.2        # identity (fast)
    python tmp/duality_sigma_grad.py --compare-dir --couple-h 0.2  # direction vs K
    python tmp/duality_sigma_grad.py --quick                       # 4 conditions, small
    python tmp/duality_sigma_grad.py --couple-h 0.2 --matched-step 0.01
    python tmp/duality_sigma_grad.py --noise uniform --couple-h 0.2 --matched-step 0.02

Figures are written to out/duality_sigma_grad/ (--out to change).
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from nnn import model  # noqa: E402
from nnn.credit import ManualOpt, cov_weight, covariance_credit  # noqa: E402
from nnn.stats import Capture, crossing_layers, kde_slope  # noqa: E402

NUM_POINTS = 128
UNIFORM_CENTER = 0.0
CONDITIONS = ("theta_only", "spsa", "sigma_cov_deriv", "sigma_grad")
COLORS = {"theta_only": "tab:gray", "spsa": "tab:orange",
          "sigma_cov_deriv": "tab:green", "sigma_grad": "tab:red"}


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="theta-P duality via the sigma gradient")
    p.add_argument("--noise", choices=("gaussian", "uniform"), default="gaussian")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-samples", type=int, default=32, help="T per input")
    p.add_argument("--lr-theta", type=float, default=1e-2)
    p.add_argument("--lr-p", type=float, default=5e-3)
    p.add_argument("--sigma", type=float, default=0.5, help="base gaussian std")
    p.add_argument("--radius", type=float, default=1.0, help="base uniform half-width")
    p.add_argument("--sigma-min", type=float, default=0.02,
                   help="floor of the field (a dead unit must stay dead)")
    p.add_argument("--crossing-h", type=float, default=0.2,
                   help="fixed crossing threshold (ignored when --couple-h > 0)")
    p.add_argument("--couple-h", type=float, default=0.0,
                   help="tie the threshold to the field: h_k = couple_h * sigma_k. "
                        "This makes the crossing a pure function of d/sigma, which "
                        "makes the identity exact.  0 disables (fixed --crossing-h)")
    p.add_argument("--matched-step", type=float, default=0.0,
                   help="give every P-step the SAME field displacement per epoch: "
                        "normalise the update to this norm instead of using Adam. "
                        "0 disables (Adam with --lr-p)")
    p.add_argument("--jac-ema", type=float, default=0.9)
    p.add_argument("--eps-p", type=float, default=0.05, help="SPSA perturbation size")
    p.add_argument("--lambda-sparse", type=float, default=1e-3)
    p.add_argument("--lambda-overlap", type=float, default=1e-2)
    p.add_argument("--field-bias", type=float, default=0.3,
                   help="initial front/back split of P_A and P_B")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check", action="store_true",
                   help="only run the dL/dsigma fidelity check")
    p.add_argument("--check-h", type=float, default=0.02,
                   help="finite-difference step of the identity check")
    p.add_argument("--check-passes", type=int, default=40,
                   help="passes averaged into each E[zbar] of the identity check")
    p.add_argument("--check-n", type=int, default=4000,
                   help="batch size of the identity check")
    p.add_argument("--compare-dir", action="store_true",
                   help="only measure cos(SPSA direction, sigma-gradient direction) "
                        "as the SPSA sample budget K grows")
    p.add_argument("--spsa-k", type=str, default="1,4,16,64,256",
                   help="SPSA budgets K averaged for --compare-dir")
    p.add_argument("--out", type=str, default="out/duality_sigma_grad")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 120)
        args.hidden_dim = min(args.hidden_dim, 16)
        args.num_samples = min(args.num_samples, 16)
        args.check_passes = min(args.check_passes, 10)
        args.check_n = min(args.check_n, 1000)
    args.out_dir = Path(args.out)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return args


def base_scale(args) -> float:
    return args.sigma if args.noise == "gaussian" else args.radius


def savefig(fig, path: Path) -> None:
    fig.savefig(path, dpi=140)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {path} (+ .pdf)")


# ============================================================
# Task, model, noise field
# ============================================================
def make_tasks(device):
    """Two targets on one input grid: sin(x) for field A, sin(2x) for field B."""
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, NUM_POINTS, dtype=np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    tA = torch.tensor(np.sin(x_raw).astype(np.float32), device=device).unsqueeze(1)
    tB = torch.tensor(np.sin(2.0 * x_raw).astype(np.float32), device=device).unsqueeze(1)
    return x_raw, x, tA, tB


def build_model(args, device):
    """[1, H, H, 1] sample-level NNN with layer-1 bumps tiled over the input."""
    structure = [1, args.hidden_dim, args.hidden_dim, 1]
    if args.noise == "gaussian":
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


def init_fields(args, n_hidden: int, device):
    """P_A biased to the front half, P_B to the back half.

    The field IS the per-unit scale parameter (std, or uniform radius); there is no
    softplus reparameterisation, because the sigma gradient is available directly
    and a hardware field would be clipped rather than squashed.
    """
    H, s0, b = args.hidden_dim, base_scale(args), args.field_bias
    half = H // 2
    fa, fb = [], []
    for _ in range(n_hidden):
        ga = torch.full((H,), 1.0 - b, device=device)
        ga[:half] = 1.0 + b
        gb = torch.full((H,), 1.0 + b, device=device)
        gb[:half] = 1.0 - b
        fa.append((s0 * ga * (1.0 + 0.05 * torch.randn(H, device=device))
                   ).clamp(min=args.sigma_min))
        fb.append((s0 * gb * (1.0 + 0.05 * torch.randn(H, device=device))
                   ).clamp(min=args.sigma_min))
    return fa, fb


def clone_field(f):
    return [v.detach().clone() for v in f]


def threshold_for(args, scale):
    """The crossing threshold h: fixed, or tied to the field by --couple-h.

    Tying h to sigma makes zbar a pure function of d/sigma, so the crossing becomes
    a scale family in d and  d zbar/d sigma = -(d/sigma) d zbar/d d  holds EXACTLY,
    with no O(h^2) residue.  h is also the bandwidth of phi_T', so a fixed h floors
    how sharp the field can usefully get; tying it removes that floor.
    """
    if args.couple_h > 0.0:
        return args.couple_h * scale
    return args.crossing_h


def forward_with_field(net, x, field, args):
    """Run the net under a per-unit field, leaving that field on the crossing layers.

    A gaussian layer stores the std handed to `forward`, and a uniform layer reads
    `self.radius`, so after this call `kde_slope` re-runs a crossing layer under
    exactly this field.  The threshold is written on the layer either way.
    """
    layers = crossing_layers(net)
    if args.couple_h > 0.0:
        for cl, v in zip(layers, field):
            cl.h = threshold_for(args, v)
    if args.noise == "gaussian":
        return net(x, stds=list(field))
    for cl, r in zip(layers, field):
        cl.radius = r
    return net(x)


def overlap(fa, fb) -> float:
    """Mean cosine similarity of the two fields across layers."""
    tot = sum(float((a / (a.norm() + 1e-8) * (b / (b.norm() + 1e-8))).sum())
              for a, b in zip(fa, fb))
    return tot / len(fa)


def reg_value(fa, fb, args) -> torch.Tensor:
    n = len(fa)
    sparse = sum(v.mean() for v in fa + fb) / (2 * n)
    ov = sum((u / (u.norm() + 1e-8) * (v / (v.norm() + 1e-8))).sum()
             for u, v in zip(fa, fb)) / n
    return args.lambda_sparse * sparse + args.lambda_overlap * ov


def reg_grads(fa, fb, args):
    """Gradient of the regulariser w.r.t. each field.

    The regulariser is an explicit function of the field alone, so autograd here
    never touches the network: this stays local and weight-transport-free.
    """
    with torch.enable_grad():                        # the caller runs under no_grad
        a = [v.detach().clone().requires_grad_(True) for v in fa]
        b = [v.detach().clone().requires_grad_(True) for v in fb]
        reg_value(a, b, args).backward()
    return [v.grad for v in a], [v.grad for v in b]


# ============================================================
# Forward statistics and credit (nnn.stats / nnn.credit only)
# ============================================================
def forward_stats(net, cap, x, t, field, args) -> dict:
    """One stochastic pass, plus everything the credit rules read from it."""
    y = forward_with_field(net, x, field, args)           # [N, 1]; fires the hooks
    n_hidden = cap.n_hidden
    return {
        "y": y,
        "ys": cap.y_samples,                             # [N, T, 1] pre-ensemble
        "L": (cap.y_samples.squeeze(-1) - t) ** 2,       # [N, T] per-sample loss
        "z": [cap.z[l] for l in range(n_hidden)],
        "d": [cap.d[l] for l in range(n_hidden)],
        # dz/dd from the crossing's own counts, under the field just used
        "slope": [kde_slope(cap.crossings[l], cap.d[l]) for l in range(n_hidden)],
    }


def update_mirrors(W_ema: dict, st: dict, n_hidden: int, ema: float) -> None:
    """Measure W_hat = Cov(d_next, z)/Var(z) from this pass and EMA it in place.

    Both fields see the SAME theta, so every pass measures the same W and each one
    is a fresh observation of it.
    """
    meas = {"out": cov_weight(st["ys"], st["z"][-1], pool=True)}
    for l in range(1, n_hidden):
        meas[l] = cov_weight(st["d"][l], st["z"][l - 1], pool=True)
    if not W_ema:
        W_ema.update(meas)
        return
    for k, v in meas.items():
        W_ema[k] = ema * W_ema[k] + (1.0 - ema) * v


def jac_credit(st, W_ema, t, n_hidden: int) -> list:
    """cov_jac credit: recurse the readout error down the weight mirrors.

    Returns a[l] ~= dL_n/dz[l], of shape [N, H] per hidden layer.
    """
    slope_mean = [s.mean(dim=1) for s in st["slope"]]    # [N, H]
    a = [None] * n_hidden
    a[-1] = (2.0 * (st["y"] - t)) * W_ema["out"]         # [N,1]*[1,H] -> [N,H]
    for l in range(n_hidden - 2, -1, -1):
        a[l] = (a[l + 1] * slope_mean[l + 1]) @ W_ema[l + 1]
    return a


def deriv_credit(st, n_hidden: int) -> list:
    """cov_deriv credit: the scalar regression Cov(L, z)/Var(z), per input."""
    return [covariance_credit(st["z"][l], st["L"], "per_input")[0].squeeze(1)
            for l in range(n_hidden)]


def sigma_grad(st, a, field, n_hidden: int, sigma_min: float) -> list:
    """dL/dsigma[l] from the SAME credit that drives the weights.

    The three-factor product of the weight gradient with z_prev replaced by
    -d/sigma, since dz/dsigma = -(d/sigma) phi_bar'(d) for any scale family.
    """
    out = []
    for l in range(n_hidden):
        s = field[l].clamp(min=sigma_min).view(1, 1, -1)
        dz_dsigma = -(st["d"][l] / s) * st["slope"][l]                 # [N, T, H]
        out.append((a[l].unsqueeze(1) * dz_dsigma).mean(dim=(0, 1)))   # [H]
    return out


def hidden_grads(st, a, x, n_hidden: int):
    """dL/dW and dL/db of the hidden layers, from the same credit a[l]."""
    N, T = x.shape[0], st["L"].shape[1]
    z_prev = [x.unsqueeze(1).expand(N, T, x.shape[1])] + st["z"][:-1]
    out = []
    for l in range(n_hidden):
        delta = a[l].unsqueeze(1) * st["slope"][l]                     # [N, T, H]
        out.append((torch.einsum("nto,nti->oi", delta, z_prev[l]) / (N * T),
                    delta.mean(dim=(0, 1))))
    return out


def readout_grads(st, t):
    """dL/dW and dL/db of the readout, on the ensemble-mean features (exact, local)."""
    err = 2.0 * (st["y"] - t)                                          # [N, 1]
    z_bar = st["z"][-1].mean(dim=1)                                    # [N, H]
    return (torch.einsum("no,ni->oi", err, z_bar) / x_size(st),
            err.mean(dim=0))


def x_size(st) -> int:
    return st["y"].shape[0]


def theta_step(net, opt, W_ema, tasks, x, n_hidden: int, lr: float) -> None:
    """One Adam step on theta from the summed gradients of every task.

    `tasks` is a list of (stats, credit, target).  After the step the mirrors are
    shifted by the same known increment, so they only have to pin down the static
    offset instead of chasing a moving target (Kolen-Pollack tracking).
    """
    hid = [[torch.zeros_like(net.fcs[l].weight),
            torch.zeros_like(net.fcs[l].bias)] for l in range(n_hidden)]
    gWout = torch.zeros_like(net.fcs[-1].weight)
    gbout = torch.zeros_like(net.fcs[-1].bias)
    for st, a, t in tasks:
        for l, (gW, gb) in enumerate(hidden_grads(st, a, x, n_hidden)):
            hid[l][0] += gW
            hid[l][1] += gb
        gW, gb = readout_grads(st, t)
        gWout += gW
        gbout += gb

    steps = {}
    for l in range(n_hidden):
        steps[l] = opt.update(f"w{l}", net.fcs[l].weight, hid[l][0], lr)
        opt.update(f"b{l}", net.fcs[l].bias, hid[l][1], lr)
    steps["out"] = opt.update("wout", net.fcs[-1].weight, gWout, lr)
    opt.update("bout", net.fcs[-1].bias, gbout, lr)

    W_ema["out"] = W_ema["out"] - steps["out"]
    for l in range(1, n_hidden):
        W_ema[l] = W_ema[l] - steps[l]


# ============================================================
# P-step variants
# ============================================================
def apply_field_step(field, grads, opt, tag, args) -> None:
    """Move the field along `grads`, either by Adam or by a matched-norm step.

    --matched-step gives every rule the SAME displacement per epoch, so a
    comparison then reflects the DIRECTION of the estimate and nothing else.
    """
    if args.matched_step > 0.0:
        norm = torch.cat([g.reshape(-1) for g in grads]).norm() + 1e-12
        for v, g in zip(field, grads):
            v -= args.matched_step * g / norm
            v.clamp_(min=args.sigma_min)
        return
    for l, g in enumerate(grads):
        opt.update(f"p{tag}{l}", field[l], g, args.lr_p)
        field[l].clamp_(min=args.sigma_min)


def spsa_direction(net, x, field, other, t, a_first: bool, args) -> list:
    """One symmetric SPSA estimate of dL/dfield: two extra forward passes."""
    deltas = [torch.randn_like(v) for v in field]
    fp = [(v + args.eps_p * dv).clamp(min=args.sigma_min)
          for v, dv in zip(field, deltas)]
    fm = [(v - args.eps_p * dv).clamp(min=args.sigma_min)
          for v, dv in zip(field, deltas)]
    lp = (float(((forward_with_field(net, x, fp, args) - t) ** 2).mean())
          + float(reg_value(*((fp, other) if a_first else (other, fp)), args)))
    lm = (float(((forward_with_field(net, x, fm, args) - t) ** 2).mean())
          + float(reg_value(*((fm, other) if a_first else (other, fm)), args)))
    scale = (lp - lm) / (2.0 * args.eps_p)
    return [scale * dv for dv in deltas]


def spsa_step(net, x, fa, fb, tA, tB, opt, args) -> None:
    """SPSA on both fields.  P_A moves first and is then held fixed for P_B,
    matching the sequential scheme of the earlier examples/dual_* scripts."""
    for tag, field, other, t, a_first in (("a", fa, fb, tA, True),
                                          ("b", fb, fa, tB, False)):
        g = spsa_direction(net, x, field, other, t, a_first, args)
        apply_field_step(field, g, opt, tag, args)


def sigma_grad_step(cond, stats, creds, fa, fb, opt, args, n_hidden: int) -> None:
    """Update each field by the sigma gradient plus the local regulariser gradient."""
    ga_reg, gb_reg = reg_grads(fa, fb, args)
    for tag, field, greg in (("a", fa, ga_reg), ("b", fb, gb_reg)):
        st = stats[tag]
        a = creds[tag] if cond == "sigma_grad" else deriv_credit(st, n_hidden)
        gs = sigma_grad(st, a, field, n_hidden, args.sigma_min)
        apply_field_step(field, [g + r for g, r in zip(gs, greg)], opt, tag, args)


# ============================================================
# Training: one condition
# ============================================================
def run_condition(cond: str, net, x, tA, tB, fa, fb, args) -> list:
    """theta by cov_jac in every condition; only the P-step differs."""
    n_hidden = len(net.structure) - 2
    cap = Capture(net)
    opt_t, opt_p = ManualOpt("adam"), ManualOpt("adam")
    W_ema: dict = {}
    log = []

    for epoch in range(args.epochs):
        with torch.no_grad():
            stats, creds = {}, {}
            for tag, field, t in (("a", fa, tA), ("b", fb, tB)):
                st = forward_stats(net, cap, x, t, field, args)
                update_mirrors(W_ema, st, n_hidden, args.jac_ema)
                stats[tag], creds[tag] = st, jac_credit(st, W_ema, t, n_hidden)

            theta_step(net, opt_t, W_ema,
                       [(stats["a"], creds["a"], tA), (stats["b"], creds["b"], tB)],
                       x, n_hidden, args.lr_theta)

            if cond == "spsa":
                spsa_step(net, x, fa, fb, tA, tB, opt_p, args)
            elif cond in ("sigma_grad", "sigma_cov_deriv"):
                sigma_grad_step(cond, stats, creds, fa, fb, opt_p, args, n_hidden)

            lA = float(((stats["a"]["y"] - tA) ** 2).mean())
            lB = float(((stats["b"]["y"] - tB) ** 2).mean())
            log.append({"loss_A": lA, "loss_B": lB, "total": lA + lB,
                        "overlap": overlap(fa, fb),
                        "mean_P": float(sum(v.mean() for v in fa + fb)
                                        / (2 * n_hidden))})
        if epoch % 200 == 0 or epoch == args.epochs - 1:
            e = log[-1]
            print(f"  [{cond:15s}] epoch {epoch:5d}  total={e['total']:.4f}  "
                  f"A={e['loss_A']:.4f}  B={e['loss_B']:.4f}  ovlp={e['overlap']:.3f}")
    cap.remove()
    return log


# ============================================================
# Fidelity check: estimated dL/dsigma vs a CRN finite difference
# ============================================================
def crossing_layer_for(args, scale: float):
    """A single crossing layer, detached from any network."""
    h = threshold_for(args, scale)
    if args.noise == "gaussian":
        from nnn.layer import GaussianCrossingSampleLayer
        return GaussianCrossingSampleLayer(std=scale, h=h)
    from nnn.layer import UniformCrossingSampleLayer
    return UniformCrossingSampleLayer(radius=scale, center=UNIFORM_CENTER, h=h)


def layer_zbar(args, d, scale: float, reps: int) -> torch.Tensor:
    """E[zbar] of one crossing layer at pre-activation d, under scale `scale`."""
    cl = crossing_layer_for(args, scale)
    dd = d.view(1, 1, -1).expand(args.check_n, args.num_samples, d.numel()).contiguous()
    acc = torch.zeros_like(d)
    for _ in range(reps):
        acc += cl(dd).mean(dim=(0, 1)) / reps
    return acc


def run_check(args, device) -> None:
    """Is  dz/dsigma == -(d/sigma) * phi_T'(d)  for the crossing AS IMPLEMENTED?

    This is checked on ONE crossing layer, not through a network, and that is
    deliberate.  A network-level finite-difference reference does not work here:
    perturbing a single parameter moves the loss by less than the residual noise of
    E[y], and the binary crossing makes the paired difference degenerate (the
    negative result of the paper's Appendix B).  Under that protocol even the
    weight gradient of cov_jac, which the paper validates at cosine ~0.998 against
    autograd, scores only r ~ 0.67.  The instrument, not the estimator, is the
    limit.  So the identity is verified where it can be measured cleanly, and the
    network-level question is answered by the learning comparison instead.
    """
    print("=== identity check on one crossing layer ===")
    print(f"    d zbar/d sigma   vs   -(d/sigma) * kde_slope(d)\n")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    scale = base_scale(args)
    d = torch.linspace(-2.4 * scale, 2.4 * scale, 41)

    with torch.no_grad():
        num = ((layer_zbar(args, d, scale + args.check_h, args.check_passes)
                - layer_zbar(args, d, scale - args.check_h, args.check_passes))
               / (2.0 * args.check_h))
    cl = crossing_layer_for(args, scale)
    dd = d.view(1, 1, -1).expand(args.check_n, args.num_samples, d.numel()).contiguous()
    est = -(d / scale) * kde_slope(cl, dd).mean(dim=(0, 1))

    r = float(np.corrcoef(est.numpy(), num.numpy())[0, 1])
    ok = num.abs() > 0.02 * float(num.abs().max())
    ratio = float((est[ok] / num[ok]).mean())
    print(f"  Pearson r = {r:.4f}   mean(est / num) = {ratio:.3f}   "
          f"(scale of num = {float(num.abs().max()):.3f})")
    print(f"  note: h={args.crossing_h} costs O(h^2) here; try --crossing-h 0.1")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    lim = 1.05 * float(max(num.abs().max(), est.abs().max()))
    axes[0].plot([-lim, lim], [-lim, lim], "k--", lw=1.0, alpha=0.6, label="y = x")
    axes[0].scatter(num.numpy(), est.numpy(), s=26, alpha=0.7, color="tab:red")
    axes[0].set(title=f"identity (Pearson r = {r:.4f})",
                xlabel="numerical  d zbar / d sigma",
                ylabel="estimated  -(d/sigma) * phi_T'(d)")
    axes[1].plot(d.numpy(), num.numpy(), "k-", lw=2.5, alpha=0.4,
                 label="numerical  d zbar/d sigma")
    axes[1].plot(d.numpy(), est.numpy(), color="tab:red", lw=1.6, ls="--",
                 label="-(d/sigma) * phi_T'(d)")
    axes[1].set(title="the field sensitivity across the pre-activation",
                xlabel="pre-activation d", ylabel="d zbar / d sigma")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    h_tag = (f"h={args.couple_h}*sigma" if args.couple_h > 0
             else f"h={args.crossing_h} (fixed)")
    fig.suptitle(f"Direction A: dz/dsigma needs no new machinery, only phi_T'  "
                 f"(noise={args.noise}, scale={scale}, {h_tag}, "
                 f"T={args.num_samples})", fontsize=10)
    fig.tight_layout()
    savefig(fig, args.out_dir / "fig_sigma_identity.png")


# ============================================================
# Direction check: is SPSA just a noisy read of the sigma gradient?
# ============================================================
def run_compare_dir(args, device) -> None:
    """cos(SPSA averaged over K draws, sigma gradient) as the budget K grows.

    SPSA is unbiased, so averaging K independent draws converges on the true field
    gradient.  If the cosine against the sigma gradient climbs towards 1 as K grows,
    then the two agree on the direction and the only difference is variance: SPSA
    needs 2K forward passes to find what the sigma gradient reads off the passes the
    network was making anyway.  That is the claim of direction A, measured directly.
    """
    print("=== direction: SPSA (budget K) vs the sigma gradient ===")
    ks = [int(v) for v in args.spsa_k.split(",") if v.strip()]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _, x, tA, tB = make_tasks(device)
    net = build_model(args, device)
    n_hidden = len(net.structure) - 2
    fa, fb = init_fields(args, n_hidden, device)
    cap = Capture(net)
    W_ema: dict = {}

    with torch.no_grad():
        opt = ManualOpt("adam")
        pre = max(1, args.epochs // 5)
        for _ in range(pre):
            st = forward_stats(net, cap, x, tA, fa, args)
            update_mirrors(W_ema, st, n_hidden, args.jac_ema)
            a = jac_credit(st, W_ema, tA, n_hidden)
            theta_step(net, opt, W_ema, [(st, a, tA)], x, n_hidden, args.lr_theta)
        print(f"  pretrained theta for {pre} epochs, "
              f"mse={float(((st['y'] - tA) ** 2).mean()):.5f}")

        # the sigma gradient, averaged over a few passes to damp its own variance
        n_est = 20
        acc = [torch.zeros(args.hidden_dim, device=device) for _ in range(n_hidden)]
        for _ in range(n_est):
            st = forward_stats(net, cap, x, tA, fa, args)
            a = jac_credit(st, W_ema, tA, n_hidden)
            for l, g in enumerate(sigma_grad(st, a, fa, n_hidden, args.sigma_min)):
                acc[l] += g / n_est
        greg, _ = reg_grads(fa, fb, args)
        g_sigma = torch.cat([(g + r).reshape(-1) for g, r in zip(acc, greg)])

        # SPSA at each budget, plus a one-draw baseline repeated for a mean +- std
        cos_mean, cos_std = [], []
        for K in ks:
            vals = []
            for _ in range(8):                       # 8 independent budgets of K
                tot = [torch.zeros(args.hidden_dim, device=device)
                       for _ in range(n_hidden)]
                for _ in range(K):
                    for l, g in enumerate(spsa_direction(net, x, fa, fb, tA, True,
                                                         args)):
                        tot[l] += g / K
                g_spsa = torch.cat([g.reshape(-1) for g in tot])
                vals.append(float(torch.nn.functional.cosine_similarity(
                    g_spsa, g_sigma, dim=0)))
            cos_mean.append(float(np.mean(vals)))
            cos_std.append(float(np.std(vals)))
            print(f"  K = {K:4d}  ({2 * K:5d} extra forwards)   "
                  f"cos(SPSA_K, sigma_grad) = {cos_mean[-1]:.3f} +- {cos_std[-1]:.3f}")
    cap.remove()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(ks, cos_mean, yerr=cos_std, fmt="o-", color="tab:red", capsize=4,
                label="cos(SPSA averaged over K, sigma gradient)")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set(title="SPSA converges on the direction the sigma gradient reads for free",
           xlabel="SPSA budget K  (2K extra forward passes)",
           ylabel="cosine with the sigma gradient")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"noise={args.noise}, H={args.hidden_dim}, T={args.num_samples}, "
                 f"couple_h={args.couple_h}, h={args.crossing_h}", fontsize=9)
    fig.tight_layout()
    savefig(fig, args.out_dir / "fig_spsa_vs_sigma_direction.png")


# ============================================================
# Plots
# ============================================================
def plot_learning(logs, fields, args) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for c in CONDITIONS:
        ep = range(len(logs[c]))
        axes[0].semilogy(ep, [e["total"] for e in logs[c]], color=COLORS[c], lw=1.3,
                         label=c)
        axes[1].plot(ep, [e["overlap"] for e in logs[c]], color=COLORS[c], lw=1.3,
                     label=c)
        axes[2].plot(ep, [e["mean_P"] for e in logs[c]], color=COLORS[c], lw=1.3,
                     label=c)
    axes[0].set(title="Total loss  (L_A + L_B)", xlabel="epoch", ylabel="MSE")
    axes[1].set(title="Overlap(P_A, P_B)", xlabel="epoch", ylabel="cosine similarity")
    axes[2].set(title="Mean field  E[P]", xlabel="epoch", ylabel="mean scale")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"theta-P duality: one covariance credit drives both  "
                 f"(noise={args.noise}, H={args.hidden_dim}, T={args.num_samples}, "
                 f"epochs={args.epochs})", fontsize=9)
    fig.tight_layout()
    savefig(fig, args.out_dir / "fig_duality_learning.png")

    fa, fb = fields["sigma_grad"]
    n_hidden = len(fa)
    fig, axes = plt.subplots(n_hidden, 1, figsize=(9, 3.2 * n_hidden), squeeze=False)
    for l in range(n_hidden):
        idx = np.arange(len(fa[l]))
        axes[l][0].bar(idx - 0.2, fa[l].cpu().numpy(), width=0.4, color="tab:blue",
                       alpha=0.85, label="P_A (sin x)")
        axes[l][0].bar(idx + 0.2, fb[l].cpu().numpy(), width=0.4, color="tab:orange",
                       alpha=0.85, label="P_B (sin 2x)")
        axes[l][0].set(title=f"learned field, hidden layer {l + 1}",
                       xlabel="unit index", ylabel="scale")
        axes[l][0].legend(fontsize=8)
        axes[l][0].grid(alpha=0.2, axis="y")
    fig.suptitle("Fields learned by sigma_grad", fontsize=10)
    fig.tight_layout()
    savefig(fig, args.out_dir / "fig_duality_fields.png")


# ============================================================
# Main
# ============================================================
def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    print("theta-P duality PoC (direction A of docs/idea_duality.md)")
    print(f"  noise={args.noise}  H={args.hidden_dim}  T={args.num_samples}  "
          f"epochs={args.epochs}  seed={args.seed}\n")

    if args.check:
        run_check(args, device)
        return
    if args.compare_dir:
        run_compare_dir(args, device)
        return

    _, x, tA, tB = make_tasks(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    proto = build_model(args, device)
    n_hidden = len(proto.structure) - 2
    init_state = copy.deepcopy(proto.state_dict())
    fa0, fb0 = init_fields(args, n_hidden, device)

    logs, fields = {}, {}
    for cond in CONDITIONS:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        net = build_model(args, device)
        net.load_state_dict(init_state)
        fa, fb = clone_field(fa0), clone_field(fb0)
        logs[cond] = run_condition(cond, net, x, tA, tB, fa, fb, args)
        fields[cond] = (fa, fb)

    print("\n" + "=" * 74)
    print("Final (last epoch):")
    for cond in CONDITIONS:
        e = logs[cond][-1]
        print(f"  {cond:16s}  total={e['total']:.4f}  A={e['loss_A']:.4f}  "
              f"B={e['loss_B']:.4f}  overlap={e['overlap']:.4f}  "
              f"mean_P={e['mean_P']:.3f}")
    print("=" * 74)
    plot_learning(logs, fields, args)


if __name__ == "__main__":
    main()
