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

  * SPSA climbs towards the sigma gradient's direction, but stops SHORT of it, and
    the budget it spends getting there is the strongest claim here.  Averaging K
    independent draws and taking the cosine against the sigma gradient (couple_h,
    seed 0) gives 0.048, 0.111, 0.192, 0.397, 0.655, 0.821 at K = 1 .. 1024.
    Fitting cos(K) = c/sqrt(1 + v/K) with c FREE (an earlier pass fixed c = 1, which
    assumed the answer) gives c_inf = 0.94, v = 297, and the pooled draws say the
    same with no fit at all: c_inf = 0.93, v = 249.  So:
        - v is the variance SPSA pays.  Theory says v = (d+1) + d*sigma_eta^2/|g|^2,
          i.e. its FLOOR is the field dimension; killing the loss noise (T -> 512)
          drives v to 72.3 against a floor of d+1 = 65.  The sigma gradient's cost
          does not move with d at all.
        - c_inf is NOT 1, so "the difference is variance alone" is false.  It is the
          sigma gradient's fidelity to the true field gradient, and 0.94 is well
          below the 0.998 the weight gradient reaches in the paper's 5.3.  It is not
          the Var(y) term either: c_inf plateaus at ~0.95 as T grows rather than
          climbing to 1, and giving the reference the same T does not move it, so
          the residue is structural in the cov_jac credit.
    The honest budget statement compares both against the SAME reference: SPSA needs
    v*c_inf^2/(1-c_inf^2) draws to be as close to the true gradient as the sigma
    gradient already is.  That is 4551 extra forwards here, 1721 with common random
    numbers (--spsa-crn, the strongest form of the baseline), 1657-7005 over seeds.
    The sigma gradient reads its direction off the T samples inference already drew:
    0 extra passes.  See docs/idea_duality.md 11.

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
    python tmp/duality_sigma_grad.py --compare-dir --couple-h 0.2 --spsa-crn
    python tmp/duality_sigma_grad.py --eval-t-sweep --couple-h 0.2 # is it Var(y)?
    python tmp/duality_sigma_grad.py --dim-sweep --couple-h 0.2    # v against dim
    python tmp/duality_sigma_grad.py --quick                       # 4 conditions, small
    python tmp/duality_sigma_grad.py --couple-h 0.2 --matched-step 0.01
    python tmp/duality_sigma_grad.py --noise uniform --couple-h 0.2 --matched-step 0.02

Figures are written to out/duality_sigma_grad/ (--out to change).
"""
from __future__ import annotations

import argparse
import copy
import math
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
    p.add_argument("--spsa-crn", action="store_true",
                   help="replay the same noise in the + and - SPSA passes (common "
                        "random numbers), the strongest form of the baseline")
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
    p.add_argument("--spsa-k", type=str, default="1,4,16,64,256,1024",
                   help="SPSA budgets K averaged for --compare-dir")
    p.add_argument("--spsa-m", type=int, default=4096,
                   help="independent SPSA draws pooled by --compare-dir. Every budget "
                        "K resamples these, and they also carry the moment estimate of "
                        "v, so the fitted curve has an independent cross-check")
    p.add_argument("--n-est", type=int, default=200,
                   help="passes averaged into the reference sigma gradient. Its own "
                        "noise attenuates c_inf, so this is measured (split-half) "
                        "rather than assumed")
    p.add_argument("--dim-sweep", action="store_true",
                   help="measure v at several field dimensions. SPSA's variance grows "
                        "with the dimension it must search, so v ~ dim turns the "
                        "budget gap into a scaling law")
    p.add_argument("--dim-list", type=str, default="8,16,32,64",
                   help="hidden widths H swept by --dim-sweep (field dim = 2H)")
    p.add_argument("--sweep-m", type=int, default=2048,
                   help="SPSA draws per width in --dim-sweep")
    p.add_argument("--eval-t-sweep", action="store_true",
                   help="measure c_inf against the T that SPSA's loss is averaged "
                        "over, with theta and the field held fixed. This isolates "
                        "the Var(y) term the sigma gradient drops")
    p.add_argument("--eval-t-list", type=str, default="8,32,128,512",
                   help="evaluation sample counts swept by --eval-t-sweep")
    p.add_argument("--out", type=str, default="out/duality_sigma_grad")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 120)
        args.hidden_dim = min(args.hidden_dim, 16)
        args.num_samples = min(args.num_samples, 16)
        args.check_passes = min(args.check_passes, 10)
        args.check_n = min(args.check_n, 1000)
        args.spsa_m = min(args.spsa_m, 512)
        args.sweep_m = min(args.sweep_m, 512)
        args.n_est = min(args.n_est, 20)
        args.spsa_k = "1,4,16,64"
        args.dim_list = "8,16"
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
    """One symmetric SPSA estimate of dL/dfield: two extra forward passes.

    --spsa-crn replays the same noise draws in both passes (common random numbers).
    That is the textbook way to stop the loss's own sampling noise from swamping a
    difference quotient, and it is the strongest form of this baseline, so it is
    offered here rather than assumed away.  Whether it helps is a question about the
    crossing: the field enters as a scale, xi = sigma * u, so replaying u makes the
    two passes differ smoothly in sigma -- but the crossing is binary, and a paired
    difference over binary responses is exactly what degenerated in Appendix B.
    """
    deltas = [torch.randn_like(v) for v in field]
    fp = [(v + args.eps_p * dv).clamp(min=args.sigma_min)
          for v, dv in zip(field, deltas)]
    fm = [(v - args.eps_p * dv).clamp(min=args.sigma_min)
          for v, dv in zip(field, deltas)]
    state = torch.get_rng_state() if args.spsa_crn else None
    lp = (float(((forward_with_field(net, x, fp, args) - t) ** 2).mean())
          + float(reg_value(*((fp, other) if a_first else (other, fp)), args)))
    if state is not None:
        torch.set_rng_state(state)
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
def pretrain_theta(net, cap, x, t, field, args, n_hidden: int):
    """Train theta by cov_jac for a while, so the field gradient is measured at a
    state the network actually reaches rather than at initialisation."""
    opt = ManualOpt("adam")
    W_ema: dict = {}
    pre = max(1, args.epochs // 5)
    for _ in range(pre):
        st = forward_stats(net, cap, x, t, field, args)
        update_mirrors(W_ema, st, n_hidden, args.jac_ema)
        a = jac_credit(st, W_ema, t, n_hidden)
        theta_step(net, opt, W_ema, [(st, a, t)], x, n_hidden, args.lr_theta)
    return W_ema, float(((st["y"] - t) ** 2).mean()), pre


def sigma_reference(net, cap, x, t, fa, fb, W_ema, args, n_hidden: int, n_est: int):
    """The sigma gradient of field A, averaged over `n_est` passes, flattened.

    Averaging damps the estimator's own variance but not its bias, which is the
    point: what stays as n_est grows is the systematic direction, and that is what
    c_inf below is meant to score.  The regulariser gradient is exact, so it is
    added once at the end.
    """
    acc = [torch.zeros(args.hidden_dim, device=x.device) for _ in range(n_hidden)]
    for _ in range(n_est):
        st = forward_stats(net, cap, x, t, fa, args)
        a = jac_credit(st, W_ema, t, n_hidden)
        for l, g in enumerate(sigma_grad(st, a, fa, n_hidden, args.sigma_min)):
            acc[l] += g / n_est
    greg, _ = reg_grads(fa, fb, args)
    return torch.cat([(g + r).reshape(-1) for g, r in zip(acc, greg)])


def spsa_draws(net, x, fa, fb, t, args, m: int) -> torch.Tensor:
    """`m` independent SPSA estimates of dL/dP_A, flattened into [m, dim].

    Pooling the draws once and resampling them for every budget K costs 2m forwards
    instead of 2*sum(K)*repeats, and the same draws then carry the moment estimate
    of v, so the fitted curve can be checked against a statistic that assumes no
    functional form at all.
    """
    rows = []
    for _ in range(m):
        g = spsa_direction(net, x, fa, fb, t, True, args)
        rows.append(torch.cat([v.reshape(-1) for v in g]))
    return torch.stack(rows)


def spsa_moments(G: torch.Tensor) -> tuple:
    """(|g_true|^2, tr Sigma, v) from the pooled draws, with no curve fitting.

    Each draw is unbiased, so E|mean|^2 = |g|^2 + tr(Sigma)/m overstates the signal
    by exactly the noise it carries; subtracting that term is what makes |g|^2 an
    estimate of the TRUE gradient's length rather than of this sample's.  Then
    v = tr(Sigma)/|g|^2 is the same v that appears in cos(K) = c_inf/sqrt(1 + v/K).
    """
    m = G.shape[0]
    gbar = G.mean(dim=0)
    tr_sigma = float(G.var(dim=0, unbiased=True).sum())
    g2 = float((gbar ** 2).sum()) - tr_sigma / m
    return g2, tr_sigma, tr_sigma / g2 if g2 > 0 else float("nan")


def cos_curve(G: torch.Tensor, u: torch.Tensor, ks, reps: int, gen) -> tuple:
    """Mean +- sd of cos(SPSA averaged over K draws, u), by resampling the pool."""
    mean, sd = [], []
    for K in ks:
        vals = []
        for _ in range(reps):
            idx = torch.randint(0, G.shape[0], (K,), generator=gen, device=G.device)
            gm = G[idx].mean(dim=0)
            vals.append(float(torch.nn.functional.cosine_similarity(gm, u, dim=0)))
        mean.append(float(np.mean(vals)))
        sd.append(float(np.std(vals)))
    return mean, sd


def fit_cos_curve(ks, cos, fix_c=None) -> tuple:
    """Least squares fit of cos(K) = c / sqrt(1 + v/K); returns (c, v, rms).

    `fix_c=1.0` reproduces the one-parameter fit, which ASSUMES the two estimators
    agree in the limit.  Leaving c free instead measures the limit, and c is linear
    given v, so only v needs searching.
    """
    k = np.asarray(ks, dtype=float)
    y = np.asarray(cos, dtype=float)

    def best_c(v):
        f = 1.0 / np.sqrt(1.0 + v / k)
        c = fix_c if fix_c is not None else float((y * f).sum() / (f * f).sum())
        return c, float(np.sqrt(((y - c * f) ** 2).mean()))

    grid = np.logspace(-2.0, 6.0, 4000)
    v = min(grid, key=lambda g: best_c(g)[1])
    fine = np.linspace(v * 0.5, v * 2.0, 4000)
    v = float(min(fine, key=lambda g: best_c(g)[1]))
    c, rms = best_c(v)
    return c, v, rms


def budget_for(target: float, c_inf: float, v: float):
    """K at which cos(SPSA_K, sigma_grad) reaches `target`, or None if unreachable.

    SPSA cannot exceed c_inf however large the budget, so a target above it has no
    answer: the gap there is not variance any more, it is the two estimators
    disagreeing about what the gradient is.
    """
    if target >= c_inf:
        return None
    return v / ((c_inf / target) ** 2 - 1.0)


def matching_budget(c_inf: float, v: float):
    """K at which SPSA is as close to the TRUE gradient as the sigma gradient is.

    cos(SPSA_K, g_true) = 1/sqrt(1 + v/K) and cos(sigma_grad, g_true) = c_inf, so
    the budget that buys SPSA the fidelity the sigma gradient already has is
    v*c_inf^2/(1 - c_inf^2).  This is the honest form of the budget claim: it
    compares both rules against the same reference instead of against each other.
    """
    if not 0.0 < c_inf < 1.0:
        return None
    return v * c_inf ** 2 / (1.0 - c_inf ** 2)


def measure_direction(args, device, seed_offset: int = 0) -> dict:
    """One (v, c_inf) measurement at the current args: the whole pipeline."""
    torch.manual_seed(args.seed + seed_offset)
    np.random.seed(args.seed + seed_offset)
    _, x, tA, tB = make_tasks(device)
    net = build_model(args, device)
    n_hidden = len(net.structure) - 2
    fa, fb = init_fields(args, n_hidden, device)
    cap = Capture(net)
    out = {"dim": n_hidden * args.hidden_dim, "H": args.hidden_dim}

    with torch.no_grad():
        W_ema, mse, pre = pretrain_theta(net, cap, x, tA, fa, args, n_hidden)
        out["pretrain_mse"], out["pretrain_epochs"] = mse, pre

        # two independent references: their cosine says how much of c_inf's shortfall
        # is just the reference's own noise (attenuation), and how much is systematic
        g1 = sigma_reference(net, cap, x, tA, fa, fb, W_ema, args, n_hidden,
                             args.n_est)
        g2 = sigma_reference(net, cap, x, tA, fa, fb, W_ema, args, n_hidden,
                             args.n_est)
        rho = float(torch.nn.functional.cosine_similarity(g1, g2, dim=0))
        g_sigma = 0.5 * (g1 + g2)
        out["split_half"] = rho

        G = spsa_draws(net, x, fa, fb, tA, args, args.spsa_m)
    cap.remove()

    g2n, tr_sigma, v_mom = spsa_moments(G)
    out["v_moment"], out["g_true_sq"], out["tr_sigma"] = v_mom, g2n, tr_sigma
    out["dim_plus_1"] = out["dim"] + 1

    gbar = G.mean(dim=0)
    cos_bar = float(torch.nn.functional.cosine_similarity(gbar, g_sigma, dim=0))
    out["c_moment"] = cos_bar * math.sqrt(1.0 + v_mom / G.shape[0])
    out["G"], out["g_sigma"] = G, g_sigma
    return out


def run_compare_dir(args, device) -> None:
    """cos(SPSA averaged over K draws, sigma gradient) as the budget K grows.

    SPSA is unbiased for the true field gradient, so averaging K draws converges on
    it.  Where that cosine goes as K grows therefore separates two very different
    claims: the RATE says how much variance SPSA is paying (v), and the LIMIT says
    whether the sigma gradient is pointing where the true gradient points (c_inf).
    Forcing the limit to 1, as the first pass did, assumes the second answer instead
    of measuring it, and the sigma gradient is not exactly unbiased for this
    objective: it estimates the signal term of E[L] = (E[y]-t)^2 + Var(y) and drops
    the variance term.  So c_inf is fitted here, not fixed.
    """
    print("=== direction: SPSA (budget K) vs the sigma gradient ===")
    ks = [int(v) for v in args.spsa_k.split(",") if v.strip()]
    r = measure_direction(args, device)
    print(f"  pretrained theta for {r['pretrain_epochs']} epochs, "
          f"mse={r['pretrain_mse']:.5f}")
    print(f"  field dim = {r['dim']},  SPSA draws pooled = {args.spsa_m}")
    print(f"  reference sigma gradient: split-half cos = {r['split_half']:.4f} "
          f"({args.n_est} passes each; attenuation "
          f"{math.sqrt(max(r['split_half'], 0.0)):.4f})")

    gen = torch.Generator(device=r["G"].device).manual_seed(args.seed + 1)
    cos_mean, cos_std = cos_curve(r["G"], r["g_sigma"], ks, 16, gen)
    for K, m, s in zip(ks, cos_mean, cos_std):
        print(f"  K = {K:5d}  ({2 * K:6d} extra forwards)   "
              f"cos(SPSA_K, sigma_grad) = {m:.3f} +- {s:.3f}")

    c1, v1, rms1 = fit_cos_curve(ks, cos_mean, fix_c=1.0)
    c2, v2, rms2 = fit_cos_curve(ks, cos_mean)
    print("\n  fits of cos(K) = c / sqrt(1 + v/K):")
    print(f"    c fixed at 1 : v = {v1:8.1f}                 rms = {rms1:.4f}")
    print(f"    c free       : v = {v2:8.1f}   c_inf = {c2:.4f}   rms = {rms2:.4f}")
    print(f"    moments      : v = {r['v_moment']:8.1f}   c_inf = {r['c_moment']:.4f}"
          f"   (no fit; from the {args.spsa_m} draws)")
    print(f"    dim + 1      : v = {r['dim_plus_1']:8.1f}   "
          f"(SPSA's variance with a noiseless loss)")

    print("\n  budget for SPSA to reach a given cos with the sigma gradient:")
    for target in (0.5, 0.9, 0.99):
        K = budget_for(target, c2, v2)
        if K is None:
            print(f"    cos = {target:.2f}   UNREACHABLE (above c_inf = {c2:.3f})")
        else:
            print(f"    cos = {target:.2f}   K = {K:9.0f}   "
                  f"{2 * K:10.0f} extra forwards")
    Km = matching_budget(c2, v2)
    if Km is not None:
        print(f"\n  SPSA matches the sigma gradient's own fidelity to g_true "
              f"(cos = {c2:.3f})\n    at K = {Km:.0f}, i.e. {2 * Km:.0f} extra "
              f"forwards.  The sigma gradient uses 0.")

    ax_k = np.logspace(0, math.log10(max(ks) * 8), 200)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(ks, cos_mean, yerr=cos_std, fmt="o", color="tab:red", capsize=4,
                zorder=3, label="measured: cos(SPSA averaged over K, sigma gradient)")
    ax.plot(ax_k, c2 / np.sqrt(1.0 + v2 / ax_k), color="tab:red", lw=1.4,
            label=f"fit c/sqrt(1+v/K):  c_inf = {c2:.3f}, v = {v2:.0f}")
    ax.plot(ax_k, 1.0 / np.sqrt(1.0 + v1 / ax_k), color="tab:gray", lw=1.2, ls="--",
            label=f"fit with c_inf forced to 1:  v = {v1:.0f}")
    ax.axhline(c2, color="tab:red", lw=0.8, ls=":", alpha=0.8)
    ax.annotate(f"c_inf = {c2:.3f}: SPSA converges HERE, not on 1.\n"
                f"The sigma gradient is not the true gradient either.",
                xy=(ks[0], c2), xytext=(ks[0] * 1.2, c2 + 0.04), fontsize=7,
                color="tab:red")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set(title="SPSA spends thousands of forwards climbing to a direction\n"
                 "the sigma gradient reads for free (and stops short of it)",
           xlabel="SPSA budget K  (2K extra forward passes)",
           ylabel="cosine with the sigma gradient", ylim=(-0.05, 1.05))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.suptitle(f"noise={args.noise}, H={args.hidden_dim}, T={args.num_samples}, "
                 f"couple_h={args.couple_h}, h={args.crossing_h}, "
                 f"dim={r['dim']}", fontsize=9)
    fig.tight_layout()
    savefig(fig, args.out_dir / "fig_spsa_vs_sigma_direction.png")


def run_eval_t_sweep(args, device) -> None:
    """c_inf against the T that SPSA's loss is averaged over, everything else fixed.

    The two rules do not estimate the same thing.  SPSA differences the loss the
    network actually incurs,

        E[L] = (E[y] - t)^2 + Var(y),   Var(y) = Var(single sample) / T,

    while the sigma gradient estimates the signal term only: the identity converts
    d zbar/d sigma, and zbar is a MEAN response, so nothing in it knows about the
    spread of y.  That is the caveat 9.7 flagged and left open.  It is testable:
    the variance term decays as 1/T, so if it is what separates the two, c_inf must
    climb towards 1 as T grows.  theta, the field and the reference sigma gradient
    are all held fixed here, so T is the only thing that moves and the sweep cannot
    be confounded by the network being retrained per T.
    """
    print("=== estimand: c_inf against the evaluation T (theta and field fixed) ===")
    ts = [int(v) for v in args.eval_t_list.split(",") if v.strip()]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _, x, tA, tB = make_tasks(device)
    net = build_model(args, device)
    n_hidden = len(net.structure) - 2
    fa, fb = init_fields(args, n_hidden, device)
    cap = Capture(net)
    rows = []

    with torch.no_grad():
        W_ema, mse, pre = pretrain_theta(net, cap, x, tA, fa, args, n_hidden)
        print(f"  pretrained theta for {pre} epochs at T={args.num_samples}, "
              f"mse={mse:.5f}")
        g1 = sigma_reference(net, cap, x, tA, fa, fb, W_ema, args, n_hidden,
                             args.n_est)
        g2 = sigma_reference(net, cap, x, tA, fa, fb, W_ema, args, n_hidden,
                             args.n_est)
        rho = float(torch.nn.functional.cosine_similarity(g1, g2, dim=0))
        g_sigma = 0.5 * (g1 + g2)
        print(f"  reference sigma gradient fixed at T={args.num_samples}: "
              f"split-half cos = {rho:.4f}")
        for T in ts:
            net.sampled_layer.numT = T
            G = spsa_draws(net, x, fa, fb, tA, args, args.spsa_m)
            _, _, v = spsa_moments(G)
            gbar = G.mean(dim=0)
            c = float(torch.nn.functional.cosine_similarity(gbar, g_sigma, dim=0))
            c_inf = c * math.sqrt(1.0 + v / G.shape[0])

            # the same comparison, but letting the sigma gradient read T samples too.
            # Its credit is built from ratios of T-sample covariances, which are
            # biased at finite T however many passes are averaged, so this separates
            # "the credit is short of samples" from "the credit is the wrong shape".
            h1 = sigma_reference(net, cap, x, tA, fa, fb, W_ema, args, n_hidden,
                                 args.n_est)
            h2 = sigma_reference(net, cap, x, tA, fa, fb, W_ema, args, n_hidden,
                                 args.n_est)
            rho_T = float(torch.nn.functional.cosine_similarity(h1, h2, dim=0))
            g_sig_T = 0.5 * (h1 + h2)
            c_T = float(torch.nn.functional.cosine_similarity(gbar, g_sig_T, dim=0))
            c_inf_T = c_T * math.sqrt(1.0 + v / G.shape[0])

            rows.append({"T": T, "v": v, "c_inf": c_inf,
                         "c_dis": c_inf / math.sqrt(max(rho, 1e-6)),
                         "c_matched": c_inf_T / math.sqrt(max(rho_T, 1e-6)),
                         "rho_T": rho_T})
            print(f"  T = {T:5d}   v = {v:8.1f}   c_inf = {c_inf:.4f}  "
                  f"(disatt {rows[-1]['c_dis']:.4f})   "
                  f"sigma_grad also at T: {rows[-1]['c_matched']:.4f}  "
                  f"(split-half {rho_T:.4f})")
        net.sampled_layer.numT = args.num_samples
    cap.remove()

    print("\n  If the Var(y) term were the whole gap, c_inf -> 1 as T grows.")
    print("  Where c_inf plateaus instead is the sigma gradient's own directional\n"
          "  bias against the signal term it is actually estimating.  The last\n"
          "  column gives that estimator the same T, so what survives there is\n"
          "  structural: the shape of the cov_jac credit, not a shortage of samples.")

    ts_a = np.array([q["T"] for q in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].semilogx(ts_a, [q["c_dis"] for q in rows], "o-", color="tab:red", base=2,
                     label="c_inf, sigma gradient fixed at the training T")
    axes[0].semilogx(ts_a, [q["c_matched"] for q in rows], "s--", color="tab:purple",
                     base=2, label="c_inf, sigma gradient given the same T")
    axes[0].axhline(1.0, color="k", lw=0.8, ls="--", label="agreement in the limit")
    axes[0].set(title="The gap is not the variance term: c_inf does not climb to 1",
                xlabel="T averaged into SPSA's loss  (Var(y) ~ 1/T)",
                ylabel="c_inf", ylim=(0.0, 1.05))
    axes[1].loglog(ts_a, [q["v"] for q in rows], "o-", color="tab:orange", base=2,
                   label="v = tr(Sigma)/|g|^2")
    axes[1].axhline(2 * args.hidden_dim + 1, color="tab:gray", lw=1.0, ls="-.",
                    label="dim + 1 (noiseless-loss floor)")
    axes[1].set(title="SPSA's variance does fall with T (its loss gets quieter)",
                xlabel="T averaged into SPSA's loss", ylabel="v")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"noise={args.noise}, H={args.hidden_dim}, couple_h={args.couple_h}, "
                 f"draws={args.spsa_m}, crn={args.spsa_crn}", fontsize=9)
    fig.tight_layout()
    savefig(fig, args.out_dir / "fig_cinf_vs_eval_T.png")


def run_dim_sweep(args, device) -> None:
    """v against the field dimension.

    SPSA replaces a d-dimensional gradient with one scalar times a random direction,
    so its variance carries the whole dimension: with a noiseless loss the algebra
    gives tr(Sigma)/|g|^2 = d + 1 exactly, and loss noise inflates that by a factor
    but leaves it proportional to d.  The sigma gradient reads every component from
    passes it was making anyway, so its cost does not move with d at all.  If v ~ d
    holds, the budget gap is not a constant factor to be tuned away: it widens with
    the size of the field being learned.
    """
    print("=== scaling: SPSA's variance v against the field dimension ===")
    hs = [int(v) for v in args.dim_list.split(",") if v.strip()]
    all_ks = [int(v) for v in args.spsa_k.split(",") if v.strip()]
    rows = []
    m_full = args.spsa_m
    for H in hs:
        args.hidden_dim = H
        # |g|^2 is recovered by subtracting tr(Sigma)/m from |mean|^2, so the draws
        # must outnumber v or that subtraction eats the signal and v is overstated.
        # v grows with the dimension, so the budget has to grow with it too.
        args.spsa_m = int(args.sweep_m * max(1.0, (2 * H) / 64.0))
        ks = [k for k in all_ks if k <= args.spsa_m // 2]
        r = measure_direction(args, device)
        gen = torch.Generator(device=r["G"].device).manual_seed(args.seed + 1)
        cos_mean, _ = cos_curve(r["G"], r["g_sigma"], ks, 16, gen)
        c2, v2, rms2 = fit_cos_curve(ks, cos_mean)
        rows.append({"H": H, "dim": r["dim"], "v_fit": v2, "c_fit": c2,
                     "v_moment": r["v_moment"], "c_moment": r["c_moment"],
                     "split_half": r["split_half"], "rms": rms2,
                     "mse": r["pretrain_mse"], "m": args.spsa_m,
                     "v_over_m": r["v_moment"] / args.spsa_m})
        print(f"  H = {H:4d}  dim = {r['dim']:4d}  m = {args.spsa_m:6d}   "
              f"v_moment = {r['v_moment']:8.1f}   v_fit = {v2:8.1f}   "
              f"c_inf = {c2:.3f}   (dim+1 = {r['dim'] + 1}, "
              f"v/m = {r['v_moment'] / args.spsa_m:.2f})")
    args.spsa_m = m_full
    print("  v/m must stay well under 1 for v_moment to be trustworthy.")

    dims = np.array([q["dim"] for q in rows], dtype=float)
    vmom = np.array([q["v_moment"] for q in rows], dtype=float)
    vfit = np.array([q["v_fit"] for q in rows], dtype=float)
    slope = float((dims * vmom).sum() / (dims ** 2).sum())     # v = slope * dim
    resid = vmom - slope * dims
    print(f"\n  v_moment = {slope:.2f} * dim   "
          f"(rms residual {float(np.sqrt((resid ** 2).mean())):.1f})")
    print(f"  ratio v/(dim+1) per width: "
          + ", ".join(f"{q['v_moment'] / (q['dim'] + 1):.2f}" for q in rows))
    print("  The floor is v = dim+1 (noiseless loss); the excess is the loss's own\n"
          "  sampling noise entering through the SPSA difference quotient.")
    print("\n  extra forwards for SPSA to match the sigma gradient's fidelity:")
    for q in rows:
        Km = matching_budget(q["c_fit"], q["v_moment"])
        s = f"{2 * Km:10.0f}" if Km is not None else "         -"
        print(f"    dim = {q['dim']:4d}   {s}   (sigma gradient: 0)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(dims, vmom, "o-", color="tab:red", label="v from the draws (moments)")
    axes[0].plot(dims, vfit, "s--", color="tab:orange", alpha=0.8,
                 label="v from the cos(K) fit")
    axes[0].plot(dims, slope * dims, ":", color="tab:red", alpha=0.6,
                 label=f"v = {slope:.1f} * dim")
    axes[0].plot(dims, dims + 1.0, "-.", color="tab:gray",
                 label="v = dim + 1 (noiseless-loss floor)")
    axes[0].set(title="SPSA's variance carries the whole field dimension",
                xlabel="field dimension d", ylabel="v = tr(Sigma) / |g|^2")
    axes[1].plot(dims, [2 * (matching_budget(q["c_fit"], q["v_moment"]) or np.nan)
                        for q in rows], "o-", color="tab:red",
                 label="SPSA: forwards to match the sigma gradient")
    axes[1].axhline(0.0, color="tab:blue", lw=1.6, label="sigma gradient: 0")
    axes[1].set(title="Cost of reaching the sigma gradient's fidelity",
                xlabel="field dimension d", ylabel="extra forward passes")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"noise={args.noise}, T={args.num_samples}, "
                 f"couple_h={args.couple_h}, draws={args.sweep_m}", fontsize=9)
    fig.tight_layout()
    savefig(fig, args.out_dir / "fig_spsa_variance_vs_dim.png")


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
    if args.dim_sweep:
        run_dim_sweep(args, device)
        return
    if args.eval_t_sweep:
        run_eval_t_sweep(args, device)
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
