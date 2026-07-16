"""
duality_gauge_check.py — what h = c*sigma does to the theta-P duality.

Tying the crossing threshold to the field (--couple-h, section 10.1 of
docs/idea_duality.md) was adopted because it makes the sigma identity exact.  It
also, unnoticed at the time, makes zbar a function of d/sigma ALONE, and that has
a consequence the memo's framing did not survive:

    a_k -> alpha_k a_k,   sigma_k -> alpha_k sigma_k      leaves f unchanged

for every hidden unit k independently.  The noise enters as xi = sigma*u and the
threshold as h = c*sigma, so under a shared noise draw d, xi and h all scale by
alpha_k and the crossing, being scale invariant in (signal, threshold), cancels it.
The invariance is therefore PATHWISE and exact, not a statement about expectations.

This script measures the four things that follow, all to machine precision:

  1. INVARIANCE, and why depth does not save it.  The gauge leaves the unit's
     OUTPUT alone, so z^(l) is bit-identical and the next layer sees nothing to
     compensate for.  This is unlike the positive homogeneity of a ReLU net, where
     scaling a unit's weights rescales its output and the NEXT layer's weights must
     absorb it; there the symmetry is a conspiracy between adjacent layers, here it
     closes inside one unit.  That is why it composes to any depth.

  2. HOW BIG the redundancy is.  One gauge dof per hidden unit, one sigma per
     hidden unit: for a single context the field can be gauged flat (sigma := 1)
     with the function unchanged, so it carries no functional information at all.
     Two contexts share a_k, so alpha_k must move sigma_A,k and sigma_B,k together:
     half the field is gauge and the surviving half is the RATIO sigma_B/sigma_A.

  3. THE GAUGE IDENTITY.  Invariance forces  dL/dlog sigma_k + a_k . dL/da_k = 0,
     so the sigma gradient is the negated radial part of the weight gradient rather
     than an independent second reading.  Note the PoC's estimators satisfy this
     identically whatever h is, because both are built from the same credit and the
     same slope; the MODEL only satisfies it under coupling.  With a fixed h the
     estimator is therefore imposing a symmetry the network does not have.

  4. WHAT IT COSTS THE METRICS.  overlap = cos(P_A, P_B) and mean(P) are not gauge
     invariant, so two parameter settings computing bit-identical functions score
     differently on both.

Run
---
    python tmp/duality_gauge_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "tmp")):
    if p not in sys.path:
        sys.path.insert(0, p)

from duality_sigma_grad import (build_model, forward_stats, forward_with_field,  # noqa: E402
                                hidden_grads, init_fields, jac_credit, make_tasks,
                                parse_args, sigma_grad, update_mirrors)
from nnn.stats import Capture  # noqa: E402


def run(net, x, field, args, state):
    """One forward under a FIXED noise stream; returns y and every hidden z.

    Replaying the stream is what makes the invariance checkable exactly: the gauge
    scales xi = sigma*u, so it only cancels when u is held.
    """
    cap = Capture(net)
    torch.set_rng_state(state)
    with torch.no_grad():
        y = forward_with_field(net, x, field, args)
    z = [cap.z[l].clone() for l in range(cap.n_hidden)]
    cap.remove()
    return y, z


def apply_gauge(net, field, alphas, args, device):
    """Copy of (net, field) with unit k of layer l scaled by alphas[l][k].

    Row k of fcs[l] IS unit k's incoming weight vector, and unit k's own sigma sits
    in field[l][k], so the whole orbit is local to the unit.  fcs[-1] (the readout)
    is never touched: it reads z, which does not move.
    """
    net2 = build_model(args, device)
    net2.load_state_dict(net.state_dict())
    f2 = [v.clone() for v in field]
    with torch.no_grad():
        for l, a in enumerate(alphas):
            if a is None:
                continue
            net2.fcs[l].weight.data *= a.unsqueeze(1)
            net2.fcs[l].bias.data *= a
            f2[l] *= a
    return net2, f2


def overlap_of(u, v) -> float:
    return sum(float((p / p.norm() * (q / q.norm())).sum())
               for p, q in zip(u, v)) / len(u)


def check_invariance(net, x, fa, args, device, n_hidden) -> None:
    """Gauge each layer alone, then both, then gauge the field away entirely."""
    print("=== 1. invariance, and why the hidden layers do not break it ===")
    torch.manual_seed(1)
    state = torch.get_rng_state()
    y0, z0 = run(net, x, fa, args, state)
    a1 = 0.5 + 1.5 * torch.rand(args.hidden_dim)
    a2 = 0.5 + 1.5 * torch.rand(args.hidden_dim)
    print("  layer 1 feeds every unit of layer 2, so gauging layer 1 ALONE is what")
    print("  should break things.  The z columns are the answer: nothing moves.")
    for name, al in (("layer 1 only", [a1, None]),
                     ("layer 2 only", [None, a2]),
                     ("both layers ", [a1, a2])):
        net2, f2 = apply_gauge(net, fa, al, args, device)
        y1, z1 = run(net2, x, f2, args, state)
        dz = "  ".join(f"max|dz^({l + 1})| = {float((z1[l] - z0[l]).abs().max()):.1e}"
                       for l in range(n_hidden))
        print(f"    {name}:  max|dy| = {float((y1 - y0).abs().max()):.2e}    {dz}")

    print("\n  the strong form: alpha_k = 1/sigma_k gauges the field flat.")
    net2, f2 = apply_gauge(net, fa, [1.0 / v for v in fa], args, device)
    y1, _ = run(net2, x, f2, args, state)
    before, after = torch.cat(fa), torch.cat(f2)
    print(f"    field before: mean {float(before.mean()):.4f}  "
          f"range [{float(before.min()):.3f}, {float(before.max()):.3f}]")
    print(f"    field after : mean {float(after.mean()):.4f}  "
          f"range [{float(after.min()):.3f}, {float(after.max()):.3f}]")
    print(f"    max|dy| = {float((y1 - y0).abs().max()):.2e}"
          f"   -> a flat field computes the SAME function.")

    couple = args.couple_h
    args.couple_h = 0.0
    y0f, z0f = run(net, x, fa, args, state)
    net2, f2 = apply_gauge(net, fa, [a1, a2], args, device)
    y1f, z1f = run(net2, x, f2, args, state)
    print(f"\n  with a FIXED h (the NCE paper's setting) the symmetry is absent:")
    print(f"    both layers :  max|dy| = {float((y1f - y0f).abs().max()):.2e}    "
          f"max|dz^(1)| = {float((z1f[0] - z0f[0]).abs().max()):.1e}")
    print("    h is a second scale, so sigma is a real degree of freedom there.")
    args.couple_h = couple


def check_identity(net, x, tA, fa, args, n_hidden) -> None:
    """dL/dlog sigma_k + (w_k . dL/dw_k + b_k dL/db_k) = 0 on the PoC's estimators."""
    print("\n=== 2. the gauge identity on the PoC's own estimators ===")
    print("    dL/dlog sigma_k  +  (w_k . dL/dw_k + b_k dL/db_k)  =  0 ?")
    for couple in (args.couple_h, 0.0):
        saved = args.couple_h
        args.couple_h = couple
        torch.manual_seed(2)
        cap = Capture(net)
        W_ema: dict = {}
        with torch.no_grad():
            st = forward_stats(net, cap, x, tA, fa, args)
            update_mirrors(W_ema, st, n_hidden, args.jac_ema)
            a = jac_credit(st, W_ema, tA, n_hidden)
            gs = sigma_grad(st, a, fa, n_hidden, args.sigma_min)
            hg = hidden_grads(st, a, x, n_hidden)
        cap.remove()
        tag = f"h={couple}*sigma" if couple > 0 else "h fixed   "
        for l in range(n_hidden):
            gW, gb = hg[l]
            radial = ((net.fcs[l].weight.data * gW).sum(dim=1)
                      + net.fcs[l].bias.data * gb)
            g_log = fa[l] * gs[l]
            resid = (g_log + radial).abs().max()
            print(f"    {tag}  layer {l + 1}:  max|residual| = {float(resid):.2e}"
                  f"   relative = {float(resid / (g_log.abs().max() + 1e-12)):.2e}")
        args.couple_h = saved
    print("    Identical either way: the ESTIMATOR always obeys it, because both")
    print("    gradients are the same credit times the same slope.  Only the MODEL")
    print("    needs the coupling.  With a fixed h the estimator is imposing a")
    print("    symmetry the network does not have -- a bias no sampling can fix.")


def check_radial(net, x, tA, fa, args, n_hidden) -> None:
    """How much of the weight gradient does the radial projection actually see?"""
    print("\n=== 3. how much of the weight gradient is radial? ===")
    torch.manual_seed(3)
    cap = Capture(net)
    W_ema: dict = {}
    with torch.no_grad():
        st = forward_stats(net, cap, x, tA, fa, args)
        update_mirrors(W_ema, st, n_hidden, args.jac_ema)
        a = jac_credit(st, W_ema, tA, n_hidden)
        hg = hidden_grads(st, a, x, n_hidden)
    cap.remove()
    for l in range(n_hidden):
        gW, gb = hg[l]
        A = torch.cat([net.fcs[l].weight.data, net.fcs[l].bias.data[:, None]], dim=1)
        G = torch.cat([gW, gb[:, None]], dim=1)
        r = ((A * G).sum(dim=1)).abs() / (A.norm(dim=1) * G.norm(dim=1) + 1e-12)
        print(f"    layer {l + 1}:  |radial| / (|a||g|)   mean = {float(r.mean()):.3f}"
              f"   median = {float(r.median()):.3f}   max = {float(r.max()):.3f}")
    print("    1.0 would mean the sigma gradient reads the whole weight gradient.")
    print("    At 0.2-0.4 it reads a minor component, which is the likely reason a")
    print("    0.998 weight-gradient cosine still leaves c_inf ~ 0.95 (section 11).")


def check_metrics(net, x, args, device, n_hidden) -> None:
    """Are overlap and mean(P) properties of the function, or of the gauge?"""
    print("\n=== 4. overlap and mean_P are not gauge invariant ===")
    torch.manual_seed(0)
    fa, fb = init_fields(args, n_hidden, device)
    torch.manual_seed(7)
    state = torch.get_rng_state()
    yA0, _ = run(net, x, fa, args, state)
    yB0, _ = run(net, x, fb, args, state)
    print(f"    before gauge:  overlap = {overlap_of(fa, fb):.4f}   "
          f"mean_P = {float(torch.cat(fa + fb).mean()):.4f}")

    # both contexts share a_k, so one alpha_k must move both fields together
    al = [1.0 / v for v in fa]
    net2, fa2 = apply_gauge(net, fa, al, args, device)
    fb2 = [v * a for v, a in zip(fb, al)]
    yA1, _ = run(net2, x, fa2, args, state)
    yB1, _ = run(net2, x, fb2, args, state)
    print(f"    after gauge :  overlap = {overlap_of(fa2, fb2):.4f}   "
          f"mean_P = {float(torch.cat(fa2 + fb2).mean()):.4f}")
    print(f"    functions unchanged:  max|dy_A| = "
          f"{float((yA1 - yA0).abs().max()):.2e}   max|dy_B| = "
          f"{float((yB1 - yB0).abs().max()):.2e}")
    d = max(float((p / q - r / s).abs().max())
            for p, q, r, s in zip(fb, fa, fb2, fa2))
    print(f"    the RATIO sigma_B/sigma_A is invariant:  max|change| = {d:.2e}")
    print("    So a differentiation claim must be scored on the ratio (or on the")
    print("    log field with the per-unit context mean removed), never on overlap.")


def main() -> None:
    sys.argv = [sys.argv[0], "--couple-h", "0.2"]
    args = parse_args()
    device = torch.device("cpu")
    torch.manual_seed(0)
    _, x, tA, _ = make_tasks(device)
    net = build_model(args, device)
    n_hidden = len(net.structure) - 2
    fa, _ = init_fields(args, n_hidden, device)
    print(f"structure = {net.structure}   hidden units = {n_hidden * args.hidden_dim}"
          f"   field dim = {n_hidden * args.hidden_dim}   (they are equal: the field")
    print("is exactly as large as the gauge group, for one context)\n")

    check_invariance(net, x, fa, args, device, n_hidden)
    check_identity(net, x, tA, fa, args, n_hidden)
    check_radial(net, x, tA, fa, args, n_hidden)
    check_metrics(net, x, args, device, n_hidden)


if __name__ == "__main__":
    main()
