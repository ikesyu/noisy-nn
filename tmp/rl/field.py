"""tmp/rl.field -- per-unit noise-field prototypes (§7).

A field is a list of one [H] std vector per hidden layer.  These prototypes let us test
whether SPATIALLY ALLOCATING noise (some units quiet, some loud) can escape the uniform-
sigma computation-vs-control tension found by the SR sweep (idea_rl.md §20.16): quiet
units stay in the good-mirror / decisive regime, loud units carry exploration.
"""
from __future__ import annotations

import torch


def uniform(H, sigma, n_layers=2):
    return [torch.full((H,), float(sigma)) for _ in range(n_layers)]


def split(H, lo, hi, frac_lo=0.5, n_layers=2):
    """First frac_lo of units at `lo`, the rest at `hi` (a two-level spatial field)."""
    n_lo = int(round(H * frac_lo))
    v = torch.empty(H)
    v[:n_lo] = float(lo)
    v[n_lo:] = float(hi)
    return [v.clone() for _ in range(n_layers)]


def graded(H, lo, hi, n_layers=2):
    """Smooth ramp of std from `lo` to `hi` across units."""
    v = torch.linspace(float(lo), float(hi), H)
    return [v.clone() for _ in range(n_layers)]


def recruit(H, sigma, side, n_layers=2, quiet=0.0):
    """Recruitment field: one half of the units at `sigma`, the other half at `quiet`
    (0.0 = fully detached).  `side` 0 recruits the first half, 1 the second half.  Two
    such fields address DISJOINT subnetworks on shared weights -- the noise-field option
    mechanism (§7.1).  `quiet`>0 leaves the off-half faintly active."""
    n = H // 2
    v = torch.full((H,), float(quiet))
    if side == 0:
        v[:n] = float(sigma)
    else:
        v[n:] = float(sigma)
    return [v.clone() for _ in range(n_layers)]


def bump(H, center, sigma, tau=0.15, n_layers=1, cutoff=0.1):
    """Continuous noise field (§7.3): a Gaussian recruitment bump centred at `center` in
    [0,1] over the H units.  Sliding `center` slides which subnetwork is recruited, so the
    field becomes a CONTINUOUS option coordinate (§19).  Units below `cutoff`*sigma are set
    to 0 (dead)."""
    coord = torch.linspace(0.0, 1.0, H)
    v = float(sigma) * torch.exp(-0.5 * ((coord - float(center)) / tau) ** 2)
    v = torch.where(v >= cutoff * sigma, v, torch.zeros_like(v))
    return [v.clone() for _ in range(n_layers)]


def overlapping_pair(H, sigma, recruit_frac=0.7, n_layers=2):
    """Two recruitment fields whose active units OVERLAP (recruit_frac>0.5).  P0 recruits
    the first `block` units, P1 the last `block`; the middle is SHARED.  Returns the two
    fields plus the last-hidden index sets (shared / p0_only / p1_only) for the lesion
    test (front_comp L2: do shared units carry BOTH behaviors, i.e. multiplexing, or did
    the net partition into disjoint groups?)."""
    block = min(max(int(round(H * recruit_frac)), 1), H)
    v0 = torch.zeros(H)
    v0[:block] = float(sigma)
    v1 = torch.zeros(H)
    v1[H - block:] = float(sigma)
    a0, a1 = v0 > 0, v1 > 0
    idx = {
        "shared": torch.where(a0 & a1)[0].tolist(),
        "p0_only": torch.where(a0 & ~a1)[0].tolist(),
        "p1_only": torch.where(~a0 & a1)[0].tolist(),
    }
    fields = [[v0.clone() for _ in range(n_layers)], [v1.clone() for _ in range(n_layers)]]
    return fields, idx


def prototypes(H, lo=0.3, hi=1.3, mid=0.6):
    """The comparison set for Sub-A: three uniforms bracketing the SR range plus two
    spatial fields with the same total noise budget region."""
    return {
        "uniform_lo": uniform(H, lo),
        "uniform_mid": uniform(H, mid),
        "uniform_hi": uniform(H, hi),
        "split": split(H, lo, hi),
        "graded": graded(H, lo, hi),
    }
