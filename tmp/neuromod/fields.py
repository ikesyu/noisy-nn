"""Neuromodulator-like noise fields on a virtual unit sheet.

A field assigns each hidden unit a noise intensity sigma_k.  Units are laid out
on a square sheet purely so that a field can be a localized, graded bump with a
partially overlapping support; the network itself has no topology, so the sheet
is a device for generating controlled overlap, not a claim about anatomy.

Two things here differ from the original demo, both because the multiplexing
experiment (L2) depends on them:

1. **Centres sit on a ring, equidistant.**  The original placed three bumps at
   the corners of a right triangle, so food-threat were sqrt(2) times farther
   apart than the other pairs and shared only 2 units (Jaccard 0.043) against 8
   (0.200) for the others.  The pair the lesion experiment most wants to test was
   therefore almost perfectly PARTITIONED, which is the hypothesis the experiment
   is supposed to refute.  With `ring_centers` every pair overlaps equally and the
   ring radius becomes a controlled knob: sweep it to sweep overlap.

2. **Participation is measured by the crossing rate nu_k = E[z_k], not by
   thresholding sigma_k.**  The absolute value of sigma is gauge-dependent, so a
   set defined as {k : sigma_k > c} moves under a gauge transformation that leaves
   the network's output bit-for-bit identical, which makes any Jaccard computed
   from it meaningless.  nu_k is homogeneous of degree zero and therefore gauge
   invariant.  See `docs/idea_core.md` sections 3.3 / 4.3 / 4.8 and
   `docs/idea_neuromod.md` sections 5.3 / 7.3.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch

from nnn import stats


def sheet_side(hidden_dim: int) -> int:
    """Side length of the square unit sheet holding `hidden_dim` units."""
    side = int(round(math.sqrt(hidden_dim)))
    if side * side != hidden_dim:
        raise ValueError(
            f"hidden_dim must be a perfect square to lay out a square sheet; "
            f"got {hidden_dim}.")
    return side


def unit_sheet(hidden_dim: int) -> np.ndarray:
    """Coordinates of the unit sheet, flattened to [hidden_dim, 2] in [0, 1]^2."""
    side = sheet_side(hidden_dim)
    axis = np.linspace(0.0, 1.0, side, dtype=np.float32)
    gx, gy = np.meshgrid(axis, axis)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


def ring_centers(categories, radius: float = 0.28, phase_deg: float = 90.0,
                 centre=(0.5, 0.5)) -> dict[str, tuple[float, float]]:
    """Bump centres equally spaced on a ring, so every pair overlaps equally.

    For n categories the pairwise centre distance is 2 * radius * sin(pi / n),
    i.e. radius * sqrt(3) for the standard three.  Larger radius pushes the
    supports apart (towards partition), smaller pulls them together (towards full
    sharing), which is exactly the axis the L2 experiment wants to sweep.
    """
    n = len(categories)
    out = {}
    for i, key in enumerate(categories):
        angle = math.radians(phase_deg) + 2.0 * math.pi * i / n
        out[key] = (centre[0] + radius * math.cos(angle),
                    centre[1] + radius * math.sin(angle))
    return out


def make_noise_field(center, hidden_dim: int, base_std: float, sigma: float,
                     theta_cut: float) -> torch.Tensor:
    """One localized noise-std vector of length `hidden_dim`.

        intensity[i] = exp(-||sheet_i - center||^2 / (2 sigma^2))

    Values below `theta_cut` are truncated to zero and the result is scaled by
    `base_std`.  The truncation carves out the spatial SUPPORT of the field; it
    is not a unit-retirement operation.  A unit with sigma_k = 0 falls silent
    exactly only in the first hidden layer and only at mean-field level; deeper
    layers keep crossing on upstream sample fluctuation (the sigma=0 leak).  To
    truly silence a unit use the kill triple in `kill_units`.
    """
    grid = unit_sheet(hidden_dim)
    center = np.asarray(center, dtype=np.float32)
    d2 = np.sum((grid - center) ** 2, axis=1)
    intensity = np.exp(-d2 / (2.0 * sigma ** 2))
    intensity[intensity < theta_cut] = 0.0
    return torch.tensor(base_std * intensity, dtype=torch.float32)


def build_fields(categories, hidden_dim: int, base_std: float, sigma: float,
                 theta_cut: float, radius: float = 0.28,
                 centers: dict = None) -> dict[str, torch.Tensor]:
    """One noise field per category, centred on a ring unless `centers` is given."""
    if centers is None:
        centers = ring_centers(categories, radius=radius)
    return {key: make_noise_field(centers[key], hidden_dim, base_std, sigma, theta_cut)
            for key in categories}


def blend_fields(fields: dict[str, torch.Tensor], weights: np.ndarray,
                 categories) -> torch.Tensor:
    """Convex combination of the per-category fields, in `categories` order."""
    out = torch.zeros_like(next(iter(fields.values())))
    for w, key in zip(weights, categories):
        out = out + float(w) * fields[key]
    return out


# ============================================================
# Participation: the gauge-invariant crossing rate
# ============================================================

def crossing_rates(net, obs: torch.Tensor, field: torch.Tensor,
                   n_hidden: int) -> list[np.ndarray]:
    """Per-layer crossing rate nu_k = E[z_k], averaged over the observations.

    Works for every NNN variant: the analytic model's hidden output IS the
    expected response, and the sample model's is averaged over its T samples.
    Returns one [hidden_dim] array per hidden layer.
    """
    dev = next(net.parameters()).device
    obs = obs.to(dev)
    field = field.to(dev)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    layers = stats.crossing_layers(net)

    def make_hook(idx):
        def hook(module, inputs, output):
            captured[idx] = output.detach()
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))
    try:
        with torch.no_grad():
            net(obs, stds=[field] * n_hidden)
    finally:
        for handle in handles:
            handle.remove()

    return [captured[i].reshape(-1, captured[i].shape[-1]).mean(0).cpu().numpy()
            for i in range(len(layers))]


def recruited_set(nu: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Support of the crossing rate, {k : nu_k > eps}, as a boolean mask.

    `eps` is a numerical floor, not a participation threshold: nu is exactly zero
    for a unit that never crosses, so any small positive eps gives the same set.
    """
    return nu > eps


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Jaccard overlap of two recruited sets; 0 when both are empty."""
    union = int((mask_a | mask_b).sum())
    return float((mask_a & mask_b).sum()) / union if union else 0.0


def overlap_report(net, obs: torch.Tensor, fields: dict[str, torch.Tensor],
                   categories, n_hidden: int, layer: int = 0,
                   eps: float = 1e-4) -> dict:
    """Recruited-set sizes and pairwise Jaccard overlaps, measured through nu.

    `layer` selects which hidden layer to report; layer 0 is the one where a
    zero-sigma unit is genuinely silent, so it is the honest place to define the
    recruited set.
    """
    nus = {key: crossing_rates(net, obs, fields[key], n_hidden)[layer]
           for key in categories}
    masks = {key: recruited_set(nus[key], eps) for key in categories}
    pairs = {f"{a}|{b}": jaccard(masks[a], masks[b])
             for a, b in itertools.combinations(categories, 2)}
    shared = {f"{a}|{b}": int((masks[a] & masks[b]).sum())
              for a, b in itertools.combinations(categories, 2)}
    union = np.zeros_like(next(iter(masks.values())))
    for mask in masks.values():
        union |= mask
    return {
        "nu": nus,
        "masks": masks,
        "sizes": {key: int(mask.sum()) for key, mask in masks.items()},
        "jaccard": pairs,
        "shared": shared,
        "never_recruited": int((~union).sum()),
        "hidden_dim": int(union.size),
        "layer": layer,
    }


def print_overlap_report(report: dict) -> None:
    """Human-readable dump of `overlap_report`."""
    H = report["hidden_dim"]
    print(f"\nParticipation via crossing rate nu (hidden layer {report['layer'] + 1}):")
    for key, size in report["sizes"].items():
        nu = report["nu"][key]
        print(f"  field '{key:7s}': {size:3d}/{H} units participate, "
              f"peak nu={nu.max():.3f}")
    print("  pairwise overlap (Jaccard on the nu support):")
    for pair, value in report["jaccard"].items():
        print(f"    {pair:20s} {value:.3f}  (shared {report['shared'][pair]})")
    print(f"  units never recruited by any field: {report['never_recruited']}/{H}")


def kill_units(net, mask: np.ndarray, fields: dict[str, torch.Tensor],
               layer: int = 0, h_dead: float = 1e6) -> None:
    """Lesion units by the kill triple, in place.

    Setting sigma_k = 0 alone does NOT silence a unit beyond the first hidden
    layer, because upstream sample fluctuation keeps carrying it across the +-h
    band.  A lesion experiment built on sigma alone therefore reports "damaged but
    still working" and proves nothing.  The triple is

        sigma_k <- 0,  h_k <- H_DEAD,  W[l+1][:, k] <- 0

    (`docs/idea_core.md` section 3.5).  `fields` is edited too, so every later
    forward pass keeps the lesion.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    for field in fields.values():
        field[idx] = 0.0
    crossings = stats.crossing_layers(net)
    crossing = crossings[layer]
    if hasattr(crossing, "h"):
        dev = next(net.parameters()).device
        h = crossing.h
        if not torch.is_tensor(h):
            h = torch.full((mask.size,), float(h), device=dev)
        h = h.clone().to(dev)
        h[idx] = h_dead
        crossing.h = h
    with torch.no_grad():
        net.fcs[layer + 1].weight[:, idx] = 0.0
