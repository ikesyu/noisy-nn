"""tmp/rl.metrics -- comparison metrics for the credit estimators (idea_rl.md §20.6)."""
from __future__ import annotations

import torch


def flatten(grads, keys):
    """Concatenate a {key: tensor} grad dict into one vector in `keys` order."""
    return torch.cat([grads[k].reshape(-1) for k in keys])


def cosine(grads_a, grads_b, keys):
    a = flatten(grads_a, keys)
    b = flatten(grads_b, keys)
    denom = a.norm() * b.norm()
    if denom < 1e-12:
        return float("nan")
    return float((a @ b) / denom)


def hidden_keys(keys, n_fcs):
    """Keys of the hidden layers only (exclude the top readout layer index)."""
    readout = n_fcs - 1
    return [k for k in keys if k[0] != readout]


def normalized_variance(vectors):
    """E||g - E g||^2 / ||E g||^2 for a list of estimate vectors (M2-(b), §20.6)."""
    G = torch.stack(vectors)                  # [S, P]
    mean = G.mean(dim=0)
    num = ((G - mean) ** 2).sum(dim=1).mean()
    den = (mean ** 2).sum()
    return float(num / (den + 1e-12))
