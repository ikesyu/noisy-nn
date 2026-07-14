"""fncl.perturb — 外部摂動ゲート (cov_deriv_gate / cov_deriv_gate_crn, Appendix B).

Perturber は forward プレフックで隠れ層の pre-activation に小摂動 xi を注入する。
rng_snapshot/rng_restore は common-random-number (antithetic) ペアのための
RNG 状態の保存・復元、gate_masks は摂動・更新対象のブロックゲート。
"""
from __future__ import annotations

import math

import torch

from .network import crossing_layers


# ============================================================
# Perturbation gate (cov_deriv_gate)
# ------------------------------------------------------------
# The perturbation gate is intended to reduce covariance-credit bias by correlating the
# loss with an explicitly injected local perturbation, rather than with all spontaneous
# unit fluctuations.  Instead of regressing L onto z's own (large, binary) forward noise,
# we inject a small EXTRA perturbation xi into a rotating BLOCK of hidden pre-activations
# and regress L onto xi.  Only that block is perturbed, and only that block is updated.
#
# Crucially the injection is done via forward PRE-hooks on the crossing layers, so the
# perturbation enters the ACTUAL loss-bearing forward pass:
#
#       d_sample = d + base_nnn_noise + alpha * G_k * xi_std          (xi_std ~ N(0,1))
#
# The model's own noise field / stochastic crossing is left completely intact -- we only
# add xi on top, on the gated units.  The injected perturbation p = alpha * G_k * xi_std
# is recorded per pass so the hidden credit can be estimated as Cov(L, p)/Var(p).
#
# The Perturber also supports an ANTITHETIC / COMMON-RANDOM-NUMBER mode ("crn", used by
# cov_deriv_gate_crn): a FIXED perturbation is added as +xi or -xi on demand, so the caller
# can run a paired (+xi, -xi) forward under the SAME crossing noise (via an RNG-state reset)
# and cancel the intrinsic-noise nuisance in the loss difference (see train_cov / §8.3).
# ============================================================
class Perturber:
    """Injects a small block-gated perturbation into hidden pre-activations via forward
    pre-hooks so credit can be taken from Cov(L, perturbation).

    Two modes:
      mode="gate" (default): each forward draws a fresh p = alpha * mask * N(0,1) and
        records it in `p[l]` (shape [N, T, H_l], zero off the gated block).
      mode="crn"           : adds `sign * fixed_p[l]` (a caller-supplied fixed perturbation),
        used to run antithetic (+/-) pairs under common random numbers.

    Set `masks` (a list of [H_l] float gate vectors, 1.0 on the currently gated units)
    before each forward pass."""

    def __init__(self, net, alpha: float, mode: str = "gate"):
        self.crossings = crossing_layers(net)
        self.n_hidden = len(self.crossings)
        self.alpha = float(alpha)
        self.mode = mode
        self.masks = [None] * self.n_hidden     # [H_l] float gate, set each epoch
        self.p = [None] * self.n_hidden         # last injected perturbation [N, T, H_l]
        self.fixed_p = [None] * self.n_hidden   # crn: caller-supplied fixed perturbation
        self.sign = 1.0                         # crn: +1 or -1 for the antithetic pair
        self.handles = []
        for l, layer in enumerate(self.crossings):
            self.handles.append(layer.register_forward_pre_hook(self._make(l)))

    def _make(self, l):
        def pre_hook(module, inp):
            d = inp[0]                                          # [N, T, H] pre-activation
            if self.mode == "crn":
                # add the SAME fixed perturbation with the current sign (antithetic pair)
                p = self.sign * self.fixed_p[l]
                return (d + p,) + tuple(inp[1:])
            mask = self.masks[l]
            if mask is None or self.alpha == 0.0:
                self.p[l] = torch.zeros_like(d)
                return None                                    # no injection this pass
            # nonzero only on the gated block; base forward noise is added by the layer
            p = (self.alpha * mask) * torch.randn_like(d)      # alpha * G_k * xi_std
            self.p[l] = p.detach()
            return (d + p,) + tuple(inp[1:])                   # perturb the ACTUAL forward
        return pre_hook

    def remove(self):
        for hnd in self.handles:
            hnd.remove()


def rng_snapshot(device: torch.device):
    """Snapshot the RNG state (for common-random-numbers antithetic pairs)."""
    if device.type == "cuda":
        return ("cuda", device, torch.cuda.get_rng_state(device))
    return ("cpu", None, torch.get_rng_state())


def rng_restore(snap) -> None:
    """Restore an rng_snapshot so the next forward re-draws the identical noise."""
    kind, device, state = snap
    if kind == "cuda":
        torch.cuda.set_rng_state(state, device)
    else:
        torch.set_rng_state(state)


def gate_masks(hidden_sizes, epoch: int, block_size: int, mode: str,
               device: torch.device):
    """Block gate G_k for each hidden layer: a rotating (cyclic) or random contiguous
    block of `block_size` units is set to 1.0, the rest 0.0.  Returns a list of [H_l]."""
    masks = []
    for H in hidden_sizes:
        n_blocks = max(1, math.ceil(H / block_size))
        if mode == "cyclic":
            b = epoch % n_blocks
        else:  # random
            b = int(torch.randint(0, n_blocks, (1,)).item())
        m = torch.zeros(H, device=device)
        m[b * block_size:(b + 1) * block_size] = 1.0
        masks.append(m)
    return masks
