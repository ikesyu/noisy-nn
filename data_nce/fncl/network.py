"""fncl.network — NNN モデルの構築と forward パスの計測.

- 交差活性の期待応答のノイズ由来局所微分 phi'(d) (gaussian / uniform の解析形) と
  分布フリーの KDE slope 推定 (kde_slope),
- build_model(): nnn ライブラリの Sample/UniformSample モデル (1 次元入力の
  bump タイリング初期化 + per-unit noise field つき),
- Capture: forward フックで per-sample の d, z, readout サンプルを記録する.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# Use the local nnn library models.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from nnn import model  # noqa: E402

from .constants import SQRT2, UNIFORM_CENTER  # noqa: E402


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


def kde_slope(crossing_layer, d_clean: torch.Tensor) -> torch.Tensor:
    """Distribution-FREE local slope dz_bar/dd from the Crossing's OWN internal density
    estimate -- no analytic phi', no knowledge of the noise distribution.

    The library's CrossingSample.backward computes, per unit,

        coeff = mean_t(xor2 - xor1) / (2h)

    where xor1/xor2 count level crossings of the SAME noisy samples at the thresholds +h
    and -h.  Evaluating the two thresholds on one shared sample set is exactly an
    ANTITHETIC / COMMON-RANDOM-NUMBER finite difference: shifting the threshold by +/-h is
    equivalent to shifting the pre-activation d by -/+h, and the common samples make the
    noise cancel in the difference -> a low-variance, distribution-free estimate of dz/dd.

    We obtain it by a LOCAL grad-enabled re-run of just this crossing layer (fresh noise;
    a single unit's activation derivative -- NO transposed-weight backward, no readout).
    Because the noise addition d -> d + eta has unit gradient w.r.t. d, autograd routes
    CrossingSample.backward's coeff straight onto d_clean.grad.

    Returns dz/dd of shape [N, T, H] (constant across T; the library repeats coeff over T).
    """
    with torch.enable_grad():
        d_req = d_clean.detach().clone().requires_grad_(True)
        z = crossing_layer(d_req)                 # noise + CrossingSample, this layer only
        z.sum().backward()                        # grad_output = 1 -> d_req.grad = coeff
    return d_req.grad.detach()


# ============================================================
# Build a library model and give layer-1 a spread "bump tiling" so the readout has a
# good basis (the default init does not tile the 1-D input).
# ============================================================
def build_model(noise: str, hidden: int, sigma: float, radius: float, h: float,
                t: int, device: torch.device, field: torch.Tensor = None):
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

    # --- per-unit noise field (recruitment gate) read by cov_deriv_field_gate ---
    # `net.noise_field[l]` is a [H] vector of each hidden unit's noise-field strength.
    # When a non-trivial field is supplied it becomes a REAL noise field: the per-unit
    # Gaussian std (or uniform radius) is scaled by it, so a zero-field unit gets NO
    # noise -> it is deterministic (dead) and receives no update under the field gate.
    # Default (field is None) -> all-ones, so the network and every other method are
    # byte-identical to before (the scalar std/radius is left untouched).
    n_hidden = len(structure) - 2
    if field is None:
        s = torch.ones(H, device=device)
    else:
        s = field.detach().to(device).float()
        if noise == "gaussian":
            net.std = sigma * s                                  # [H], broadcasts over [N,T,H]
        else:
            for cl in net.uniform_crossing:
                cl.radius = radius * s
    net.noise_field = [s for _ in range(n_hidden)]
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
