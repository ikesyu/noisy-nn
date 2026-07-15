"""Forward-pass statistics of a sample-level NNN: activations and local slopes.

The expected crossing response and its local derivative follow from the same
injected noise, so one stochastic forward pass already carries what a learning
rule needs. `Capture` records the per-sample internals of a pass; `kde_slope`
and `phi_prime` give the local slope dE[z]/dd distribution-free and in closed
form respectively.
"""
import torch


def crossing_layers(net):
    """The hidden crossing modules of a Sample / UniformSample model."""
    return getattr(net, "gaussian_crossing", None) or net.uniform_crossing


class Capture:
    """Records the per-sample internals of every forward pass, via hooks.

    For each hidden crossing layer, its input `d[l]` and output `z[l]`, both of
    shape [N, T, H]. For the readout, `y_samples` of shape [N, T, 1], taken
    before the ensemble mean.
    """

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
        self.y_samples = out.detach()

    def remove(self):
        for hnd in self.handles:
            hnd.remove()


def kde_slope(crossing_layer, d_clean: torch.Tensor) -> torch.Tensor:
    """Distribution-free local slope dE[z]/dd, from the crossing's own samples.

    `CrossingSample.backward` returns mean_t(xor2 - xor1) / (2h), the crossing
    counts of one shared sample set at the thresholds +h and -h. Shifting the
    threshold by +/-h is equivalent to shifting `d` by -/+h, so this is an
    antithetic finite difference whose shared noise cancels in the difference.
    Re-running only this layer under autograd routes that coefficient onto
    `d_clean.grad`; no transposed weights and no readout take part.

    Returns dz/dd of shape [N, T, H] (constant along T).
    """
    with torch.enable_grad():
        d_req = d_clean.detach().clone().requires_grad_(True)
        z = crossing_layer(d_req)
        z.sum().backward()
    return d_req.grad.detach()


def phi_prime(d: torch.Tensor, pdf, cdf) -> torch.Tensor:
    """Analytic local slope of the expected crossing response.

    From phi_bar(d) = 2 F(d) (1 - F(d)) follows

        phi_bar'(d) = 2 (1 - 2 F(d)) p(d),

    which holds for any noise distribution. `pdf` and `cdf` are the closures
    returned by the `nnn.noise` factories, so the same call covers Gaussian,
    uniform, and stable noise.
    """
    F = cdf(d)
    return 2.0 * (1.0 - 2.0 * F) * pdf(d)
