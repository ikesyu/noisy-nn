# Forward-Noise-Based Covariance Learning for Noise-Modulated Neural Networks

## 1. Overview

This document describes a proof-of-concept learning algorithm for
Noise-modulated Neural Networks (NNNs) that approximates the *structure* of
backpropagation **without** performing ordinary backward error propagation through
transposed weight matrices. The method reuses the stochastic forward samples that an
NNN produces anyway to (i) estimate hidden-layer credit from the covariance between
loss and node activity, and (ii) convert that credit into weight updates using the
*noise-induced* local derivative of the crossing activation.

The accompanying example is
[`examples/forward_noise_covariance_learning.py`](../examples/forward_noise_covariance_learning.py).

## 2. Motivation

Backpropagation is powerful but hardware-unfriendly. Its hidden-layer credit
assignment,

```
delta_l = (W_{l+1}^T delta_{l+1}) * phi'(d_l),
```

requires a **backward pass** that multiplies error signals by **transposed weight
matrices** `W_{l+1}^T`. On dedicated hardware (e.g. FPGAs / analog accelerators) this
"weight transport" is expensive: it needs a second data path, transposed memory
access, and tight synchronization between forward and backward phases.

NNNs, however, are *intrinsically stochastic*: the crossing activation flips
depending on injected noise, so a single input naturally yields many forward samples.
Those samples already contain information about how each unit's fluctuation affects
the loss. The idea here is to **reuse the forward stochastic samples as implicit
perturbations** for credit assignment, eliminating the transposed-weight backward
pass.

## 3. NNN crossing activation

**Sample-level (binary spike) crossing activation.** For a pre-activation `d`,

```
eta1, eta2 ~ N(0, sigma^2)
z = XOR(d >= eta1, d >= eta2)          # z in {0, 1}
  = ((d >= eta1) != (d >= eta2)).float()
```

**Analytical expected activation.** Taking the expectation over the two independent
Gaussian thresholds gives a smooth bump-shaped response,

```
F(d; sigma)      = 0.5 * (1 + erf(d / (sigma * sqrt(2))))          # Gaussian CDF
phi_bar(d; sigma) = 2 * F(d) * (1 - F(d))                          # in [0, 0.5]
```

**Noise-induced local derivative.** Differentiating `phi_bar` analytically,

```
p(d; sigma)        = Gaussian PDF with std sigma
phi_bar'(d; sigma) = 2 * (1 - 2 F(d)) * p(d)
```

`phi_bar'` is positive for `d < 0`, negative for `d > 0`, and vanishes far from the
threshold — it is the statistical slope `dz/dd` induced by the noise, and it is the
factor the proposed rule uses in place of a hand-designed surrogate gradient.

**Bounded-uniform variant (`--noise uniform`, `SimpleNNNUniformSample`).** With
uniform noise on `[center-r, center+r]` the expected crossing response is the exact
parabola `phi_bar(d) = 0.5 [1 - ((d-c)/r)^2]_+`, whose local derivative is
`phi_bar'(d) = -(d-c)/r^2` for `|d-c| < r` (and `0` outside). The example uses this
derivative for `cov_deriv` when `--noise uniform`.

## 4. Proposed algorithm

For each input `x` (batched over `N` points) and target `t`, run `M` stochastic
forward passes and record, per sample `m` and point `n`, the pre-activations
`d_l^{(m,n)}`, activations `z_l^{(m,n)}`, output `y^{(m,n)}`, and per-sample loss
`L^{(m,n)} = (y^{(m,n)} - t_n)^2`.

**Step 1 — activity-side credit from covariance.** For each hidden unit `i` in layer
`l`,

```
g_z[l, i] = Cov(L, z_{l,i}) / (Var(z_{l,i}) + eps)   ~=   dL / dz_{l,i}
```

where the covariance and variance are estimated over the pooled `M x N` samples.
This replaces the backprop term `W_{l+1}^T delta_{l+1}` — **no transposed weights are
used**.

**Step 2 — local sensitivity from the noise-induced derivative.**

```
phi'[l, i] ~= phi_bar'(d_{l,i}) ~= dz_{l,i} / dd_{l,i}
```

**Step 3 — pseudo-error and local weight update.**

```
delta_hat[l, i]     = g_z[l, i] * phi_bar'(d_{l,i})
Delta W_l[i, j]     = -eta * mean_{m,n}( delta_hat[l,i] * z_{l-1,j} )
Delta b_l[i]        = -eta * mean_{m,n}( delta_hat[l,i] )
```

Because `g_z * phi' ~= (dL/dz) * (dz/dd) = dL/dd`, the update has the same *form* as a
backprop weight update `delta * z_prev^T`, but the "delta" is assembled entirely from
forward statistics plus a local activation derivative.

**Output layer.** For stability the output (readout) layer uses a **direct local
readout gradient**, which needs no hidden-layer error propagation:

```
dL/dy         = 2 (y - t)
Delta W_out   = -eta * mean_{m,n}( dL/dy * z_last^T )
Delta b_out   = -eta * mean_{m,n}( dL/dy )
```

This is exact for the (linear) readout because `y` is linear in `W_out`; it does not
propagate error into the hidden layers.

## 5. Relation to backpropagation

| Ingredient | Ordinary backprop | Proposed forward-noise method |
|---|---|---|
| Hidden activity credit | `W_{l+1}^T delta_{l+1}` (backward, transposed weights) | `Cov(L, z) / Var(z)` (forward statistics) |
| Local sensitivity | `phi'(d)` (analytic surrogate) | noise-induced crossing derivative `phi_bar'(d)` |
| Weight update | `delta * z_prev^T` | `delta_hat * z_prev^T` |
| Data path | forward **and** backward | mostly forward + streaming statistics |

The proposed rule approximates the *structure* of backprop while avoiding the
`W^T delta` backward term. It is an **approximation**, not an exact reproduction of
the backprop gradient (see Limitations).

## 6. Relation to node perturbation

The method can be read as **implicit node perturbation**. Classical node
perturbation injects an extra artificial perturbation into each unit *solely for
learning* and correlates it with the change in loss. Here, the NNN's crossing
activation already fluctuates from the injected noise during the ordinary forward
pass, so the fluctuation `z - E[z]` *is* the perturbation. `Cov(L, z)/Var(z)`
regresses the loss onto that spontaneous fluctuation — reusing computation the
network performs anyway rather than adding a separate perturbation channel.

## 7. Hardware-oriented interpretation

The rule is attractive for streaming / FPGA-style implementation because the
hidden-layer update needs **no backward multiplication by transposed weights**. It is
dominated by forward sampling plus simple running statistics.

Required per-unit statistics (all accumulable online over the `M x N` samples):

```
mean L,   mean z,   mean (L z),   mean (z^2)
```

from which `Cov(L,z) = mean(Lz) - mean(L) mean(z)` and
`Var(z) = mean(z^2) - mean(z)^2`.

Required operations:

- accumulation (running sums),
- multiplication and subtraction (covariance / variance),
- one (optionally approximate) division per unit for `g_z`,
- a small LUT or piecewise approximation for `phi_bar'(d)`,
- a multiply-accumulate for the outer product `delta_hat * z_prev`.

**Qualitative resource comparison.**

| Aspect | Ordinary backprop | Proposed method |
|---|---|---|
| Backward transposed-weight matmul | required (per layer) | **not required** |
| Extra transposed-weight memory / routing | required | not required |
| Forward/backward phase synchronization | tight | forward-only + statistics |
| Extra state per unit | error `delta` buffers | 4 scalar accumulators (L, z, Lz, z^2) |
| Nonlinearity backward | `phi'` evaluation | `phi_bar'` LUT (same order) |
| Samples per update | 1 | `M` (stochastic) |

The trade is: drop the transposed-weight backward path in exchange for `M` forward
samples and a handful of per-unit accumulators. Whether this is a net win depends on
the target hardware and is **not** claimed here without an actual implementation.

## 8. What the example script demonstrates

Task: 1-D regression `y = sin(x)`, `x in [-2pi, 2pi]`, using an ACTUAL library model
from `nnn/model.py` as the forward network (`input=1 -> hidden(H) -> hidden(H) ->
output=1`), selectable with `--noise`:

- `--noise gaussian` -> `nnn.model.SimpleNNNSample` (Gaussian-noise crossing);
- `--noise uniform`  -> `nnn.model.SimpleNNNUniformSample` (bounded-uniform crossing).

Both models draw `t` (`--num-samples`) stochastic samples internally and average them
at the output (`EnsembleMeanLayer`), so the network output — and therefore the readout
target — is an **expected value**. Per-sample hidden activations are captured from the
model with forward hooks.

Three learning methods are compared:

1. **`backprop`** — reference only. Ordinary PyTorch autograd (Adam) on the **same**
   selected model. It does use backward propagation; it is the exact-gradient baseline.
2. **`cov_only`** — hidden weights updated with covariance credit **only**
   (`Delta W ~ -g_z * z_prev`, no `phi'`); readout on the ensemble-mean features.
3. **`cov_deriv`** — proposed; covariance credit **times** the noise-induced crossing
   derivative (`Delta W ~ -g_z * phi'(d) * z_prev`); readout on the ensemble-mean.

All three start from identical initial weights. For `cov_only`/`cov_deriv` the network
is updated with **no autograd / no `.backward()`** — manual tensor arithmetic under
`torch.no_grad()`. Autograd is used only for the `backprop` reference.

**Hidden-credit estimator (`--credit`).**

- `per_input` (default): `g_z[n,i] = Cov_t(L_n, z_{n,i}) / (Var_t(z_{n,i}) + eps)`,
  estimated **within each input over the `t` samples**, giving an input-dependent
  credit `[N, H]`. Centring over `t` removes the confounding between-input loss
  variation, so this is a much less biased estimate of the local gradient `dL_n/dz`.
- `pooled`: `g_z[i]` estimated over all `N·t` samples at once — one global scalar per
  unit `[H]`. Simpler but cannot represent input-dependent credit and mixes in the
  between-input loss variation.
- `--credit-passes K` accumulates the statistics over `K` forward passes (effective
  samples `K·t`), a variance-reduction knob (helps most when `t` is small).

**Expected outcomes (default `per_input`, `t≈100`, 1500 epochs).**

- `backprop` (exact gradients, Adam) fits `sin(x)` almost perfectly (MSE ~5e-4).
- **`cov_deriv` genuinely learns the sine** (MSE ~0.03, a good-amplitude fit) and
  **clearly beats `cov_only`** (MSE ~0.11): adding the noise-induced derivative `phi'`
  supplies the correct `dz/dd` factor, so the update better matches the true gradient.
- Both approach the backprop reference — the forward-noise credit works without any
  transposed-weight backward pass.

**Why `per_input` matters so much (the key finding).** With the frozen initial features
an exact least-squares readout reaches MSE ~1e-5, so the basis is excellent and the job
is really hidden-credit assignment. The `pooled` credit is a single global scalar per
unit that (i) cannot express input-dependent credit and (ii) is dominated by the
between-input loss variation, so it barely learns (and can perturb the good features) —
MSE stays ~0.4. Switching to `per_input` (centre over `t` per input) removes that
confound and yields an input-specific, near-unbiased `dL_n/dz`, which drops the fit from
~0.41 to ~0.03 at equal budget. The expected-value (ensemble-mean) readout — averaging
the `t` binary samples — separately removes the single-sample `diag(Var z)` shrinkage
that would otherwise cap the readout. Together they make the method fit convincingly.

**A straightforward extension (`cov_deriv_field_gate`).** Multiply each hidden update
by a fixed or learnable per-unit noise-field gate `s_i` (e.g. a unit-wise `sigma`
vector), `Delta W_l[i,j] *= s_i`. This connects credit assignment to the NNN
recruitment/noise-field idea (units with zero field are detached and receive no
update). It is a one-line change on top of `cov_deriv` and is left as an extension.

## 9. Limitations

- `Cov(L, z)/Var(z)` is a **single-variable regression approximation** of `dL/dz`:
  even with `--credit per_input` it linearises and ignores cross-unit interactions
  (it does not equal the exact partial derivative when many units fluctuate together).
  The `pooled` variant is weaker still (global scalar per unit; mixes in between-input
  loss variation) and is kept mainly as a baseline / ablation.
- **A residual accuracy floor (~0.015 MSE) that is estimator bias, not slow
  convergence.** `cov_deriv`(`per_input`) plateaus by ~3000–4000 epochs and does not
  improve with more epochs. Empirically this floor is a **systematic bias of the
  covariance estimator** (linearisation + the large binary fluctuation), not a
  reducible "noise ball": learning-rate decay (`--lr-decay cosine`) does **not** lower
  it, and an Adam step (`--opt adam`) makes it **worse** — Adam normalises by `1/√v`,
  which amplifies the high-variance covariance gradients (and `lr=1e-2` is far too
  large for Adam here). Only larger `t` helps, and marginally (t=100→300: 0.016→0.014).
  Reaching backprop-level (~5e-4) would need a **less-biased credit estimator** (e.g.
  higher-order/curvature-corrected, or a genuinely small-perturbation regime), which
  is beyond this proof-of-concept. The `--opt` / `--lr-decay` knobs are provided for
  experimentation but default to `sgd` / `none`.
- The method needs **multiple stochastic forward samples** (`t`) per update, trading
  compute/variance for the removal of the backward pass.
- The **output layer uses a local readout gradient** on the ensemble mean, so the
  demonstration removes backward propagation only from the hidden layers, not the
  readout.
- This is a **proof-of-concept**, not an optimized or validated hardware
  implementation; the regression task is a toy problem.
- More realistic benchmarks (deeper nets, classification) and concrete resource
  estimates are needed before any hardware-superiority claim.

## 10. Implications for manuscript writing

The algorithm can be positioned as **a hardware-friendly approximate
backpropagation method for NNNs, combining forward-noise-based credit assignment with
a noise-induced activation derivative**.

**Safe claims:**

- "approximates hidden-layer credit assignment without explicit backward propagation
  through transposed weight matrices";
- "reuses forward stochastic samples as implicit node perturbations";
- "uses the statistical derivative of the crossing activation to convert activity
  credit into weight updates".

**Claims to avoid:**

- "fully replaces backpropagation";
- "exactly computes backpropagation gradients";
- "proves FPGA superiority" (without an actual hardware implementation).

## 11. How to run

```bash
python examples/forward_noise_covariance_learning.py
```

Select the noise/model and arguments:

```bash
python examples/forward_noise_covariance_learning.py --noise gaussian   # SimpleNNNSample (default)
python examples/forward_noise_covariance_learning.py --noise uniform    # SimpleNNNUniformSample
python examples/forward_noise_covariance_learning.py --epochs 1500 --num-samples 64 --hidden-dim 64
```

Compare the credit estimators (this is the decisive knob):

```bash
python examples/forward_noise_covariance_learning.py --credit per_input --num-samples 100  # good fit
python examples/forward_noise_covariance_learning.py --credit pooled    --num-samples 100  # underfits (ablation)
```

CPU-friendly defaults: `noise=gaussian`, `credit=per_input`, `credit-passes=1`,
`opt=sgd`, `lr-decay=none`, `epochs=1500`, `hidden-dim=64`, `num-samples=64` (the
model's internal `t`), `lr=1e-2`, `sigma=0.5` (Gaussian std), `radius=1.0` (uniform
half-width), `crossing-h=0.2`, `seed=0`, `device=cpu`, `hidden-lr-scale=1.0`
(`--opt adam` / `--lr-decay cosine` are available but do not lower the floor — see §9).
A fast smoke test:

```bash
python examples/forward_noise_covariance_learning.py --epochs 50 --num-samples 16 --hidden-dim 16
```

## 12. Figures

The script displays three figures with `plt.show()` (nothing is written to disk):

```
learning curves        # backprop vs cov_only vs cov_deriv (log-scale MSE)
predictions on sin(x)  # target and the three method predictions
cov_deriv layer-1 stats# mean activity, g_z, and phi'(d) per hidden unit
```

The console also prints the final MSE of each method, a short interpretation, and
whether `cov_deriv` improved over `cov_only`.
