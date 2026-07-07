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
[`tmp/forward_noise_covariance_learning.py`](../tmp/forward_noise_covariance_learning.py).

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

The `cov_deriv_gate` method (§8.1) makes this reading **explicit**: it injects an extra
small perturbation `xi` into a gated block of pre-activations and regresses the loss onto
`xi` instead. That is unbiased by construction (the perturbation is independent of the
other units) but higher-variance — the classic node-perturbation bias/variance trade.

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

**Default verification set (what the script runs).** Running the script compares six methods:
`backprop`, `cov_only`, `cov_deriv_analytic`, `cov_deriv_kde`, `cov_jac_sgd`, `cov_jac_adam`
(the two `cov_jac` rows differ only in the local optimiser and both use `--jac-track` **on by
default**). The gate/field methods below are kept in the code but run only with `--include-gates`.
Use `--fit-check` for a focused figure confirming `cov_jac_adam` fits `sin(x)` as tightly as
`backprop`. All methods are described next; the default six are a subset of the eight.

Eight learning methods are implemented:

1. **`backprop`** — reference only. Ordinary PyTorch autograd (Adam) on the **same**
   selected model. It does use backward propagation; it is the exact-gradient baseline.
2. **`cov_only`** — hidden weights updated with covariance credit **only**
   (`Delta W ~ -g_z * z_prev`, no `phi'`); readout on the ensemble-mean features.
3. **`cov_deriv`** — proposed, **the default method**. Covariance credit **times** the
   crossing's **own distribution-free** density slope `(xor2 - xor1)/(2h)`
   (`--slope kde`, default), so no analytic noise model is used (see §8.2). `Delta W ~
   -g_z * (dz/dd)_kde * z_prev`; readout on the ensemble-mean.
4. **`cov_deriv_analytic`** — ablation: the same rule with the **hand-coded analytic**
   `phi'(d)` (`--slope analytic`). It matches `cov_deriv`, confirming the analytic phi' is
   not needed (see §8.2).
5. **`cov_deriv_gate`** — proposed **perturbation gate** (see §8.1): credit from an
   explicitly injected local perturbation, `Cov(L, xi)/Var(xi)`, on a rotating block of
   hidden units; readout on the ensemble-mean.
6. **`cov_jac`** — proposed **structured / recursive** covariance credit (see §8.4):
   propagate the exact output error **down the graph** using forward weights recovered from
   covariance, `W_hat[l+1] = Cov(d[l+1], z[l])/Var(z[l])` (EMA-smoothed) — a forward-only,
   weight-transport-free reconstruction of true backprop. **Beats `cov_deriv`** at full
   budget.
7. **`cov_deriv_gate_crn`** — proposed **antithetic / common-random-number gate** (see
   §8.3): the gate's `xi` credit, but from a paired `(+xi, -xi)` forward under identical
   crossing noise, `Cov(L(+xi)-L(-xi), xi)/(2 Var xi)`. Unbiased and low-variance *in
   principle*; a negative result on the binary crossing (see §8.3).
8. **`cov_deriv_field_gate`** — proposed **noise-field / recruitment gate** (see §8.5):
   `cov_deriv` with each hidden update multiplied by the unit's noise-field strength
   `s_i`, `Delta W_l[i,j] *= s_i`. With `--field-sparsity 0` (default) the gate is
   all-ones and the rule is exactly `cov_deriv`; with `--field-sparsity f>0` a fraction
   `f` of hidden units are **un-recruited** (`s_i = 0` — zero forward noise, hence dead,
   **and** zero update), tying credit assignment to the NNN recruitment/noise-field idea.

All eight start from identical initial weights. For the `cov_*` methods the network
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

**A straightforward extension (`cov_deriv_field_gate`), now implemented (see §8.5).**
Multiply each hidden update by a fixed or learnable per-unit noise-field gate `s_i`
(e.g. a unit-wise `sigma` vector), `Delta W_l[i,j] *= s_i`. This connects credit
assignment to the NNN recruitment/noise-field idea (units with zero field are detached
and receive no update). It is a one-line change on top of `cov_deriv`.

### 8.1 The perturbation gate (`cov_deriv_gate`)

**Motivation — bias of the spontaneous-fluctuation covariance.** `cov_only`/`cov_deriv`
regress the loss onto each unit's **own spontaneous** forward fluctuation
`z - E[z]`. That fluctuation is large (the crossing output is binary), which gives a
low-variance estimate, but it is **shared** across units that fluctuate together, so
`Cov(L, z)/Var(z)` is *biased* by cross-unit interactions (see §6, §10). The perturbation
gate is intended to **reduce this covariance-credit bias by correlating the loss with an
explicitly injected local perturbation, rather than with all spontaneous unit
fluctuations** — turning the *implicit* node perturbation of §6 into an *explicit*,
controlled one.

**Mechanism.** On top of the model's own forward noise, a small extra perturbation is
injected into the **pre-activation** `d` of a rotating **block** `G_k` of hidden units,
via forward pre-hooks so it enters the **actual loss-bearing forward pass** (it is *not*
a post-hoc update mask):

```
d_sample = d + base_nnn_noise + alpha * G_k * xi        xi ~ N(0, 1)   (per sample)
```

The model's noise field / stochastic crossing is left completely intact; `xi` is added on
top, only on the gated block. The injected perturbation `p = alpha * G_k * xi` is recorded
per pass and the hidden credit is taken from **its** covariance with the loss:

```
g_d[n, i] = Cov_t(L_n, p_{n, i}) / (Var_t(p_{n, i}) + eps)   ~=   dL_n / dd_{n, i}
delta_hat[n, i] = g_d[n, i]                                  (gated units only)
```

Because `p` is injected **at the pre-activation `d`**, `Cov(L, p)/Var(p)` already estimates
`dL/dd = delta` **directly** — the local crossing slope is captured implicitly through the
loss's response to `p`. Multiplying by `phi'(d)` again would double-count that slope, so
`cov_deriv_gate` uses `delta_hat = g_d` **without** the extra `phi'` factor (this is the
one structural difference from `cov_deriv`, whose credit `g_z` needs the explicit `phi'`).

**Gating.** Each epoch a contiguous block of `--gate-block-size` units is selected —
cyclically (`--gate-mode cyclic`, default) or at random (`--gate-mode random`). **Only the
gated block is perturbed, and only the gated block is updated** (non-gated credit is zeroed,
so their rows of `Delta W_l` and `Delta b_l` are exactly zero). CLI knobs:
`--gate-block-size` (8), `--gate-alpha` (0.05, the perturbation strength), `--gate-mode`
(`cyclic`).

**Empirical result — an unbiased but higher-variance credit.** At the CPU-friendly default
budget the perturbation gate **does not beat `cov_deriv`**, and the example prints this
honestly. The reason is a **bias/variance trade**, not a bug:

- The injected `xi` is *independent* of every other source, so `Cov(L, xi)/Var(xi)` is an
  **unbiased** estimate of `dL/dd` (the stated bias-reduction goal) — but the signal
  `dL/dd_i · alpha` is a *small* perturbation of `L`, while `L` also fluctuates from **all
  other units'** large intrinsic crossing noise. That nuisance variance is not removed by
  gating (ungated units still fire), so the estimator variance is high: its standard error
  scales like `~ sqrt(Var L / T) / alpha`. Smaller `alpha` → less-perturbed forward but
  *noisier* credit.
- Isolating credit quality (gate every unit every epoch, so there is no update-frequency
  penalty) still leaves `cov_deriv_gate` well above `cov_deriv` at `t≈48`, confirming the
  gap is estimator **variance**, not the block schedule.
- Realising the bias advantage needs variance reduction: much larger `t` (or
  `--credit-passes`), a larger `--gate-alpha` (at the cost of perturbing the forward more),
  or a paired/antithetic baseline (a genuine small-perturbation node-perturbation regime) —
  the last is beyond this proof-of-concept.

So `cov_deriv_gate` is included as a **conceptually cleaner (unbiased) but higher-variance**
credit rule that makes the §6 "implicit node perturbation" reading explicit, and as a direct
demonstration of the bias/variance trade behind the residual floor discussed in §10.

> **A cleaner source of the same "perturbation" signal.** The gate injects an *external*
> perturbation and pays a large variance for it. But the crossing activation **already
> contains an internal antithetic / common-random-number perturbation** — the dual `+/-h`
> thresholds evaluated on the *same* samples — which yields a low-variance, distribution-free
> local slope for free. Exploiting that (rather than an external `xi`) is the **default**
> `cov_deriv` slope, next.

### 8.2 Distribution-free crossing slope from `xor1`/`xor2` (the default `cov_deriv`)

The local slope factor `dz/dd` can come from an **analytic** `phi'(d)` hand-coded per noise
distribution (`2(1-2F)p` for Gaussian, `-(d-c)/r^2` for uniform — the `phi_prime()`
function), **or** from the crossing's **own** internal density estimator, valid for *any*
noise distribution. The example uses the latter **by default** (`--slope kde`); `--slope
analytic` selects the former as an ablation (`cov_deriv_analytic`).

**What `CrossingSample.backward` computes.** For a hidden unit with noisy samples
`s_t = d + eta_t`, the activation binarises at two thresholds and counts level crossings,

```
bin1_t = 1[s_t > +h]      xor1 = |bin1_{t+1} - bin1_t|     (crossings of level +h)
bin2_t = 1[s_t > -h]      xor2 = |bin2_{t+1} - bin2_t|     (crossings of level -h)
z_t = (xor1_t + xor2_t)/2                                   (the crossing output)
```

and its backward returns, **per unit**,

```
coeff = mean_t(xor2 - xor1) / (2h)   ~=   d z_bar / d d      (a kernel density estimate)
```

**Why this is exactly an antithetic / common-random-number estimator.** Evaluating the
crossing at `+h` and at `-h` on the **same** sample set `{s_t}` is a symmetric finite
difference over a threshold shift of `2h`. Shifting the threshold by `+/-h` is equivalent to
shifting the pre-activation `d` by `-/+h`, so `(xor2 - xor1)/(2h)` is precisely
`[z(d+h) - z(d-h)] / (2h)` evaluated with **common random numbers** (the shared `s_t`). The
shared samples make the noise **cancel in the difference**, which is why the estimate is
low-variance and — crucially — **distribution-free**: it never references the Gaussian or
uniform density, it just measures how often the realised samples straddle the two thresholds.
This is the same "antithetic + CRN" device that would be needed to de-noise the perturbation
gate (§8.1) — except the crossing performs it *internally, for free*, on its own samples
rather than on an injected `xi`.

**The default slope.** Use that internal slope in place of the analytic `phi'`:

```
delta_hat[l, i] = g_z[l, i] * coeff[l, i]        coeff = (xor2 - xor1)/(2h)   (per unit)
```

In the example, `coeff` is read out with a **local, single-layer** grad-enabled re-run of
just that crossing module (`kde_slope()`): because the additive noise `d -> d + eta` has unit
gradient w.r.t. `d`, autograd routes the library's own `(xor2 - xor1)/(2h)` straight onto the
pre-activation. This is a purely **local activation derivative** — no readout, **no
transposed-weight backward**, fully consistent with the method's constraint.

**Empirical result.** The default (`kde`) `cov_deriv` **matches the analytic-`phi'` ablation
(`cov_deriv_analytic`)** to within noise on both models — Gaussian and uniform, at `t≈48`,
`H=32` — while using **no analytic noise model at all**. So the hand-coded `phi_prime()` is
not needed, and the default rule is the more faithful "NNN-native", distribution-agnostic
form. (This addresses the *slope* factor `dz/dd`; it does **not** change the covariance
*activity* credit `dL/dz`, so it does not by itself lower the residual floor of §10 — that
floor lives in the `dL/dz` estimator, not in `phi'`. That estimator is the subject of §8.4.)

### 8.3 Antithetic / common-random-number gate (`cov_deriv_gate_crn`) — a negative result

`cov_deriv_gate` (§8.1) is unbiased but high-variance because the injected `xi` signal is
swamped by every other unit's intrinsic crossing noise in `L`. The textbook fix is
**common random numbers (CRN)**: run a paired `(+xi, -xi)` forward with the **same** noise
held fixed, and correlate the loss **difference** with `xi`,

```
L_plus  = L(d + xi ; eta)          eta drawn from RNG state S
L_minus = L(d - xi ; eta)          RNG reset to S  =>  identical eta
g_d[i]  = Cov_t(L_plus - L_minus, xi_i) / (2 Var_t(xi_i))   ~=   dL/dd_i
```

In the example this is done with a genuine RNG-state snapshot/restore around the two
forwards (`rng_snapshot`/`rng_restore`), and the readout / `z_prev` use the antithetic
**average** `(L_plus+L_minus)/2` (the `O(alpha)` perturbation cancels). The CRN is
**bit-exact**: two forwards under a restored RNG state differ by `0.0`, so the intrinsic
noise `eta` truly cancels in `L_plus - L_minus`.

**Yet it barely helps** — `cov_deriv_gate_crn` (≈0.386) is only marginally better than
`cov_deriv_gate` (≈0.395) and still far worse than `cov_deriv` (≈0.157), at equal budget,
across `alpha in {0.1 … 1.0}` (larger `alpha` only adds bias and makes both worse). This is
a **real, informative negative result**, and it is specific to the **binary level-crossing**
nonlinearity:

- CRN removes the `eta` *nuisance* variance, but the crossing output `z_i` is a **step
  function** of `d_i`. Shifting `d_i` by a small `+/- xi` with the samples held fixed flips
  `z_i` only on the **measure-zero** set of samples whose threshold lies within `[d-xi,
  d+xi]`. So `L_plus - L_minus` is **mostly exactly zero**, with a few discrete jumps: the
  *pathwise* derivative of a binary crossing is degenerate. The estimator variance is now
  dominated by this **discreteness**, not by `eta`, so the CRN cancellation buys almost
  nothing.
- This is precisely **why the NNN crossing carries a density-estimating backward in the
  first place** (§8.2). The informative quantity is the **rate** at which samples cross the
  threshold — a smooth density that `xor1`/`xor2` estimate by **counting crossings over all
  `T` samples** — not the pathwise response of any single realisation to a shift. Node
  perturbation (even antithetic CRN) extracts information only from the few samples that
  happen to flip and wastes the rest; the internal density estimator uses them all.

**Takeaway.** For binary/level-crossing (spiking-like) units, **external node perturbation is
the wrong tool** — the pathwise gradient is degenerate, and neither a bigger perturbation nor
CRN rescues it. The crossing's **own** density estimator (§8.2) is the right,
low-variance, distribution-free way to obtain the local slope, and the covariance credit
`Cov(L, z)/Var(z)` — which regresses onto the unit's *large spontaneous* fluctuation — is a
far stronger activity-credit signal than any small injected `xi`. `cov_deriv_gate_crn` is kept
as an instructive ablation that makes this point concretely.

### 8.4 Structured (recursive) covariance credit (`cov_jac`) — *what to take the Cov against*

Everything above estimates hidden credit from `Cov(L, z_i)` — the correlation of a unit's
fluctuation with the **scalar global loss**. That single-variable regression **collapses the
entire downstream network onto one number**, discarding all structure (which downstream units
a hidden unit actually feeds, and how). This collapse is the source of the residual
estimator-bias floor of §10. A natural question — *what else could we take the covariance
against?* — leads to a **structure-aware** credit that follows the actual computational graph.

**Key building block (verified).** Because a pre-activation is **linear** in the previous
layer's activity, `d^{l+1} = W^{l+1} z^l + b`, the forward noise lets us **measure the forward
weight matrix directly from covariance**, with no explicit transpose:

```
Cov_t(d^{l+1}_j, z^l_i) / Var_t(z^l_i)   ~=   W^{l+1}_{ji}
```

Empirically (Gaussian, `H=24`, `t = 256 x 80` samples, per input) this recovers `W^{l+1}` with
**Pearson r ≈ 0.97** (median magnitude ratio ≈ 0.85; the residual is finite-sample variance).
*Important detail:* one must correlate with the **continuous pre-activation `d^{l+1}`**, not
the **binary** `z^{l+1}` — correlating with `z^{l+1}` gives only `r ≈ 0.14` (the same
binary-degeneracy problem as §8.3).

**The recursive rule.** With the forward weights recovered as `Ŵ`, credit can be propagated
**layer by layer** exactly like backprop, but forward-only and weight-transport-free:

```
delta_out                = 2 (y - t)                                (exact local output error)
delta^l_i = (dz/dd)^l_i * sum_j [ Cov(d^{l+1}_j, z^l_i)/Var(z^l_i) ] * delta^{l+1}_j
          = (dz/dd)^l_i * sum_j  Ŵ^{l+1}_{ji}  delta^{l+1}_j
```

with the slope `(dz/dd)` from the §8.2 KDE estimate. This replaces backprop's
`W^{l+1,T} delta^{l+1}` with a **covariance-estimated** `Ŵ^{l+1,T} delta^{l+1}`. Compared with
`cov_deriv` (which uses a **single scalar** `Cov(L, z_i)` per unit), the recursive rule uses a
full `[H_{l+1} x H_l]` covariance **and** the downstream `delta` — i.e. it uses the network's
structure, and is much closer to the exact gradient.

**Why it should lower the floor.** It never collapses onto the scalar `L`; credit flows along
the real graph, and each step is a clean low-dimensional regression whose linear part (the
weight) is recovered nearly unbiased (`r ≈ 0.97`).

**Relation to known ideas.** This is exactly the **weight-mirror** mechanism (Akrout et al.,
2019) — estimating `W^T` by correlating activity fluctuations — realised *natively* by the
NNN's intrinsic noise, and it is closely related to **feedback alignment / target propagation**
and to noise-based Jacobian estimation. The NNN is unusual in that the perturbations needed to
run a weight mirror are **already present for free** in every forward pass.

**Implementation (`cov_jac`).** Each epoch it (i) measures the forward weights from
covariance — `W_hat[l+1] = mean_n Cov_t(d[l+1], z[l])/Var_t(z[l])` for the hidden layers and
`Cov(y_samples, z[top])/Var(z[top])` for the readout — and smooths them with an EMA
(`--jac-ema`, default 0.9) for variance reduction; then (ii) runs the recursion above from the
exact output error `2(y-t)`, using the §8.2 KDE slope. The per-layer weight update is the same
outer product `delta^l * z_prev^T` as the other methods. Only forward passes and covariance
statistics are used — **no transposed-weight backward**.

**Empirical result — it lowers the floor.** At full budget (Gaussian, `H=32`, `t=64`,
`lr=1e-2`, **1500 epochs**), `cov_jac` **consistently beats `cov_deriv`** across seeds:

| seed | `cov_deriv` | `cov_jac` |
|---|---|---|
| 0 | 0.0588 | **0.0407** |
| 1 | 0.0513 | **0.0363** |
| 2 | 0.0600 | **0.0435** |

a ~30% reduction of the residual MSE, moving toward the backprop reference (~0.001). The gain
comes precisely from **not** collapsing onto scalar `L`: credit follows the graph via the
recovered weights.

**Cost — slower convergence (a real trade).** `cov_jac` converges **more slowly early on** (the
weight-mirror EMA needs time to stabilise and the recursion adds variance), so at a *short*
budget (e.g. 1000 epochs) it can trail `cov_deriv` before overtaking it. It reaches a **lower
floor**, not a faster descent. A larger `--jac-ema` trades early noise for slower tracking.

**Other costs / open risks.**
- Per-layer covariance-Jacobian is `O(H^2)` statistics per layer (vs `O(H)` for the scalar
  rule), though still forward-only and online-accumulable.
- Variance still **compounds through the recursion** (a product of estimated Jacobians); the
  EMA mirror mitigates but does not remove this. Depth beyond two hidden layers is untested.
- The slope comes from the density estimator (§8.2); correlating with the **binary `z`** instead
  of the continuous `d` breaks the weight recovery (r≈0.14, §8.4 above) — the implementation
  correlates with `d`. The readout update stays local/exact.

**Status.** Implemented and verified as the `cov_jac` method: a forward-only, weight-transport-
free reconstruction of backprop that **pushes below the `cov_deriv` estimator-bias floor** at
the cost of slower convergence — the most principled of the variants explored here.

**Faster/lower-variance mirrors (`--jac-track`, Kolen–Pollack + pooling).** Two refinements
attack the mirror's cost directly. (1) *Tracking (idea 1 / Kolen–Pollack):* we apply the weight
update ourselves, so the true weights move by a **known** increment `-DW`; integrating that same
`DW` into the mirror (`W_hat <- W_hat - DW`) makes it follow the moving weights **exactly**, so
the covariance measurement only has to fix the *static* initial offset instead of chasing a moving
target. (2) *Pooling (idea 2):* because `W` is **input-independent** (unlike the gradient), the
per-input-centred mirror statistics can be summed over inputs, `W_hat = (sum_n Cov_t)/(sum_n Var_t)`
— all `N*T` samples for one estimate. Per-input centring is kept, so the between-input cross-unit
confound (the same one that made `pooled` worse than `per_input` for `cov_deriv`) is **not**
reintroduced. Both are enabled by `--jac-track`.

**Is there an irreducible (directional) bias, or only reducible estimation error?** Verified
empirically: **`cov_jac`'s credit is an essentially *unbiased* estimate of the true backprop
gradient.** Averaged over stochastic draws, the cosine similarity between `cov_jac`'s per-layer
update and the exact autograd gradient is **≈ 0.9996–1.0000 with magnitude ratio ≈ 1.0** for every
layer (input→h1, h1→h2, readout), both untrained and partially trained. This is structurally
different from `cov_deriv`: `Cov(L, z)/Var(z)` collapses the whole downstream onto the scalar `L`,
a linearisation whose bias does **not** vanish with more samples (a real floor, §10); `cov_jac`
never collapses — it recovers the actual weights (the mirror hits Pearson `r ≈ 1.0`, magnitude
`≈ 1.0`, already at `t = 64` under pooling) and runs the exact chain rule, so its error is
**reducible estimation variance, not a directioned bias**. Consequently more samples/updates (and
matching the optimiser) *do* drive `cov_jac` toward the backprop solution; the practical gap is
dominated by **optimisation** (`sgd`, `lr=1e-2` vs the Adam baseline) and small finite-sample
mirror variance, both reducible. The measured benefit of `--jac-track` in this regime is modest
(~2% final MSE) precisely *because* the mirror is already near-exact at `t = 64` — the refinements
bind when the mirror is the bottleneck (small `t`, high `lr`, or deeper nets), not here.

**Confirmation — with a matched (Adam) optimiser, `cov_jac+track` reaches backprop-level final
error.** Because the gradient is essentially unbiased, the residual gap must be optimisation, and
it is: swapping SGD for Adam (a *local* per-weight rule that keeps the no-weight-transport /
FPGA-friendly property — it never touches the backward path) collapses the gap almost entirely
(H=32, t=64, 1500 epochs, seed 0, `--jac-track`):

| method | optimiser | final MSE (8-pass predict) |
|---|---|---|
| backprop (reference) | Adam `lr=1e-2` | **0.0011** |
| `cov_jac+track` | Adam `lr=3e-3` | **0.0012** |
| `cov_jac+track` | Adam `lr=1e-2` | 0.0015 |
| `cov_jac+track` | **SGD** `lr=1e-2` | 0.0302 |

So the ~0.03 "floor" of `cov_jac` was an **SGD artefact, not an estimator bias**: Adam drops it
~25× to **backprop level** (0.0012 vs 0.0011). Note this is the *opposite* of `cov_deriv`, where
Adam makes things **worse** (§10): Adam's `1/sqrt(v)` normalisation amplifies variance, which hurts
`cov_deriv`'s high-variance scalar-`Cov(L,z)` credit but is harmless for `cov_jac`'s low-variance,
near-exact-gradient credit. This is the sharpest evidence that `cov_jac` carries **no irreducible
directional bias** in this regime — a forward-only, weight-transport-free rule that attains the
exact-backprop solution.

**The one conditional caveat (where a directional bias *would* appear).** The single-variable
mirror `Cov(d, z_i)/Var(z_i)` equals `W_{ji}` only if a layer's units are **uncorrelated**. This
holds exactly for the first hidden layer (independent injected noise at a fixed input) and remains
**empirically** near-exact for deeper layers here (`r ≈ 1.0`), because each unit's *own* crossing
noise dominates its variance and swamps the small shared-upstream correlation. But it is
**regime-dependent**: with weak injected noise relative to the deterministic drive, or in deeper
nets where correlations compound, the diagonal mirror would pick up neighbour weights — a
directional, **sample-irreducible** bias. The principled fix is the **multivariate** mirror
`W_hat = Cov(d, z) · Cov(z, z)^{-1}` (exact by the linearity `d = W z + b`), at `O(H^3)` per layer
and needing more samples to condition `Cov(z, z)`. The KDE-slope bandwidth bias `O(h^2)` is shared
with the autograd baseline (same `coeff`), so it is not a relative gap.

**Tested — the bias is real but empirically small in these regimes.** Driving the readout layer's
own noise weak (so upstream-driven cross-unit correlation dominates) does induce the predicted
directional bias, and it is confirmed **sample-irreducible**: at `sigma_readout = 0.05` the
single-variable readout mirror sits at Pearson `r ≈ 0.994` and **stays there** as `T` grows
`128 -> 8192` (not a variance that averages out), while the **multivariate** mirror is exact
(`r = 1.000`) throughout; the first-layer mirror stays clean (`r = 1.000`) as a control. But the
magnitude is modest: even with **three hidden layers** and strongly decreasing per-layer noise
(`stds = [0.5, 0.2, 0.05]`), the deepest single-variable mirror only falls to `r ≈ 0.991`
(off-diagonal `|corr|` of the layer's units stays `≤ 0.012`) — the confound does **not** compound
sharply with depth here, because each unit's intrinsic crossing noise keeps the units
well-decorrelated. So the single-variable mirror is robust for these NNN regression nets; the
multivariate mirror is the exact fallback if a strongly cross-correlated regime (dense/large
weights, a shared noise field, or much deeper nets) ever makes the diagonal approximation bite.
The single-variable-vs-multivariate mirror trade-off is discussed in full in the **Discussion (§9)**.

### 8.5 Noise-field / recruitment gate (`cov_deriv_field_gate`)

Everything above updates **every** hidden unit. The NNN, however, has an intrinsic **noise
field** (a per-unit noise strength / recruitment level): a unit with zero field injects no
noise, so its crossing output cannot fluctuate and it carries no information. `cov_deriv_field_gate`
makes credit assignment **respect that field** — the one-line extension flagged in §8.

**Mechanism.** Each hidden unit `i` has a noise-field strength `s_i`. The `cov_deriv` update is
gated by it,

```
delta_hat[l, i] = g_z[l, i] * phi_bar'(d_{l,i}) * s_i        (Delta W_l[i,j] *= s_i)
```

so an **un-recruited** unit (`s_i = 0`) receives **no update** and is detached from learning.
Crucially the field is a *real* per-unit noise field, not just an update mask: the example scales
each unit's Gaussian `std` (or uniform `radius`) by `s_i`, so an `s_i = 0` unit gets **zero
forward noise** — it is deterministic across the `t` samples, its crossing output is constant
(dead), and its covariance credit `Cov(L, z_i)/Var(z_i)` is therefore ≈ 0 anyway. The gate makes
this exact (a hard zero) rather than an `eps`-regularised near-zero, and generalises to any
continuous field `s_i ∈ [0, 1]` (a soft recruitment weighting).

**Consistency check.** With `--field-sparsity 0` (default) the field is all-ones, the gate is a
no-op, and the rule is **exactly `cov_deriv`** (the small MSE difference in the printout is only
training stochasticity — the two nets draw independent forward noise). With `--field-sparsity f>0`
a deterministic fraction `f` of hidden units per layer are set to `s_i = 0`; the shared network
then genuinely has that many dead units, and only `cov_deriv_field_gate` explicitly refrains from
updating them (the other methods leave them at their initial, non-learning state via the ≈0
covariance credit).

**Why it is included.** It is the cheapest possible bridge from the forward-noise credit rule to
the NNN's **recruitment/noise-field** machinery (`noise_field_demo.py`, `recruitment_field_demo.py`):
credit assignment and neuron recruitment share the same per-unit field. It is a proof-of-concept
hook — a learnable `s_i` (recruiting units where credit is high) is the natural next step and is
left open.

**Status.** Implemented as the `cov_deriv_field_gate` method and verified: un-recruited units are
confirmed dead (zero activity variance) and receive exactly zero update; at `--field-sparsity 0`
the rule reduces to `cov_deriv`.

## 9. Discussion — single-variable vs. multivariate weight mirror

This section collects the analysis behind the one residual, *directional* error identified for
`cov_jac` (§8.4) and discusses the **multivariate weight mirror** as its principled — but not
free — fix. The multivariate mirror is **not** part of the shipped example; it was validated in a
standalone probe and is recorded here as Discussion / future work.

### 9.1 Why the single-variable mirror is a diagonal approximation

`cov_jac` recovers each forward weight with a **single-variable** regression slope,

```
W_hat[l+1]_{ji} = Cov_t(d^{l+1}_j, z^l_i) / Var_t(z^l_i).
```

Because the pre-activation is exactly linear, `d^{l+1}_j = sum_k W_{jk} z^l_k + b_j`,

```
Cov(d^{l+1}_j, z^l_i) = sum_k W_{jk} Cov(z^l_k, z^l_i),
so   W_hat_{ji} = W_{ji} + sum_{k != i} W_{jk} * Cov(z^l_k, z^l_i) / Var(z^l_i).
```

The second term is the **bias**: it leaks a unit's *neighbours'* weights in proportion to the
**off-diagonal covariance** of the layer's activations. It vanishes iff the layer's units are
mutually uncorrelated (`Cov(z, z)` diagonal). Its sign and size are set by the neighbour weights
and the correlations, so it is **directional**, and it is a property of the *true* covariance, so
**more samples do not remove it** (confirmed empirically: the single-variable readout mirror sits
at `r ≈ 0.994` and stays there as `T` grows `128 -> 8192`, §8.4). In short, the single-variable
mirror is the exact regression **pre-multiplied by the diagonal of `Cov(z, z)^{-1}` instead of its
full inverse**.

### 9.2 The exact object: the multivariate mirror

Since `d = W z + b` is exactly linear, the **ordinary least-squares** estimate is unbiased under
regressor correlation:

```
W_hat = Cov(d, z) * Cov(z, z)^{-1}       (per input, then pooled over inputs; ridge-regularised).
```

In the standalone probe this recovers `W` with Pearson `r = 1.000`, magnitude ratio `1.00`, in
**every** tested regime — weak readout noise (`sigma = 0.05`) and three hidden layers with strongly
decreasing per-layer noise — where the single-variable mirror falls to `r ≈ 0.99`. So the
multivariate mirror **removes the only identified directional bias** exactly.

### 9.3 The trade-off — exactness vs. locality/cost

The multivariate mirror is not free, and the cost is exactly of the kind the method set out to
avoid:

- **Compute/state:** an `O(H^3)` inverse (or linear solve) per layer per update and `O(H^2)` state
  for the full `Cov(z, z)`, versus `O(H^2)` / `O(H)` for the diagonal mirror.
- **Conditioning:** `Cov(z, z)` must be well conditioned — it needs more samples, and ridge /
  shrinkage regularisation, especially for large `H` or small `t`, or near-collinear units.
- **Hardware / locality:** the inverse is a **globally coupled, non-local** operation over a whole
  layer. It does **not** reintroduce weight transport (we still never read `W^T`), but it *does*
  partly give back the streaming / per-unit-locality advantage that motivated the forward-noise
  approach. So it is **less FPGA-friendly** than the diagonal mirror.

The single-variable mirror is therefore the better **default** — cheap, local, and empirically
near-exact in these NNN nets — while the multivariate mirror is the exact **fallback** for regimes
where the diagonal approximation would bite.

### 9.4 Cheaper middle grounds (considerations)

Between the diagonal slope and a full inverse there is a spectrum worth exploring before paying
`O(H^3)`:

- **Shrinkage / block-diagonal `Cov(z, z)^{-1}`:** correct only the largest off-diagonal
  correlations (e.g. within local blocks of units), leaving the rest diagonal.
- **Inverse-free iterative solves:** apply `Cov(z, z)^{-1}` via a few Newton–Schulz or
  conjugate-gradient / Richardson steps — mostly matmul/MAC, streamable, no explicit inversion.
- **Recursive least squares (RLS) + Kolen–Pollack predict:** RLS maintains an *inverse-covariance*
  estimate incrementally; combined with the §8.4 trick of integrating the **known** applied
  increment `ΔW` (predict) and using RLS only to correct the static offset, this is the most
  principled hardware-aware route to a multivariate-quality mirror without a per-step inverse.
- **Whiten by design:** if the injected noise field or the connectivity were shaped so that
  `Cov(z, z)` is diagonal by construction (decorrelated per-unit fluctuations), the *diagonal*
  mirror would be exact for free — a design lever that ties back to the NNN noise-field idea and
  the recruitment gate (§8.5).

### 9.5 Recommendation and open questions

- **Keep the single-variable mirror as the default** (`cov_jac`, `--jac-track`): local, cheap, and
  within noise of the exact gradient here (cosine ≈ 1.0, §8.4).
- **Offer a multivariate / RLS mirror as opt-in** for strongly cross-correlated regimes: dense or
  large-magnitude weights, structured/shared noise fields, correlated-feature classification
  tasks, or much deeper nets.
- **Characterise the crossover.** A useful manuscript result would be a scalar diagnostic — mean
  `|off-diagonal corr|` of a layer, or the condition number of `Cov(z, z)` — and the threshold at
  which the diagonal mirror's directional bias exceeds the optimisation/variance floor. That
  threshold is where the multivariate mirror earns its `O(H^3)`.
- **Depth.** The recursion multiplies per-layer mirror estimates, so even small per-layer
  directional biases can **compound**; the 2–3 hidden-layer tests show it staying small, but a
  whitening or multivariate correction may matter more for deep nets. Untested beyond three hidden
  layers.
- **Relation to known ideas.** The whole mirror mechanism is the **weight mirror** of Akrout et al.
  (2019) realised natively by the NNN's intrinsic noise; the `--jac-track` predict step is
  **Kolen–Pollack** (1994); the multivariate upgrade is ordinary multivariate regression / a
  whitened weight mirror. The NNN's distinguishing feature is that the perturbations these methods
  need are **already present for free** in every stochastic forward pass.

## 10. Limitations

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
  experimentation but default to `sgd` / `none`. **`cov_jac` (§8.4) is exactly such a
  less-biased estimator** — it propagates credit through the graph instead of collapsing
  onto scalar `L`, and its gradient is essentially *unbiased* (cosine ≈ 1.0 vs the true
  autograd gradient, §8.4). Its own `sgd` "floor" (~0.03) is therefore **not** an estimator
  bias but an optimisation artefact: `cov_jac+track` with **Adam** (still weight-transport-
  free) reaches **backprop-level** final error (~0.0012 vs ~0.0011, §8.4). This is the
  opposite of `cov_deriv` above, whose floor *is* estimator bias and which Adam worsens —
  the difference is gradient variance (low for `cov_jac`, high for `cov_deriv`).
- **`cov_jac`'s weight mirror carries one *conditional* directional bias** — the
  single-variable slope `Cov(d, z_i)/Var(z_i)` is confounded by cross-unit correlations of
  the layer's activations, a bias that does **not** vanish with samples. It is **empirically
  small** here (single-var recovery `r ≥ 0.99` even under weak noise / three hidden layers,
  because the intrinsic per-unit noise keeps units decorrelated), and the **multivariate**
  mirror removes it exactly at `O(H^3)` cost. Full treatment and cheaper alternatives are in
  the **Discussion (§9)**.
- The method needs **multiple stochastic forward samples** (`t`) per update, trading
  compute/variance for the removal of the backward pass.
- The **output layer uses a local readout gradient** on the ensemble mean, so the
  demonstration removes backward propagation only from the hidden layers, not the
  readout.
- This is a **proof-of-concept**, not an optimized or validated hardware
  implementation; the regression task is a toy problem.
- More realistic benchmarks (deeper nets, classification) and concrete resource
  estimates are needed before any hardware-superiority claim.

## 11. Implications for manuscript writing

The algorithm can be positioned as **a hardware-friendly approximate
backpropagation method for NNNs, combining forward-noise-based credit assignment with
a noise-induced activation derivative**.

**Safe claims:**

- "approximates hidden-layer credit assignment without explicit backward propagation
  through transposed weight matrices";
- "reuses forward stochastic samples as implicit node perturbations";
- "uses the statistical derivative of the crossing activation to convert activity
  credit into weight updates";
- (for `cov_jac`) "reconstructs the backprop weight update from forward-noise covariance
  alone — a weight-transport-free weight mirror — whose credit is empirically an **unbiased**
  estimate of the true gradient (cosine ≈ 1.0), and which, with a local Adam step, **reaches
  backprop-level final accuracy** on the toy task (MSE ~1e-3, matching the autograd baseline)".

**Claims to avoid:**

- "fully replaces backpropagation";
- "exactly computes backpropagation gradients" (the credit is *near*-exact, cosine ≈ 1.0, not
  bit-exact; and one conditional mirror bias exists, §9);
- "proves FPGA superiority" (without an actual hardware implementation).

## 12. How to run

```bash
python tmp/forward_noise_covariance_learning.py
```

**The default run compares the six verification methods** — `backprop`, `cov_only`,
`cov_deriv_analytic`, `cov_deriv_kde`, `cov_jac_sgd`, `cov_jac_adam` (both `cov_jac` rows use
`--jac-track`, on by default; they differ only in the local optimiser). To **confirm
`cov_jac_adam` fits `sin(x)` as tightly as backprop**, add the focused figure:

```bash
python tmp/forward_noise_covariance_learning.py --fit-check                 # target vs backprop vs cov_jac_adam (+residuals, MSE)
python tmp/forward_noise_covariance_learning.py --fit-check --save out/     # save all figures as PNG instead of plt.show()
python tmp/forward_noise_covariance_learning.py --include-gates             # also run the gate/field methods (off by default)
python tmp/forward_noise_covariance_learning.py --no-jac-track              # disable Kolen-Pollack mirror tracking
```

Select the noise/model and arguments:

```bash
python tmp/forward_noise_covariance_learning.py --noise gaussian   # SimpleNNNSample (default)
python tmp/forward_noise_covariance_learning.py --noise uniform    # SimpleNNNUniformSample
python tmp/forward_noise_covariance_learning.py --epochs 1500 --num-samples 64 --hidden-dim 64
```

Compare the credit estimators (this is the decisive knob):

```bash
python tmp/forward_noise_covariance_learning.py --credit per_input --num-samples 100  # good fit
python tmp/forward_noise_covariance_learning.py --credit pooled    --num-samples 100  # underfits (ablation)
```

Perturbation-gate knobs (`cov_deriv_gate`, §8.1):

```bash
python tmp/forward_noise_covariance_learning.py --gate-block-size 8 --gate-alpha 0.05 --gate-mode cyclic
python tmp/forward_noise_covariance_learning.py --gate-alpha 0.3 --num-samples 128   # higher SNR / lower-variance credit
```

`cov_deriv_gate_crn` (§8.3) reuses the same `--gate-*` knobs; the antithetic pair doubles
the forward count per credit pass. It is a deliberately kept negative result — it does **not**
beat `cov_deriv_gate` on the binary crossing (see §8.3).

The structured/recursive `cov_jac` (§8.4) uses `--jac-ema` (EMA rate for the weight mirrors,
default 0.9) and `--jac-track` (Kolen–Pollack mirror tracking + input-pooling, **on by default**;
`--no-jac-track` to disable). The two `cov_jac` rows differ only in the local optimiser:
`cov_jac_sgd` shows the SGD "floor" (~0.03), and `cov_jac_adam` — Adam being a local, still
weight-transport-free per-weight rule — reaches **backprop-level** final error (~0.0011, §8.4):

```bash
python tmp/forward_noise_covariance_learning.py --epochs 1500 --num-samples 64 --fit-check  # cov_jac_adam ~ backprop
python tmp/forward_noise_covariance_learning.py --jac-ema 0.98                               # smoother, slower-tracking mirror
```

The noise-field / recruitment gate (`cov_deriv_field_gate`, §8.5) uses `--field-sparsity`
(default 0.0). At 0 it equals `cov_deriv`; a positive value makes that fraction of hidden units
un-recruited (zero forward noise **and** zero update) on the shared network:

```bash
python tmp/forward_noise_covariance_learning.py --field-sparsity 0.3   # 30% of hidden units un-recruited (dead)
```

The distribution-free slope (§8.2) is the **default** for `cov_deriv` (`--slope kde`); it is
most striking under `--noise uniform`, where it matches the analytic-`phi'` ablation
**without** using the uniform `-(d-c)/r^2` formula. Use `--slope analytic` to force the
hand-coded `phi'(d)` instead:

```bash
python tmp/forward_noise_covariance_learning.py --noise uniform          # cov_deriv uses kde slope (default)
python tmp/forward_noise_covariance_learning.py --slope analytic         # force hand-coded phi'(d)
```

CPU-friendly defaults: `noise=gaussian`, `credit=per_input`, `credit-passes=1`,
`opt=sgd`, `lr-decay=none`, `epochs=1500`, `hidden-dim=64`, `num-samples=64` (the
model's internal `t`), `lr=1e-2`, `sigma=0.5` (Gaussian std), `radius=1.0` (uniform
half-width), `crossing-h=0.2`, `seed=0`, `device=cpu`, `hidden-lr-scale=1.0`,
`slope=kde` (distribution-free crossing slope; `analytic` forces `phi'(d)`),
`gate-block-size=8`, `gate-alpha=0.05`, `gate-mode=cyclic`, `jac-ema=0.9`,
`jac-track=True` (Kolen–Pollack tracking + pooling on by default; `--no-jac-track` to disable),
`field-sparsity=0.0` (all units recruited; `>0` makes that fraction of hidden units
un-recruited — see §8.5), `fit-check=off`, `include-gates=off`, `save=None`. (For the *scalar*
`cov_deriv` credit, `--opt adam` / `--lr-decay cosine` do **not** lower its floor — see §10; but
for `cov_jac`'s near-exact gradient, `--opt adam` **is** what reaches backprop level — §8.4.)
A fast smoke test:

```bash
python tmp/forward_noise_covariance_learning.py --epochs 50 --num-samples 16 --hidden-dim 16
```

## 13. Figures

The script displays three figures with `plt.show()` (four with `--fit-check`; use `--save DIR`
to write them as PNG instead):

```
learning curves        # the verification methods (backprop, cov_only, cov_deriv_{analytic,kde}, cov_jac_{sgd,adam}[, gates])
predictions on sin(x)  # target and each method's prediction
cov_deriv layer-1 stats# mean activity, g_z, and phi'(d) per hidden unit (from cov_deriv_kde)
fit check (--fit-check)# focused: target vs backprop vs cov_jac_adam + residuals (tight-fit confirmation)
```

The console also prints the final MSE of each method and a short interpretation: whether
`cov_deriv_kde` **matches** `cov_deriv_analytic` (so the analytic `phi'` can be dropped), whether
**`cov_jac_adam` reaches backprop level** (the headline result), and whether `cov_jac_adam` beats
`cov_jac_sgd` (confirming the SGD floor is optimisation, not estimator bias). With
`--include-gates` it additionally reports the gate/field comparisons (§8.1, §8.3, §8.5).
