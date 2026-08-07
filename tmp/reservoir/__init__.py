"""Noise-field reservoir — clean core library (docs/idea_reservoir.md).

The separated architecture (Ours): a dissipative NOISE FIELD carries MEMORY, an
NNN crossing map carries NONLINEARITY (learned forward-only, no BPTT). This beats
a standard ESN (which couples the two in one tanh reservoir) on NARMA, because
separation lets you pick an optimal / biologically-grounded memory field.

Two couplings of the field to the NNN (docs §10.17-10.19):
    NoiseModulatedMap (B, on-brand): the field is the NNN's ADDITIVE per-unit
        NOISE sigma_k(t); the NNN input is a fixed operating point d_k. Faithful
        to "noise-modulated". Two wirings of field->noise:
          mix=False (diagonal): unit k reads field coord k; a pointwise (GAM)
            noise-modulation, so it needs the field's coords to pre-mix lags
            (LDN/time cells beat ESN; a single-lag delay field fails).
          mix=True (mixed): unit k reads a LEARNED combination of all coords, so
            the noise MAP does the lag-mixing — a pure-memory delay field then
            works and (B) reaches (A) (docs §10.19). This is the cleanest form of
            the separation: memory in the field, nonlinearity+mixing in the NNN.
    LearnedCrossingMap (A): the field STATE is the crossing INPUT (fixed noise).
        Stronger than diagonal (B) on NARMA, but then field==reservoir (a
        different logic); mixed (B) matches it without giving up the noise story.

Modules:
    fields   : noise-field designs (delay line, cascade, damped-orthogonal,
               Legendre/LDN = time cells, diffusion, random) + pulse_decay
    nnn_map  : NoiseModulatedMap (B), LearnedCrossingMap (A) — forward-only maps
    esn      : LeakyESN baseline
    tasks    : narma_x, mc_input
    readout  : ridge, standardisation, splits, scores
    metrics  : memory_capacity, task_nrmse
    moment   : shared harness for the moment-order (threshold/crossing/lambda)
               activation comparisons of §10.24-10.36/§13.1 (torch; not imported
               here to keep `import reservoir` numpy-only)

Side branches from the exploration (recurrent crossing reservoir, cov_jac /
forward-noise variants, functional / super-resolution, output feedback, …) are
kept only as reference scripts in tmp/ (reservoir_*.py), not in this library.
"""
from .fields import (LinearReservoir, DelayLineField, CascadeField,
                     DampedOrthField, LDNField, DiffusionField, pulse_decay)
from .nnn_map import LearnedCrossingMap, NoiseModulatedMap
from .esn import LeakyESN
from .tasks import narma_x, mc_input
from .readout import (ridge_fit, ridge_predict, standardize_fit, split_washout,
                      nrmse, corr2, r2_score)
from .metrics import memory_capacity, task_nrmse
