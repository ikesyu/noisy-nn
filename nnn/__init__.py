"""nnn — Noise-modulated Neural Network (NNN) library.

Modules:
    activation  : crossing activation functions (autograd.Function)
    layer       : noise injection, sampling, readout, and crossing layers
    model       : simple feedforward NNN models
    noise       : noise samplers and distribution functions (PDF / CDF)
    noise_field : spatial noise-field (recruitment) utilities
    stats       : forward-pass capture and local slopes (KDE / analytic)
    credit      : credit estimators from forward noise (weight mirror, optimiser)
"""
