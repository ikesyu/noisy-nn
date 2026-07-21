"""tmp/rl -- shared modules for the NNN-RL forward-fluctuation experiments (idea_rl.md §20).

Stage 1 verifies that covariance-eligibility credit, generated ENTIRELY inside the NNN
forward fluctuation path (no transposed-weight backward, no external RL algorithm), can
learn CartPole and beats node perturbation on credit variance.

Reuses the validated credit engine in `data_nce/fncl/` (Capture, kde_slope, cov_weight).
"""
