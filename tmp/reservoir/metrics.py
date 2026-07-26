"""Reservoir evaluation metrics: memory capacity and NRMSE task error."""
import numpy as np

from .readout import (ridge_fit, ridge_predict, split_washout, corr2,
                      standardize_fit, nrmse)


def memory_capacity(X, u, max_delay=40, washout=200, train_frac=0.7,
                    alpha=1e-3):
    """Total linear memory capacity MC = sum_k corr^2(u[t-k], read-out).

    Returns (mc_per_delay [max_delay], total_MC). Features are standardised on
    the training rows. Each delay k is fit independently (standard MC protocol).
    """
    n = X.shape[0]
    tr, te = split_washout(n, washout, train_frac)
    mu, sd = standardize_fit(X[tr])
    Xs = (X - mu) / sd

    mcs = np.zeros(max_delay)
    for k in range(1, max_delay + 1):
        target = np.full(n, np.nan)
        target[k:] = u[:-k]
        trk = tr[tr >= k]
        tek = te[te >= k]
        W = ridge_fit(Xs[trk], target[trk], alpha)
        yhat = ridge_predict(Xs[tek], W)
        mcs[k - 1] = corr2(target[tek], yhat)
    return mcs, float(mcs.sum())


def task_nrmse(X, y, washout=200, train_frac=0.7, alpha=1e-3):
    """Fit a linear read-out for a scalar target y(t); return test NRMSE.

    A bias column (intercept) is appended — otherwise the mean-0 standardised
    features cannot fit a target with a nonzero mean and NRMSE blows up."""
    n = X.shape[0]
    tr, te = split_washout(n, washout, train_frac)
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)      # intercept
    mu, sd = standardize_fit(Xb[tr])                       # keeps the const col
    Xs = (Xb - mu) / sd
    W = ridge_fit(Xs[tr], y[tr], alpha)
    return nrmse(y[te], ridge_predict(Xs[te], W))
