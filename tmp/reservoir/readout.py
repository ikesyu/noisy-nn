"""Linear read-out (ridge regression) with washout and train/test split.

Only the read-out is trained (RC discipline): closed-form ridge, no backprop.
Standardising features before the fit keeps ill-conditioned crossing features in
check without changing the linear span.
"""
import numpy as np


def standardize_fit(X):
    """Column standardisation that preserves constant columns (e.g. a bias
    column of ones): a near-constant column keeps mu=0, sd=1 so it survives as
    an intercept instead of being zeroed out."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    const = sd < 1e-8
    sd = np.where(const, 1.0, sd)
    mu = np.where(const, 0.0, mu)
    return mu, sd


def ridge_fit(X, Y, alpha=1e-3):
    F = X.shape[1]
    A = X.T @ X + alpha * np.eye(F)
    B = X.T @ Y
    return np.linalg.solve(A, B)


def ridge_predict(X, W):
    return X @ W


def split_washout(n, washout, train_frac=0.7):
    idx = np.arange(washout, n)
    ntr = int(len(idx) * train_frac)
    return idx[:ntr], idx[ntr:]


def corr2(y, yhat):
    """Squared Pearson correlation (used for memory capacity)."""
    y, yhat = np.ravel(y), np.ravel(yhat)
    if np.std(y) < 1e-12 or np.std(yhat) < 1e-12:
        return 0.0
    c = np.corrcoef(y, yhat)[0, 1]
    return float(c * c) if np.isfinite(c) else 0.0


def r2_score(y, yhat):
    y, yhat = np.ravel(y), np.ravel(yhat)
    ss = np.sum((y - yhat) ** 2)
    tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    return float(1.0 - ss / tot)


def nrmse(y, yhat):
    y, yhat = np.ravel(y), np.ravel(yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2) / (np.var(y) + 1e-12)))
