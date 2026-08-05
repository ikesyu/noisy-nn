"""論点1 smoke (docs/idea_reservoir.md): does adding crossing LAYERS (learned
forward-only, no BPTT) buy expressiveness over a single crossing layer and over a
linear read-out reservoir?  Contrast with standard RC (fixed reservoir + LINEAR
read-out): our claim is "fixed field + forward-only DEEP nonlinear map + linear
read-out", a position between RC and RNN.

Here we FIRST scope which task shows a depth effect: compare, on the SAME fixed
LDN field, feature sets read out by ridge:
  L0  : linear read-out of the field states A(t)        (= standard-RC-style, depth 0)
  L1  : + one crossing layer z1 = crossing(W1 A)        (single nonlinear layer)
  L2  : + two crossing layers z2 = crossing(W2 z1)      (deep nonlinear map)
with W1,W2 RANDOM here (random-feature scoping; forward-only LEARNING is the next
step). A depth effect shows as L2 < L1 < L0 on a task whose target needs composed
nonlinearity. Candidate tasks: NARMA-20, and a COMPOSED nonlinear task
  y = tanh(3*( (u_{t-2}+u_{t-8})^2 - (u_{t-5})^2 ) )      (needs square-of-sum etc.)
and a delayed-XOR-like parity proxy.  Random features only scope; if a depth gap
appears even with random layers, learning should widen it.
"""
import numpy as np

import reservoir as R
from reservoir.readout import ridge_fit, standardize_fit, nrmse


def masks(T, wo=300, fr=0.7):
    idx = np.arange(wo, T); n = int(len(idx) * fr)
    tr = np.zeros(T, bool); tr[idx[:n]] = True
    te = np.zeros(T, bool); te[idx[n:]] = True
    return tr, te


def crossing(x, h=0.2):
    b1 = (x > h).astype(float); b2 = (x > -h).astype(float)
    return 0.5 * (np.abs(np.roll(b1, -1, 0) - b1) + np.abs(np.roll(b2, -1, 0) - b2))


def ridge_nrmse(X, y, tr, te, alpha=1e-2):
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    mu, sd = standardize_fit(Xb[tr]); Xs = (Xb - mu) / sd
    W = ridge_fit(Xs[tr], y[tr], alpha)
    return nrmse(y[te], (Xs @ W)[te])


def layers(A, H=48, seed=0, sigma=0.0):
    """Random-feature crossing layers on the field states A (analytic-ish: no
    per-unit noise here, deterministic crossings of projections)."""
    rng = np.random.default_rng(seed)
    As = (A - A.mean(0)) / (A.std(0) + 1e-8)
    W1 = rng.standard_normal((H, A.shape[1])) / np.sqrt(A.shape[1])
    b1 = rng.uniform(-1, 1, H)
    z1 = crossing(As @ W1.T + b1)
    W2 = rng.standard_normal((H, H)) / np.sqrt(H)
    b2 = rng.uniform(-1, 1, H)
    z2 = crossing(z1 @ W2.T + b2)
    return As, z1, z2


def composed_task(T, seed=0):
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    lag = lambda L: np.concatenate([np.zeros(L), u[:T - L]])
    y = np.tanh(3 * ((lag(2) + lag(8)) ** 2 - lag(5) ** 2))
    return u, y - y.mean()


def parity_task(T, seed=0):
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    lag = lambda L: np.concatenate([np.zeros(L), u[:T - L]])
    y = np.sign(lag(2)) * np.sign(lag(6)) * np.sign(lag(10))     # 3-way sign parity
    return u, y - y.mean()


def run(name, u, y, seeds=4):
    T = len(u); tr, te = masks(T)
    A = R.LDNField(H=48, theta=60.0).run(u)
    e = {"L0": [], "L1": [], "L2": [], "L1+2": []}
    for s in range(seeds):
        As, z1, z2 = layers(A, seed=s)
        e["L0"].append(ridge_nrmse(As, y, tr, te))
        e["L1"].append(ridge_nrmse(z1, y, tr, te))
        e["L2"].append(ridge_nrmse(z2, y, tr, te))
        e["L1+2"].append(ridge_nrmse(np.concatenate([z1, z2], 1), y, tr, te))
    print(f"\n=== {name} ===")
    for k in ("L0", "L1", "L2", "L1+2"):
        print(f"  {k:5s} (depth {'0' if k=='L0' else k[1]}): NRMSE={np.mean(e[k]):.3f} "
              f"(+/-{np.std(e[k]):.3f})")


def main():
    T = 3000
    print("論点1 scoping: does crossing DEPTH help (random features, ridge read-out)?")
    run("NARMA-20", *R.narma_x(T, 20, seed=0))
    run("composed nonlinear  y=tanh((u2+u8)^2 - u5^2)", *composed_task(T))
    run("3-way sign parity  y=sign(u2)sign(u6)sign(u10)", *parity_task(T))


if __name__ == "__main__":
    main()
