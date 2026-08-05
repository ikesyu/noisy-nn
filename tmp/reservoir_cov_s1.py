"""S1 (docs/idea_reservoir.md §14.6): does a SECOND crossing layer respond to the
noise CORRELATION of layer 1, while layer-1 marginals stay blind?

Mechanism under test (§14.0): layer-1 outputs z1 are random; their correlation
enters layer 2 through the variance of its pre-activation,
    Var(sum_j W_j z1_j) = sum_j W_j^2 Var(z1_j) + sum_{j!=k} W_j W_k Cov(z1_j, z1_k),
so a field-modulated rho must change layer-2 crossing rates ("correlation seen as
an effective-sigma modulation") even though every layer-1 marginal is unchanged.

Layer 1: H units, fixed operating points d, noise eta with marginal sigma FIXED and
shared:private split set by rho (so marginals are identical across rho -- §14.1).
Layer 2: crossing on W2 @ z1 (+ own noise), evaluated by Monte-Carlo over numT.

PASS if: layer-1 marginal crossing rates are flat in rho, layer-2 rates are NOT,
and the layer-2 response grows with |W2 correlation alignment| (coherent weights
respond most). Also check a linear read-out of layer-2 can recover rho(t).
"""
import numpy as np
from scipy.special import ndtr


def crossings(x, h):
    """Crossing events along axis 0 (numT samples): |b_{i+1}-b_i| at +/-h, avg."""
    b1 = (x > h).astype(float); b2 = (x > -h).astype(float)
    c1 = np.abs(np.roll(b1, -1, axis=0) - b1)
    c2 = np.abs(np.roll(b2, -1, axis=0) - b2)
    return 0.5 * (c1 + c2)


def layer1_noise(numT, H, rho, sigma, rng):
    """Correlated noise with FIXED marginal sigma: shared:private split by rho.
    eta_k = sqrt(rho)*sigma*xi + sqrt(1-rho)*sigma*eps_k  -> Corr = rho (rho>=0)."""
    xi = rng.standard_normal((numT, 1))
    eps = rng.standard_normal((numT, H))
    return sigma * (np.sqrt(rho) * xi + np.sqrt(1.0 - rho) * eps)


def run(rho, H=32, numT=200000, sigma=1.0, h=0.2, seed=0, d_val=0.5,
        W2_mode="coherent", h2=0.2, d2=0.0, sigma2=0.15):
    rng = np.random.default_rng(seed)
    d = np.full(H, d_val)
    eta = layer1_noise(numT, H, rho, sigma, rng)
    x1 = d[None, :] + eta
    c1 = crossings(x1, h)                          # [numT, H] layer-1 events
    marg = c1.mean(0)                              # layer-1 marginal rates

    # layer 2: pre-activation W2 @ z1 (+ its own small private noise)
    if W2_mode == "coherent":                      # all-positive: sums correlations
        W2 = np.ones((1, H)) / np.sqrt(H)
    elif W2_mode == "balanced":                    # +/- split: cancels common mode
        s = np.ones(H); s[::2] = -1.0
        W2 = (s / np.sqrt(H))[None, :]
    else:
        W2 = rng.standard_normal((1, H)) / np.sqrt(H)
    pre2 = c1 @ W2.T + d2 + sigma2 * rng.standard_normal((numT, 1))
    c2 = crossings(pre2, h2)
    return marg.mean(), marg.std(), float(c2.mean()), float(pre2.std())


def main():
    print("=== S1: layer-1 marginals vs layer-2 response to rho ===")
    print(f"{'rho':>5} {'L1 marg(mean)':>14} {'L1 marg(sd)':>12} "
          f"{'L2 coherent':>12} {'L2 balanced':>12} {'pre2 sd(coh)':>13}")
    rows = []
    for rho in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9):
        m, msd, c2c, p2c = run(rho, W2_mode="coherent")
        _, _, c2b, _ = run(rho, W2_mode="balanced")
        rows.append((rho, m, c2c, c2b))
        print(f"{rho:5.1f} {m:14.4f} {msd:12.4f} {c2c:12.4f} {c2b:12.4f} {p2c:13.4f}")

    m0, mE = rows[0][1], rows[-1][1]
    c0, cE = rows[0][2], rows[-1][2]
    b0, bE = rows[0][3], rows[-1][3]
    print(f"\n  layer-1 marginal change over rho 0->0.9 : {mE - m0:+.4f}")
    print(f"  layer-2 (coherent W2) change            : {cE - c0:+.4f}")
    print(f"  layer-2 (balanced W2) change            : {bE - b0:+.4f}")

    # (2) can a linear read-out of layer-2 recover a time-varying rho(t)?
    print("\n=== (2) linear read-out of layer-2 recovering rho(t) ===")
    T, nT, H = 600, 20000, 32
    rho_t = 0.45 * (1 + np.sin(np.linspace(0, 12 * np.pi, T)))   # in [0, 0.9]
    F1 = np.zeros((T, H)); F2 = np.zeros((T, 3))
    rng = np.random.default_rng(1)
    d = np.full(H, 0.5)
    W2s = [np.ones((1, H)) / np.sqrt(H)]
    s = np.ones(H); s[::2] = -1.0
    W2s.append((s / np.sqrt(H))[None, :])
    W2s.append(rng.standard_normal((1, H)) / np.sqrt(H))
    for t in range(T):
        eta = layer1_noise(nT, H, float(rho_t[t]), 1.0, rng)
        c1 = crossings(d[None, :] + eta, 0.2)
        F1[t] = c1.mean(0)
        for i, W2 in enumerate(W2s):
            pre2 = c1 @ W2.T + 0.15 * rng.standard_normal((nT, 1))
            F2[t, i] = crossings(pre2, 0.2).mean()

    def r2(X, y):
        Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
        tr, te = slice(0, 400), slice(400, T)
        W = np.linalg.lstsq(Xb[tr], y[tr], rcond=None)[0]
        yh = Xb @ W
        return 1 - ((y[te] - yh[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    print(f"  R^2(rho | layer-1 marginals, H={H}) = {r2(F1, rho_t):.3f}")
    print(f"  R^2(rho | layer-2 (3 units))        = {r2(F2, rho_t):.3f}")
    print(f"  R^2(rho | layer-2 coherent only)    = {r2(F2[:, :1], rho_t):.3f}")


if __name__ == "__main__":
    main()
