"""Smoke test for covariance modulation (docs §9.x follow-up): is the co-crossing
(coincidence) feature actually sensitive to the noise COVARIANCE across units,
while the marginal crossing is blind to it?

The mean-field crossing z_k = 2Phi(d/sigma)(1-Phi) depends ONLY on the marginal
sigma_k -> it cannot see correlations. For covariance modulation to do anything,
a JOINT feature must respond to Cov(eta_j, eta_k). We test the natural NNN joint
feature: the CO-CROSSING rate (both units cross between the same consecutive
sample pair). Two checks:
  (1) sweep the cross-unit correlation rho: co-crossing must vary with rho while
      the marginal crossing rates stay fixed (covariance sensitivity);
  (2) field-modulated readout: let rho(t) be a signal; can a linear read-out of
      the co-crossing feature recover rho(t) where the marginals cannot?
"""
import numpy as np
from scipy.special import ndtr


def crossings(x, h):
    """Crossing events along axis 0 (numT samples): |b_{i+1}-b_i| at +/-h, avg.
    x: [numT, U]. Returns events [numT, U] in {0, 0.5, 1}."""
    b1 = (x > h).astype(float); b2 = (x > -h).astype(float)
    c1 = np.abs(np.roll(b1, -1, axis=0) - b1)
    c2 = np.abs(np.roll(b2, -1, axis=0) - b2)
    return 0.5 * (c1 + c2)


def joint_noise(numT, rho, sigma, rng):
    """Two units, cross-unit correlation rho, marginal std sigma, per-sample iid."""
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    return (rng.standard_normal((numT, 2)) @ L.T) * sigma


def main():
    rng = np.random.default_rng(0)
    numT, h, sigma = 400000, 0.2, 1.0
    d = np.array([0.5, 0.5])

    # (1) co-crossing vs rho; marginals must stay flat
    print("=== (1) co-crossing sensitivity to cross-unit correlation rho ===")
    print(f"{'rho':>6} {'cross_j':>9} {'cross_k':>9} {'co-cross':>9} "
          f"{'indep-prod':>11} {'excess':>8}")
    for rho in (-0.9, -0.5, 0.0, 0.5, 0.9):
        eta = joint_noise(numT, rho, sigma, rng)
        x = d[None, :] + eta
        c = crossings(x, h)                       # [numT, 2]
        cj, ck = c[:, 0].mean(), c[:, 1].mean()
        co = (c[:, 0] * c[:, 1]).mean()
        indep = cj * ck
        print(f"{rho:6.1f} {cj:9.4f} {ck:9.4f} {co:9.4f} {indep:11.4f} "
              f"{co - indep:+8.4f}")

    # (2) field-modulated readout: recover rho(t) from features
    print("\n=== (2) can a linear read-out recover rho(t)? ===")
    T, nT = 3000, 4000
    rho_t = 0.9 * np.sin(np.linspace(0, 20 * np.pi, T))     # the field-set signal
    feats_marg = np.zeros((T, 2)); feats_co = np.zeros((T, 1))
    for t in range(T):
        eta = joint_noise(nT, float(rho_t[t]), sigma, rng)
        x = d[None, :] + eta
        c = crossings(x, h)
        feats_marg[t] = c.mean(0)
        feats_co[t, 0] = (c[:, 0] * c[:, 1]).mean()

    def r2(X, y):
        Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
        tr = slice(0, 2000); te = slice(2000, T)
        W = np.linalg.lstsq(Xb[tr], y[tr], rcond=None)[0]
        yh = Xb @ W
        ss = ((y[te] - yh[te]) ** 2).sum(); tot = ((y[te] - y[te].mean()) ** 2).sum()
        return 1 - ss / tot
    print(f"  R^2(rho | marginals only)   = {r2(feats_marg, rho_t):.3f}")
    print(f"  R^2(rho | co-crossing only) = {r2(feats_co, rho_t):.3f}")
    print(f"  R^2(rho | marg + co)        = {r2(np.concatenate([feats_marg, feats_co],1), rho_t):.3f}")


if __name__ == "__main__":
    main()
