"""S2 (oracle viability, docs §14.6): CAN the covariance mechanism REPRESENT the
gated product at all? Learning M/W_L is S3; here we bypass learning and hand the
model the IDEAL marginal encodings and the IDEAL correlation, then ask whether the
layer-2 co-fluctuation recovers y = gate(u_{t-40}) u_{t-5} u_{t-15}, while a
scale-only (marginals-only) read-out cannot.

Design (oracle):
  u5-encoders : units whose marginal noise sigma encodes u(t-5): sigma=floor+softplus(g*u5)
  u15-encoders: likewise for u(t-15)
  shared mode : loads sqrt(clip(gate(u_{t-40}),0,cap)) on ALL units -> every
                u5-u15 pair gets correlation ~ gate(u_{t-40}) (context-gated).
  layer-1 marginals z_k = crossing(d_k + sigma_k * (l*xi + sqrt(1-l^2) eps)).
  layer-2 z2 = crossing(W2 z1 + b2 + priv) reads the co-fluctuation.
Compare linear read-out on {z1} (scale-only / marginals) vs {z1,z2} (covariance).
PASS if {z1,z2} solves the gated task and {z1} does not -> mechanism is viable.
Control: correlation set to CONSTANT (no context gating) -> should NOT solve it.
"""
import numpy as np


def masks(T, wo=300, fr=0.7):
    idx = np.arange(wo, T); n = int(len(idx) * fr)
    tr = np.zeros(T, bool); tr[idx[:n]] = True
    te = np.zeros(T, bool); te[idx[n:]] = True
    return tr, te


def crossings(x, h=0.2):
    b1 = (x > h).astype(float); b2 = (x > -h).astype(float)
    return 0.5 * (np.abs(np.roll(b1, -1, 0) - b1) + np.abs(np.roll(b2, -1, 0) - b2))


def ridge_nrmse(X, y, tr, te, alpha=1e-2):
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    mu = Xb[tr].mean(0); sd = Xb[tr].std(0) + 1e-8
    Xs = (Xb - mu) / sd
    W = np.linalg.solve(Xs[tr].T @ Xs[tr] + alpha * np.eye(Xs.shape[1]), Xs[tr].T @ y[tr])
    yh = Xs @ W
    return np.sqrt(((y[te] - yh[te]) ** 2).mean() / (y[te].var() + 1e-12))


def build(T=4000, floor=0.3, H2=24, numT=64, corr_cap=0.85, gate_mode="context",
          seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, T)
    lag = lambda L: np.concatenate([np.zeros(L), u[:T - L]])
    u5, u15, u40 = lag(5), lag(15), lag(40)
    gate = 0.5 * (1 + np.tanh(4 * u40))
    y = gate * u5 * u15; y = y - y.mean()

    # oracle marginal encoders: half encode u5, half u15 (varied gains)
    H1 = 16
    kind = np.array([0] * (H1 // 2) + [1] * (H1 // 2))       # 0=u5, 1=u15
    gains = rng.uniform(1.0, 3.0, H1) * rng.choice([-1, 1], H1)
    d1 = rng.uniform(-1.2, 1.2, H1)
    src = np.where(kind == 0, 1.0, 0.0)[:, None] * u5[None, :] + \
          np.where(kind == 1, 1.0, 0.0)[:, None] * u15[None, :]   # [H1,T]
    sigma = floor + np.log1p(np.exp(np.clip(gains[:, None] * src, -30, 30)))  # [H1,T]

    # oracle correlation: shared-mode load l(t) per unit
    if gate_mode == "context":
        l_shared = np.sqrt(np.clip(gate, 0, corr_cap))          # [T] gated by u40
    else:  # constant control
        l_shared = np.full(T, np.sqrt(0.5 * corr_cap))
    l = np.broadcast_to(l_shared, (H1, T))                      # every unit loads equally

    # layer-1 sample crossings (marginal preserved; correlation via shared xi)
    W2 = rng.standard_normal((H2, H1)) / np.sqrt(H1)
    b2 = rng.uniform(-0.4, 0.4, H2)
    z1 = np.empty((T, H1)); z2 = np.empty((T, H2))
    for t in range(T):
        xi = rng.standard_normal((numT, 1))
        eps = rng.standard_normal((numT, H1))
        unit = xi * l[:, t][None, :] + np.sqrt(1 - l[:, t] ** 2)[None, :] * eps
        eta = sigma[:, t][None, :] * unit
        c1 = crossings(d1[None, :] + eta)
        z1[t] = c1.mean(0)
        pre2 = c1 @ W2.T + b2 + 0.3 * rng.standard_normal((numT, H2))
        z2[t] = crossings(pre2).mean(0)
    return u, y, z1, z2


def main():
    T = 4000
    print("=== S2 oracle viability: gated product via correlation ===")
    u, y, z1, z2 = build(T, gate_mode="context", seed=0)
    tr, te = masks(T)
    e_marg = ridge_nrmse(z1, y, tr, te)                         # scale-only (marginals)
    e_cov = ridge_nrmse(np.concatenate([z1, z2], 1), y, tr, te)  # + layer-2 (co-fluct)
    print(f"  marginals only  (scale-only)      NRMSE={e_marg:.3f}")
    print(f"  marginals + layer-2 (covariance)  NRMSE={e_cov:.3f}")

    # control: constant (non-gated) correlation -> co-fluctuation carries no context
    _, yc, z1c, z2c = build(T, gate_mode="const", seed=0)
    e_cov_const = ridge_nrmse(np.concatenate([z1c, z2c], 1), yc, tr, te)
    print(f"  [control] constant-corr covariance NRMSE={e_cov_const:.3f} "
          f"(should NOT solve the gated task)")

    verdict = (e_cov < e_marg - 0.05) and (e_cov < e_cov_const - 0.05)
    print(f"\n  -> mechanism {'VIABLE' if verdict else 'NOT clearly viable'}: "
          f"context-gated correlation {'recovers' if e_cov < 0.6 else 'does not recover'} "
          f"the gated product that marginals cannot.")


if __name__ == "__main__":
    main()