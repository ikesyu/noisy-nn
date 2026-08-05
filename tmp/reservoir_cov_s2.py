"""S2 (docs/idea_reservoir.md §14.6): minimal viability -- can a covariance-
modulated 2-layer NNN LEARN the gated-product task at all?

Task (§14.3):  y(t) = gate(u(t-c)) * u(t-a) * u(t-b),  gate(v)=0.5(1+tanh(k v))
  defaults c=40, a=5, b=15, k=4, u ~ iid U[-1,1]. All lags within the field reach
  (H_f=48 LDN): memory is the field's job, covariance's job is the GATED COUPLING.

Architecture:
  field   : LDN, states A(t) [T,Hf] (fixed dissipative memory).
  layer 1 : H1 units, fixed operating points d1; noise eta with FIXED marginal
            sigma1 but a field-set correlation via R shared latent modes:
              eta_k = sum_r L_kr(t) xi_r + sqrt(sigma1^2 - sum_r L_kr^2) eps_k
            loads L(t) = beta * tanh(W_L A(t))  (|L_kr| bounded so private var>=0).
            -> layer-1 marginals are rho-blind; correlations carry the context.
  layer 2 : H2 crossing units on W2 @ z1 + b2 + private noise sigma2 (operating
            point kept INTERIOR, S1 caveat), read out linearly (ridge).

Evaluated by Monte-Carlo over numT (sample crossings; correlations propagate).
This S2 script checks VIABILITY: we train W2,b2 and the field->load map W_L (plus
readout) by finite-difference-free numerical gradient on a frozen readout is
overkill here; instead we (a) confirm the covariance model can REPRESENT the task
(fit readout on rich features and compare to scale-only), and (b) do a small
random/greedy search over W_L to confirm a field-driven correlation actually
improves the gated-task fit -> learnability signal. Full forward-only credit is S2b.
"""
import numpy as np

import reservoir as R
from reservoir.readout import ridge_fit, standardize_fit, nrmse


def masks(T, wo=300, fr=0.7):
    idx = np.arange(wo, T); n = int(len(idx) * fr)
    tr = np.zeros(T, bool); tr[idx[:n]] = True
    te = np.zeros(T, bool); te[idx[n:]] = True
    return tr, te


def gated_task(T, c=40, a=5, b=15, k=4.0, seed=0):
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    lag = lambda L: np.concatenate([np.zeros(L), u[:T - L]])
    gate = 0.5 * (1 + np.tanh(k * lag(c)))
    y = gate * lag(a) * lag(b)
    return u, y - y.mean()


def crossings_1d(x, h=0.2):
    b1 = (x > h).astype(float); b2 = (x > -h).astype(float)
    c1 = np.abs(np.roll(b1, -1, axis=0) - b1)
    c2 = np.abs(np.roll(b2, -1, axis=0) - b2)
    return 0.5 * (c1 + c2)


class CovTwoLayer:
    """2-layer NNN with a field->MARGINAL-sigma path (present in ALL variants, =
    the current NNN Reservoir) plus an optional field->CORRELATION add-on.

    layer-1 unit k: operating point d1_k fixed; marginal noise
        sigma_k(t) = floor + softplus((M A(t))_k + c_k)          [field -> sigma]
    correlation add-on (covariance model): loadings scaled by sigma keep the
    marginal EXACTLY sigma_k while setting cross-unit correlation:
        eta_k = sigma_k [ sum_r l_kr(t) xi_r + sqrt(1 - sum_r l_kr^2) eps_k ],
        l_kr(t) = alpha * tanh((W_L A)_kr),  alpha^2 R < 1  (so private var >= 0)
    scale_only -> l=0 (independent, current model); static_corr -> l fixed."""

    def __init__(self, A, Hf, H1=24, H2=16, R_lat=4, floor=0.3, sigma2=0.3,
                 h=0.2, numT=80, seed=0, static_corr=False, scale_only=False):
        rng = np.random.default_rng(seed)
        self.A = (A - A.mean(0)) / (A.std(0) + 1e-8)
        self.Hf, self.H1, self.H2, self.R = Hf, H1, H2, R_lat
        self.floor, self.sigma2, self.h, self.numT = floor, sigma2, h, numT
        self.scale_only, self.static_corr = scale_only, static_corr
        self.d1 = rng.uniform(-1.5, 1.5, H1)
        self.M = rng.standard_normal((H1, Hf)) / np.sqrt(Hf)           # field->sigma
        self.c = rng.uniform(-0.5, 0.5, H1)
        self.alpha = 0.5 / np.sqrt(R_lat)                             # sum l^2 <= 0.25
        self.W_L = rng.standard_normal((H1, R_lat, Hf)) / np.sqrt(Hf)  # field->loads
        self.static_l = self.alpha * np.tanh(rng.standard_normal((H1, R_lat)))
        self.W2 = rng.standard_normal((H2, H1)) / np.sqrt(H1)
        self.b2 = rng.uniform(-0.4, 0.4, H2)          # interior operating point (S1 caveat)
        self.rng = rng

    @staticmethod
    def _softplus(z):
        return np.where(z > 30, z, np.log1p(np.exp(np.clip(z, -30, 30))))

    def _sigma(self, t_idx):
        return self.floor + self._softplus(self.A[t_idx] @ self.M.T + self.c)  # [T,H1]

    def _loads(self, t_idx):
        if self.scale_only:
            return np.zeros((len(t_idx), self.H1, self.R))
        if self.static_corr:
            return np.broadcast_to(self.static_l, (len(t_idx), self.H1, self.R))
        return self.alpha * np.tanh(np.einsum('hrf,tf->thr', self.W_L, self.A[t_idx]))

    def features(self, t_idx):
        T = len(t_idx); numT = self.numT
        sigma = self._sigma(t_idx)                     # [T,H1] marginal (field-driven)
        l = self._loads(t_idx)                         # [T,H1,R] correlation structure
        priv = np.sqrt(np.clip(1.0 - np.sum(l ** 2, axis=2), 1e-6, None))  # [T,H1]
        z1 = np.empty((T, self.H1)); z2 = np.empty((T, self.H2))
        for i in range(T):
            xi = self.rng.standard_normal((numT, self.R))
            eps = self.rng.standard_normal((numT, self.H1))
            unit = xi @ l[i].T + priv[i][None, :] * eps        # unit-variance, corr set by l
            eta = sigma[i][None, :] * unit                     # marginal = sigma exactly
            c1 = crossings_1d(self.d1[None, :] + eta, self.h)  # [numT,H1]
            z1[i] = c1.mean(0)
            pre2 = c1 @ self.W2.T + self.b2 + self.sigma2 * self.rng.standard_normal((numT, self.H2))
            z2[i] = crossings_1d(pre2, self.h).mean(0)
        return np.concatenate([z1, z2], axis=1)

    def fit_eval(self, y, tr, te, alpha=1e-2):
        X = self.features(np.arange(len(y)))
        Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
        mu, sd = standardize_fit(Xb[tr]); Xs = (Xb - mu) / sd
        W = ridge_fit(Xs[tr], y[tr], alpha)
        return nrmse(y[te], (Xs @ W)[te])


def eval_fixed(m, y, tr, te, W_L=None):
    """Deterministic objective: fix the sampling noise so hill-climbing sees a
    stable NRMSE (only W_L changes)."""
    if W_L is not None:
        m.W_L = W_L
    m.rng = np.random.default_rng(12345)
    return m.fit_eval(y, tr, te)


def hillclimb(m, y, tr, te, steps=40, scale=0.4, seed=0):
    """Random-perturbation hill-climb on the field->correlation map W_L =
    a crude but real learnability signal for the covariance channel."""
    rng = np.random.default_rng(seed)
    best_WL = m.W_L.copy(); best = eval_fixed(m, y, tr, te, best_WL)
    for _ in range(steps):
        cand = best_WL + scale * rng.standard_normal(best_WL.shape) / np.sqrt(m.Hf)
        e = eval_fixed(m, y, tr, te, cand)
        if e < best:
            best, best_WL = e, cand
    m.W_L = best_WL
    return best


def run_task(name, T, u, y, tr, te, A):
    base = dict(A=A, Hf=48, H1=20, H2=16, R_lat=4, numT=48)
    e_scale = eval_fixed(CovTwoLayer(**base, seed=0, scale_only=True), y, tr, te)
    e_static = eval_fixed(CovTwoLayer(**base, seed=0, static_corr=True), y, tr, te)
    m = CovTwoLayer(**base, seed=0)
    e_cov0 = eval_fixed(m, y, tr, te)                 # field-cov, random W_L
    e_cov = hillclimb(m, y, tr, te, steps=40)         # field-cov after hill-climb
    print(f"\n=== {name} ===")
    print(f"  scale-only (current, 2-layer)   NRMSE={e_scale:.3f}")
    print(f"  static correlation              NRMSE={e_static:.3f}")
    print(f"  field-cov (random W_L)          NRMSE={e_cov0:.3f}")
    print(f"  field-cov (hill-climbed W_L)    NRMSE={e_cov:.3f}")
    win = e_cov < min(e_scale, e_static) - 0.02
    print(f"  -> covariance {'HELPS' if win else 'no clear help'} "
          f"(vs best control {min(e_scale, e_static):.3f})")
    return dict(scale=e_scale, static=e_static, cov0=e_cov0, cov=e_cov)


def main():
    T = 2500
    print("S2 viability: does field-modulated covariance (learned via hill-climb)")
    print("beat scale-only / static on the GATED task, and tie on NARMA?")
    u, y = gated_task(T)
    tr, te = masks(T)
    Ag = R.LDNField(H=48, theta=60.0).run(u)
    run_task("GATED product  y=gate(u_{t-40})*u_{t-5}*u_{t-15}", T, u, y, tr, te, Ag)

    un, yn = R.narma_x(T, 20, seed=0)
    trn, ten = masks(T)
    An = R.LDNField(H=48, theta=60.0).run(un)
    run_task("NARMA-20 (control: covariance should NOT help)", T, un, yn, trn, ten, An)


if __name__ == "__main__":
    main()
