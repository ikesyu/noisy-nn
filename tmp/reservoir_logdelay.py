"""Does a LOG-spaced delay line beat a uniform cascade on wide-range multiscale
memory? (docs/idea_reservoir.md §5.8 memory-allocation frontier.)

Total linear MC is bounded by H (Jaeger); the only lever is HOW the H taps are
allocated across lags. Uniform cascade/delay covers lags 1..H densely (reach ~H);
a LOG-spaced delay line puts H taps at 1,2,4,8,... (reach ~max_lag >> H) --
exponentially longer range, sparse coverage. Prediction (allocation trade-off):
log WINS when the task needs a FAR lag (> H), LOSES on DENSE short-lag tasks.

The allocation question is about the FIELD's LINEAR memory, so we use a LINEAR
ridge read-out (the correct tool: it directly tests whether the needed lags are
present; the (B) noise-map would sit identically on every field). Tests:
  (1) exact-lag MC(k): reach -- uniform block to H, log comb to max_lag;
  (2) far single lag y=u(t-2)+u(t-D): log reaches past H, uniform cliffs;
  (3) dense short y=mean_{k=1..20} u(t-k): uniform covers, log misses lags.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import reservoir as R
from reservoir.readout import ridge_fit, standardize_fit, nrmse, corr2


def _log_lags(H, max_lag):
    cand = np.unique(np.round(np.geomspace(1, max_lag, 4 * H)).astype(int))
    if len(cand) > H:
        cand = cand[np.round(np.linspace(0, len(cand) - 1, H)).astype(int)]
    return cand


class DelayField:
    """Taps at given lags, optional geometric decay (decay<1 -> dissipative/ESP)."""
    def __init__(self, lags, decay=1.0):
        self.lags = np.asarray(lags); self.H = len(self.lags); self.decay = decay

    def run(self, u):
        T = len(u); X = np.zeros((T, self.H))
        for j, L in enumerate(self.lags):
            if L < T:
                X[L:, j] = u[:T - L] * (self.decay ** L)
        return X


def masks(T, washout=600, frac=0.7):
    idx = np.arange(washout, T); ntr = int(len(idx) * frac)
    tr = np.zeros(T, bool); tr[idx[:ntr]] = True
    te = np.zeros(T, bool); te[idx[ntr:]] = True
    return tr, te


def lin_fit_eval(X, y, tr, te, alpha=1e-4):
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    mu, sd = standardize_fit(Xb[tr]); Xs = (Xb - mu) / sd
    W = ridge_fit(Xs[tr], y[tr], alpha)
    return nrmse(y[te], (Xs @ W)[te])


def exact_mc(field, ks, T=6000, seed=0):
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    X = field.run(u); tr, te = masks(T)
    mu, sd = standardize_fit(X[tr]); Xs = (X - mu) / sd
    out = []
    for k in ks:
        tgt = np.zeros(T); tgt[k:] = u[:T - k]
        W = ridge_fit(Xs[tr], tgt[tr], 1e-6)
        out.append(corr2(tgt[te], (Xs @ W)[te]))
    return np.array(out)


def main():
    os.makedirs("out/reservoir_logdelay", exist_ok=True)
    H, ML, T = 48, 512, 6000
    fields = {
        "uniform cascade":  DelayField(np.arange(1, H + 1), decay=0.92),
        "log delay":        DelayField(_log_lags(H, ML)),
        "LDN theta=60":     R.LDNField(H=H, theta=60.0),
        "LDN theta=300":    R.LDNField(H=H, theta=300.0),
    }
    COL = {"uniform cascade": "#DD8452", "log delay": "#C44E52",
           "LDN theta=60": "#4C72B0", "LDN theta=300": "#55A868"}

    def windowmean(u, D):
        w = max(2, int(0.18 * D)); T = len(u); s = np.zeros(T)
        for t in range(D + w, T):
            s[t] = u[t - D - w:t - D + w].mean()
        return s

    # (1) exact-lag MC(k): reach (sparse taps vs smooth bases)
    ks = np.arange(1, 320, 2)
    prof = {n: np.mean([exact_mc(f, ks, seed=s) for s in range(2)], 0)
            for n, f in fields.items()}
    for n in fields:
        reach = ks[np.where(prof[n] > 0.5)[0]].max() if (prof[n] > 0.5).any() else 0
        print(f"  {n:18s} exact-lag reach(MC>0.5) up to {reach}")

    # (2) SMOOTH long-range: y = u(t-2) + windowmean(u; center=D)  [linear read-out]
    Ds = [10, 25, 45, 80, 140, 220]
    far = {n: [] for n in fields}
    print("=== (2) smooth long-range y=u(t-2)+windowmean(u;D) ===")
    for D in Ds:
        line = f"  D={D:3d}:"
        for n, f in fields.items():
            errs = []
            for sd in range(3):
                u = np.random.default_rng(sd).uniform(-1, 1, T)
                y = windowmean(u, D).copy(); y[2:] += u[:T - 2]; y -= y.mean()
                tr, te = masks(T)
                errs.append(lin_fit_eval(f.run(u), y, tr, te))
            far[n].append(np.mean(errs)); line += f" {n.split()[0][:3]}={np.mean(errs):.2f}"
        print(line)

    # (3) dense short: y = mean_{k=1..20} u(t-k)
    print("=== (3) dense short-lag task y=mean_{k=1..20} u(t-k) ===")
    dense = {}
    for n, f in fields.items():
        errs = []
        for sd in range(3):
            u = np.random.default_rng(sd).uniform(-1, 1, T)
            y = np.zeros(T)
            for k in range(1, 21):
                y[k:] += u[:T - k]
            y /= 20; y -= y.mean(); tr, te = masks(T)
            errs.append(lin_fit_eval(f.run(u), y, tr, te))
        dense[n] = np.mean(errs)
        print(f"  {n:18s} dense-short NRMSE={dense[n]:.3f}")

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for n in fields:
        ax[0].plot(ks, prof[n], "-", color=COL[n], label=n, lw=1.3)
    ax[0].axvline(H, color="k", ls="--", alpha=0.4); ax[0].text(H, 0.9, " H=48", fontsize=8)
    ax[0].set(xlabel="delay k (exact lag)", ylabel="MC(k)",
              title="(1) reach: uniform=dense block to H, log=sparse comb to 512")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
    for n in fields:
        ax[1].plot(Ds, far[n], "-o", ms=4, color=COL[n],
                   label=f"{n} (dense-short={dense[n]:.2f})")
    ax[1].axvline(H, color="k", ls="--", alpha=0.4)
    ax[1].set(xlabel="long-range center D  y=u(t-2)+windowmean(u;D)", ylabel="test NRMSE",
              title="(2) smooth long-range: LDN(large theta) reaches far, others cliff")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)
    fig.suptitle("Long-range memory: compression (LDN large-theta) beats sparse "
                 "log-taps and uniform reach (§5.8, linear read-out)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fp = "out/reservoir_logdelay/logdelay.png"
    fig.savefig(fp, dpi=130); print(f"saved -> {fp}")


if __name__ == "__main__":
    main()
