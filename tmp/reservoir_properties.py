"""The system's UNIQUE properties (docs/idea_reservoir.md §13.3, path (a)).

§13.3 showed the performance-superiority claim does not survive a fair NG-RC
baseline, so the raison d'etre moves to the properties NG-RC / LMU cannot have.
This is the centerpiece figure for that thesis -- four properties, each a thing
the additive / deterministic prior art structurally lacks:

  (1) Multiplicative-noise SR optimum  the signal modulates NOISE, so there is an
      OPTIMAL noise level (interior reverse-U, §13.1). A mean-field / fixed-sigma
      system (NG-RC, LMU) has no such optimum -- noise is a functional resource.
  (2) Forward-only credit              the nonlinear feature map is learned with
      NO BPTT; the analytic credit matches finite-difference to ~1e-8.
  (3) sigma->0 ESP / fading memory     a dissipative field: after input stops the
      state decays to zero (echo-state property by construction, §2).
  (4) Time cells                       the best implicit-memory field (LDN) tiles
      the delay window -- a biological reading (bridges to §13.4).

Reloads the §13.1 SR curve from out/reservoir_sr/ ; run reservoir_sr.py first.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import ndtr

import reservoir as R
from reservoir.fields import pulse_decay

OUT = "out/reservoir_properties"
SR_CSV = "out/reservoir_sr/sr_sigma_narma20.csv"


def masks(T, w=50, f=0.7):
    idx = np.arange(w, T); n = int(len(idx) * f)
    tr = np.zeros(T, bool); tr[idx[:n]] = True
    te = np.zeros(T, bool); te[idx[n:]] = True
    return tr, te


def grad_arrays():
    """Analytic vs finite-difference gradient of (B)-mix M (W,mu,sd frozen)."""
    rng = np.random.default_rng(0)
    T, Hf, Ho = 400, 8, 6
    A = rng.standard_normal((T, Hf)); y = rng.standard_normal(T)
    tr, _ = masks(T)
    m = R.NoiseModulatedMap(A, y, tr, Ho=Ho, mix=True, seed=1); m._t = 1
    z, sig, dz_darg, pre = m._feat()
    W, mu, sd, X = m._readout(z)
    e = np.where(m.tr, (X - mu) / sd @ W - m.y, 0.0); n = m.tr.sum()
    base = (2.0 / n) * e[:, None] * (W[:m.Ho] / sd[0, :m.Ho])[None, :] * dz_darg
    dLdpre = base * (-m.d / sig ** 2) * m._sigmoid(pre)
    gM = dLdpre.T @ m.A                                    # analytic

    def loss():
        zz, _, _, _ = m._feat()
        XX = np.concatenate([zz, np.ones((len(zz), 1))], axis=1)
        ee = np.where(m.tr, (XX - mu) / sd @ W - m.y, 0.0)
        return (ee ** 2).sum() / n

    eps = 1e-6; gnum = np.zeros_like(m.M)
    it = np.nditer(m.M, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index; o = m.M[i]
        m.M[i] = o + eps; lp = loss(); m.M[i] = o - eps; lm = loss()
        m.M[i] = o; gnum[i] = (lp - lm) / (2 * eps); it.iternext()
    ga, gn = gM.ravel(), gnum.ravel()
    cos = float(ga @ gn / (np.linalg.norm(ga) * np.linalg.norm(gn) + 1e-12))
    rel = float(np.linalg.norm(ga - gn) / (np.linalg.norm(gn) + 1e-12))
    return ga, gn, cos, rel


def time_cells(Hf=48, theta=60.0, T=160):
    """LDN response to a pulse, decoded at fractional delays -> tuning curves."""
    fld = R.LDNField(H=Hf, theta=theta)
    u = np.zeros(T); u[10] = 1.0
    X = fld.run(u)
    rs = np.linspace(0.1, 0.9, 6)
    curves = [(r, X @ fld.decode_weights(r)) for r in rs]
    return curves, theta


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

    # (1) SR optimum -- reload §13.1 curve
    a = ax[0, 0]
    if os.path.exists(SR_CSV):
        D = np.loadtxt(SR_CSV, delimiter=",")
        s, na, ns = D[:, 0], D[:, 1], D[:, 2]
        a.plot(s, na, "-o", ms=3, color="#4C72B0", label="analytic (fixed-$\\sigma$/mean-field)")
        a.plot(s, ns, "-o", ms=3, color="#C44E52", label="sample (noise mechanism)")
        a.axvline(s[ns.argmin()], color="#C44E52", ls=":", alpha=0.6)
        a.set_xscale("log")
    else:
        a.text(0.5, 0.5, "run reservoir_sr.py first", ha="center")
    a.set(xlabel="global noise strength $s$", ylabel="NARMA-20 test NRMSE",
          title="(1) noise is a resource: multiplicative-noise SR optimum")
    a.legend(fontsize=8); a.grid(alpha=0.25)

    # (2) forward-only credit
    a = ax[0, 1]
    ga, gn, cos, rel = grad_arrays()
    lim = max(np.abs(ga).max(), np.abs(gn).max()) * 1.1
    a.plot([-lim, lim], [-lim, lim], color="0.6", lw=1, zorder=0)
    a.scatter(gn, ga, s=14, color="#55A868", alpha=0.8)
    a.set(xlabel="finite-difference $\\partial L/\\partial M$",
          ylabel="analytic (forward-only) $\\partial L/\\partial M$",
          title="(2) forward-only credit = exact (no BPTT)")
    a.text(0.05, 0.9, f"cosine = {cos:.6f}\nrel. err = {rel:.1e}",
           transform=a.transAxes, fontsize=9, va="top")
    a.grid(alpha=0.25)

    # (3) sigma->0 ESP / fading memory
    a = ax[1, 0]
    for name, fld, col in (("LDN (time cells)", R.LDNField(H=48, theta=60.0), "#C44E52"),
                           ("cascade", R.CascadeField(H=48, a=0.92), "#DD8452"),
                           ("damped-orth", R.DampedOrthField(H=48, rho=0.97, seed=0), "#4C72B0")):
        a.plot(pulse_decay(fld, T=200), color=col, label=name)
    a.set(xlabel="steps after an input pulse", ylabel="$\\|$field$\\|$ (normalised)",
          title="(3) $\\sigma\\to0$ ESP: dissipative field forgets its input")
    a.legend(fontsize=8); a.grid(alpha=0.25)

    # (4) time cells
    a = ax[1, 1]
    curves, theta = time_cells()
    for r, c in curves:
        a.plot(c / (np.abs(c).max() + 1e-12), label=f"$\\tau$={r*theta:.0f}")
    a.set(xlabel="time since stimulus (steps)", ylabel="decoded delay-cell (norm.)",
          title="(4) biological reading: LDN field = time cells", xlim=(0, 70))
    a.legend(fontsize=7, ncol=2); a.grid(alpha=0.25)

    fig.suptitle("What the noise-modulated reservoir has that NG-RC / LMU do not "
                 "(§13.3 path a)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fp = os.path.join(OUT, "properties_ii.png")
    fig.savefig(fp, dpi=130)
    print(f"cosine={cos:.6f} rel_err={rel:.2e}")
    print(f"saved -> {fp}")


if __name__ == "__main__":
    main()
