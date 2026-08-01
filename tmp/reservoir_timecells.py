"""Time-cell scaling test (docs/idea_reservoir.md §13.4): is 'LDN field = time
cells' a quantitative correspondence or just a metaphor?

Real hippocampal / entorhinal time cells are scale-invariant: a cell's temporal
receptive-field WIDTH grows with its PREFERRED DELAY (w ~ tau, Weber's law). We
test whether the dissipative fields reproduce this.

Field-agnostic 'time cell' definition: for a target delay D, fit a ridge decoder
w_D that reconstructs u(t-D) from the field state; its IMPULSE RESPONSE
h_D(t) = X_pulse @ w_D is the cell's temporal tuning. Measure preferred delay
tau = argmax|h_D| and width w = FWHM of |h_D|. Plot w vs tau (log-log) and fit
the scaling exponent; Weber = slope 1. Compared across LDN (time cells),
damped-orth (neuromodulator clearance) and diffusion (volume transmission), and
LDN is swept over Hf to show tiling = temporal-resolution resource allocation
(bridges to §10.21's Hf-limited long memory).

Pass (§13.4): LDN's w grows monotonically with tau and the exponent is ~1 (Weber
to order of magnitude). Else: drop 'time cells', keep 'orthogonal time basis'.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import reservoir as R
from reservoir.readout import ridge_fit, standardize_fit

OUT = "out/reservoir_timecells"


def peak_width_quality(h, D, pulse_at=10):
    """Given a decoder's impulse response h (pulse at `pulse_at`, target delay D):
    return (tau, width, ok). tau = peak delay; width = interpolated FWHM of the
    main positive lobe (sub-sample, continuous -- avoids integer quantisation and
    the tail-sensitivity of a second moment); ok flags that the peak tracks D."""
    a = np.abs(h.copy()); a[:pulse_at] = 0.0
    pk = int(a.argmax()); tau = pk - pulse_at
    if a[pk] < 1e-9:
        return tau, np.nan, False
    hm = a[pk] / 2.0

    def cross(i0, step):                                    # sub-sample half-max
        i = pk
        while 0 <= i + step < len(a) and a[i + step] > hm:
            i += step
        j = i + step
        if not (0 <= j < len(a)) or a[i] == a[j]:
            return float(i)
        return i + step * (a[i] - hm) / (a[i] - a[j])       # linear interp
    width = cross(pk, +1) - cross(pk, -1)
    ok = abs(tau - D) <= max(3, 0.25 * D)                   # peak must be near D
    return tau, float(width), ok


def delay_cells(field, T=4000, Tp=220, delays=None, alpha=1e-3, seed=0):
    """Fit delay decoders on random input; return (tau, width) for cells whose
    decoder peak actually tracks the target delay (reliable cells only)."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, T)
    X = field.run(u)
    w0 = 200
    mu, sd = standardize_fit(X[w0:])
    Xs = (X - mu) / sd
    up = np.zeros(Tp); up[10] = 1.0
    Xp = (field.run(up) - mu) / sd
    taus, widths = [], []
    for D in delays:
        tgt = np.zeros(T); tgt[D:] = u[:T - D]
        wD = ridge_fit(Xs[w0:], tgt[w0:], alpha)
        h = Xp @ wD
        tau, width, ok = peak_width_quality(h, D)
        if ok and np.isfinite(width):
            taus.append(tau); widths.append(width)
    return np.array(taus, float), np.array(widths, float)


def fit_exponent(tau, w):
    m = (tau > 1) & (w > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    p = np.polyfit(np.log(tau[m]), np.log(w[m]), 1)
    return p[0], np.exp(p[1])                    # slope (exponent), prefactor


def main():
    os.makedirs(OUT, exist_ok=True)
    delays = np.arange(4, 92, 3)
    fields = {
        "LDN (time cells)": (R.LDNField(H=48, theta=100.0), "#C44E52"),
        "damped-orth": (R.DampedOrthField(H=48, rho=0.985, seed=0), "#4C72B0"),
        "diffusion": (R.DiffusionField(H=48, D=0.8, gamma=0.008, seed=0), "#55A868"),
    }

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # panel 1: w vs tau, all fields, with Weber (slope-1) reference
    a = ax[0]
    print("=== w vs tau scaling exponent (Weber = 1) ===")
    for name, (fld, col) in fields.items():
        tau, w = delay_cells(fld, delays=delays)
        slope, pref = fit_exponent(tau, w)
        a.plot(tau, w, "o", color=col, ms=5,
               label=f"{name}: exp={slope:.2f}")
        if np.isfinite(slope):
            xs = np.array([tau[tau > 1].min(), tau.max()])
            a.plot(xs, pref * xs ** slope, "-", color=col, alpha=0.6)
        print(f"  {name:20s}: exponent={slope:.3f}  n_cells={len(tau)}")
    tref = np.array([5.0, 50.0])
    a.plot(tref, 0.9 * tref, "k--", alpha=0.5, label="Weber  $w\\propto\\tau$")
    a.set(xscale="log", yscale="log", xlabel="preferred delay $\\tau$ (steps)",
          ylabel="receptive-field width $w$ (FWHM)",
          title="(1) time-cell scaling: width vs preferred delay")
    a.legend(fontsize=8); a.grid(alpha=0.25, which="both")

    # panel 2: LDN tiling vs Hf (resource allocation)
    a = ax[1]
    for Hf, col in ((24, "#8172B3"), (48, "#C44E52"), (96, "#CCB974")):
        fld = R.LDNField(H=Hf, theta=100.0)
        tau, w = delay_cells(fld, delays=delays)
        slope, _ = fit_exponent(tau, w)
        a.plot(tau, w, "-o", color=col, ms=4,
               label=f"LDN $H_f$={Hf} (exp={slope:.2f})")
    a.set(xscale="log", yscale="log", xlabel="preferred delay $\\tau$ (steps)",
          ylabel="receptive-field width $w$ (FWHM)",
          title="(2) LDN tiling: $H_f$ = temporal-resolution budget")
    a.legend(fontsize=8); a.grid(alpha=0.25, which="both")

    fig.suptitle("§13.4 — dissipative fields vs scale-invariant time cells "
                 "(Weber's law)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fp = os.path.join(OUT, "timecells_scaling.png")
    fig.savefig(fp, dpi=130)
    print(f"saved -> {fp}")


if __name__ == "__main__":
    main()
