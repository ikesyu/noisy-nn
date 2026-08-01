"""STP (short-term synaptic plasticity) as the noise field
(docs/idea_reservoir.md §9.1 / §10.22).

Biologically-grounded alternative to the neuromodulator (diffusion) reading that
fits LDN/cascade: each field coordinate is the synaptic-efficacy state of a
Tsodyks-Markram synapse (facilitation u = residual Ca2+, depression x = available
vesicles) driven by the input rate. The state u*x is a dissipative, input-driven
memory that multiplicatively gates transmission gain -- exactly the field's role
in (B). A BANK of synapses with heterogeneous time-constants gives a
multi-timescale memory (Mongillo, Barak & Tsodyks 2008 = working memory in STP).

Continuous TM dynamics (rate r(t) in [0,1] = affine map of the reservoir input):
    du/dt = (U - u)/tau_f + U (1 - u) r
    dx/dt = (1 - x)/tau_d - u x r
    field coordinate = u * x  (gain state; persists after input = memory)

Test: total MC and (B)-mix NARMA-20 vs LDN / cascade references.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import reservoir as R
from reservoir.metrics import memory_capacity


class STPField:
    """Bank of H Tsodyks-Markram synapses with heterogeneous (tau_f, tau_d, U).
    coord: 'gain' = u*x (default), 'x' = resources, 'u' = facilitation."""

    def __init__(self, H=48, coord="gain", seed=0,
                 tf=(2.0, 100.0), td=(5.0, 250.0), U=(0.1, 0.5)):
        rng = np.random.default_rng(seed)
        self.H = H
        self.tau_f = np.exp(rng.uniform(np.log(tf[0]), np.log(tf[1]), H))
        self.tau_d = np.exp(rng.uniform(np.log(td[0]), np.log(td[1]), H))
        self.U = rng.uniform(U[0], U[1], H)
        self.coord = coord

    def run(self, u_in):
        r = 0.5 * (u_in + 1.0)                      # affine -> rate in [0,1]
        u = self.U.copy(); x = np.ones(self.H)
        X = np.empty((len(u_in), self.H))
        for t in range(len(u_in)):
            rt = r[t]
            u = u + (self.U - u) / self.tau_f + self.U * (1 - u) * rt
            x = x + (1 - x) / self.tau_d - u * x * rt
            u = np.clip(u, 0.0, 1.0); x = np.clip(x, 0.0, 1.0)
            if self.coord == "x":
                X[t] = x
            elif self.coord == "u":
                X[t] = u
            else:
                X[t] = u * x
        return X


def masks(T, washout=300, frac=0.7):
    idx = np.arange(washout, T); ntr = int(len(idx) * frac)
    tr = np.zeros(T, bool); tr[idx[:ntr]] = True
    te = np.zeros(T, bool); te[idx[ntr:]] = True
    return tr, te


def eval_field(make_field, seeds=3, T=3000, epochs=250, Ho=48):
    mcs, narmas = [], []
    for sd in range(seeds):
        u_mc = np.random.default_rng(sd).uniform(-1, 1, T)
        _, mc = memory_capacity(make_field(sd).run(u_mc), u_mc, max_delay=60)
        mcs.append(mc)
        u, y = R.narma_x(T, 20, seed=sd)
        tr, te = masks(T)
        A = make_field(sd).run(u)
        e = R.NoiseModulatedMap(A, y, tr, Ho=Ho, mix=True, seed=100 + sd).eval(te, epochs)
        narmas.append(e)
    return float(np.mean(mcs)), float(np.mean(narmas))


def main():
    os.makedirs("out/reservoir_stp", exist_ok=True)
    fields = {
        "STP gain (u*x)":  lambda sd: STPField(coord="gain", seed=sd),
        "STP resources x": lambda sd: STPField(coord="x", seed=sd),
        "STP facilit. u":  lambda sd: STPField(coord="u", seed=sd),
        "STP + delay-ish (long td)": lambda sd: STPField(coord="gain", seed=sd, td=(20.0, 600.0)),
        "LDN (ref)":     lambda sd: R.LDNField(H=48, theta=60.0),
        "cascade (ref)": lambda sd: R.CascadeField(H=48, a=0.92),
        "diffusion (ref)": lambda sd: R.DiffusionField(H=48, seed=sd),
    }
    print("=== STP field vs references: total MC and (B)-mix NARMA-20 ===")
    rows = []
    for name, mk in fields.items():
        mc, na = eval_field(mk)
        rows.append((name, mc, na))
        print(f"  {name:28s}  MC={mc:5.1f}   NARMA20(B-mix)={na:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    names = [r[0] for r in rows]; y = np.arange(len(names))
    col = ["#C44E52"] * 4 + ["#4C72B0", "#DD8452", "#55A868"]
    ax[0].barh(y, [r[1] for r in rows], color=col)
    ax[0].set_yticks(y); ax[0].set_yticklabels(names, fontsize=8); ax[0].invert_yaxis()
    ax[0].set(xlabel="total memory capacity", title="(a) MC")
    ax[0].grid(alpha=0.25, axis="x")
    ax[1].barh(y, [r[2] for r in rows], color=col)
    ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].invert_yaxis()
    ax[1].set(xlabel="(B)-mix NARMA-20 NRMSE (lower=better)", title="(b) task")
    ax[1].grid(alpha=0.25, axis="x")
    fig.suptitle("STP (Tsodyks-Markram) field vs LDN / cascade / diffusion",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fp = "out/reservoir_stp/stp.png"
    fig.savefig(fp, dpi=130)
    print(f"saved -> {fp}")


if __name__ == "__main__":
    main()
