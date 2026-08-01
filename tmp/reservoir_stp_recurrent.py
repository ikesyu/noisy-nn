"""Recurrent STP field (Mongillo-style): does RECURRENCE structure/rescue the
memory-poor feedforward STP bank? (docs §10.22 follow-up.)

A fixed recurrent rate reservoir whose recurrent synapses undergo short-term
plasticity (presynaptic facilitation u * depression x scales the effective
weight). The field stays DISSIPATIVE (recurrence kept below critical -> fading
memory / ESP, not a persistent Mongillo attractor which would break ESP). It is a
fixed reservoir, so the (B)-mix map still learns forward-only (no BPTT).

    a(t) = (1-leak) a(t-1) + leak ( W_eff a(t-1) + w_in u(t) ),
    W_eff column j scaled by g_j = u_j x_j  (presynaptic STP gain),
    u,x driven by presynaptic activity |a|.

Controls: same recurrent W with STP OFF (random / orthogonal linear reservoir),
feedforward STP (no recurrence, dim already covered), and LDN/cascade/ESN refs.
Question: recurrent-STP MC/NARMA vs feedforward-STP (MC~6) and vs no-STP recurrent.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import reservoir as R
from reservoir.metrics import memory_capacity


class RecurrentSTPField:
    def __init__(self, H=48, rho=0.95, leak=0.3, density=0.2, use_stp=True,
                 orthogonal=False, tf=(3.0, 60.0), td=(5.0, 150.0), U=(0.1, 0.5),
                 seed=0):
        rng = np.random.default_rng(seed)
        if orthogonal:
            Q, _ = np.linalg.qr(rng.standard_normal((H, H)))
            W = rho * Q
        else:
            W = rng.standard_normal((H, H)) * (rng.random((H, H)) < density)
            W *= rho / (np.max(np.abs(np.linalg.eigvals(W))) + 1e-12)
        self.W, self.leak, self.H = W, leak, H
        self.w_in = rng.choice([-1.0, 1.0], size=H)
        self.use_stp = use_stp
        self.tau_f = np.exp(rng.uniform(np.log(tf[0]), np.log(tf[1]), H))
        self.tau_d = np.exp(rng.uniform(np.log(td[0]), np.log(td[1]), H))
        self.U = rng.uniform(U[0], U[1], H)

    def run(self, u_in):
        H = self.H
        a = np.zeros(H); uu = self.U.copy(); xx = np.ones(H)
        X = np.empty((len(u_in), H))
        for t in range(len(u_in)):
            if self.use_stp:
                r = np.abs(a)
                uu = uu + (self.U - uu) / self.tau_f + self.U * (1 - uu) * r
                xx = xx + (1 - xx) / self.tau_d - uu * xx * r
                uu = np.clip(uu, 0.0, 1.0); xx = np.clip(xx, 0.0, 1.0)
                pre = self.W @ ((uu * xx) * a)
            else:
                pre = self.W @ a
            a = (1 - self.leak) * a + self.leak * (pre + self.w_in * u_in[t])
            X[t] = a
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
        "recur random +STP":  lambda sd: RecurrentSTPField(rho=0.95, use_stp=True, seed=sd),
        "recur random -STP":  lambda sd: RecurrentSTPField(rho=0.95, use_stp=False, seed=sd),
        "recur orth +STP":    lambda sd: RecurrentSTPField(rho=0.97, orthogonal=True, use_stp=True, seed=sd),
        "recur orth -STP":    lambda sd: RecurrentSTPField(rho=0.97, orthogonal=True, use_stp=False, seed=sd),
        "feedforward STP (u)": lambda sd: __import__("reservoir_stp").STPField(coord="u", seed=sd),
        "LDN (ref)":     lambda sd: R.LDNField(H=48, theta=60.0),
        "cascade (ref)": lambda sd: R.CascadeField(H=48, a=0.92),
        "ESN (ref)":     lambda sd: R.LeakyESN(H=48, seed=sd),
    }
    print("=== recurrent STP field vs controls: MC and (B)-mix NARMA-20 ===")
    rows = []
    for name, mk in fields.items():
        mc, na = eval_field(mk)
        rows.append((name, mc, na))
        print(f"  {name:22s}  MC={mc:5.1f}   NARMA20(B-mix)={na:.3f}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    names = [r[0] for r in rows]; y = np.arange(len(names))
    col = ["#C44E52", "#E29587", "#8C2D3F", "#D6A0A8", "#DD8452", "#4C72B0", "#DD8452", "#7f7f7f"]
    ax[0].barh(y, [r[1] for r in rows], color=col[:len(rows)])
    ax[0].set_yticks(y); ax[0].set_yticklabels(names, fontsize=8); ax[0].invert_yaxis()
    ax[0].set(xlabel="total memory capacity", title="(a) MC"); ax[0].grid(alpha=0.25, axis="x")
    ax[1].barh(y, [r[2] for r in rows], color=col[:len(rows)])
    ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].invert_yaxis()
    ax[1].set(xlabel="(B)-mix NARMA-20 NRMSE", title="(b) task"); ax[1].grid(alpha=0.25, axis="x")
    fig.suptitle("Does recurrence rescue the STP field? (Mongillo-style, kept "
                 "dissipative)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fp = "out/reservoir_stp/stp_recurrent.png"
    fig.savefig(fp, dpi=130)
    print(f"saved -> {fp}")


if __name__ == "__main__":
    main()
