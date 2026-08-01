"""Making the diffusion (volume-transmission) field competitive
(docs/idea_reservoir.md §5.7 / §10.14). The 1D leaky-diffusion field has thin
memory (MC~9) because its exponential time-constants are near-collinear (low
effective rank). Levers tested here:

  (1) DYNAMICS raise the intrinsic memory: 1D vs 2D grid topology, advection
      (a travelling bump = position-encodes-time), and D/gamma tuning. Total MC
      is basis-INVARIANT, so only the dynamics can move it.
  (2) EIGENBASIS reading disentangles: reading the Laplacian eigenmodes (each an
      orthogonal, distinct-time-constant channel) vs the raw entangled nodes.
      This matters for the DIAGONAL (B) coupling (one coord per unit); the mixed
      (B) absorbs a linear basis change, so it is invariant there -- we verify both.

All fields ZOH-discretised (matrix exponential) for stability. Ties to
Ikemoto & DallaLibera (noise-field spatial structure should match the task's
function-proximity): the 2D topology is the knob for that spatial match.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm

import reservoir as R
from reservoir.metrics import memory_capacity


def path_laplacian(H):
    L = np.zeros((H, H))
    for i in range(H):
        for j in (i - 1, i + 1):
            if 0 <= j < H:
                L[i, j] = -1; L[i, i] += 1
    return L


def grid_laplacian(Lx, Ly):
    H = Lx * Ly
    L = np.zeros((H, H))
    idx = lambda r, c: r * Ly + c
    for r in range(Lx):
        for c in range(Ly):
            i = idx(r, c)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < Lx and 0 <= c2 < Ly:
                    L[i, idx(r2, c2)] = -1; L[i, i] += 1
    return L


class DiffZOH:
    """Diffusion + leak, ZOH-discretised: x(t) = M x(t-1) + w_in u(t),
    M = expm(-(D*Lap + gamma*I)*dt). Optional upwind advection (drift)."""

    def __init__(self, Lap, D=0.3, gamma=0.01, v=0.0, dt=1.0, seed=0,
                 inject="dist"):
        H = Lap.shape[0]
        Adv = np.zeros((H, H))
        if v != 0.0:                                   # upwind first difference
            for i in range(H):
                Adv[i, i] += 1
                if i - 1 >= 0:
                    Adv[i, i - 1] -= 1
        Gen = D * Lap + v * Adv + gamma * np.eye(H)
        self.M = expm(-Gen * dt)
        self.H = H
        rng = np.random.default_rng(seed)
        if inject == "local":
            w = np.zeros(H); w[0] = 1.0; self.w_in = w
        else:
            self.w_in = rng.choice([-1.0, 1.0], size=H)
        self.Lap = Lap

    def run(self, u):
        x = np.zeros(self.H); X = np.empty((len(u), self.H))
        for t in range(len(u)):
            x = self.M @ x + self.w_in * u[t]; X[t] = x
        return X

    def eigvecs(self):
        _, V = np.linalg.eigh(self.Lap)
        return V


def masks(T, washout=300, frac=0.7):
    idx = np.arange(washout, T); ntr = int(len(idx) * frac)
    tr = np.zeros(T, bool); tr[idx[:ntr]] = True
    te = np.zeros(T, bool); te[idx[ntr:]] = True
    return tr, te


def eval_field(make_field, seeds=3, T=3000, epochs=250, Ho=48, mix=True,
               transform=None):
    """(total MC, (B) NARMA-20 NRMSE) averaged over seeds. transform: optional
    fixed [H,H] matrix applied to field states (e.g. eigenvectors)."""
    mcs, narmas = [], []
    for sd in range(seeds):
        u_mc = np.random.default_rng(sd).uniform(-1, 1, T)
        Xmc = make_field(sd).run(u_mc)
        if transform is not None:
            Xmc = Xmc @ transform
        _, mc = memory_capacity(Xmc, u_mc, max_delay=60)
        mcs.append(mc)
        u, y = R.narma_x(T, 20, seed=sd)
        tr, te = masks(T)
        A = make_field(sd).run(u)
        if transform is not None:
            A = A @ transform
        e = R.NoiseModulatedMap(A, y, tr, Ho=Ho, mix=mix, seed=100 + sd).eval(te, epochs)
        narmas.append(e)
    return float(np.mean(mcs)), float(np.mean(narmas))


def main():
    os.makedirs("out/reservoir_diffusion", exist_ok=True)
    H = 48
    L1 = path_laplacian(H)
    L2 = grid_laplacian(7, 7)              # 49 nodes ~ H

    fields = {
        "diff-1D baseline (D.6,g.02)": lambda sd: DiffZOH(L1, D=0.6, gamma=0.02, seed=sd),
        "diff-1D best (D.8,g.008)":    lambda sd: DiffZOH(L1, D=0.8, gamma=0.008, seed=sd),
        "diff-2D grid (D.3,g.008)":    lambda sd: DiffZOH(L2, D=0.3, gamma=0.008, seed=sd),
        "advection-diff (v.9,D.05)":   lambda sd: DiffZOH(L1, D=0.05, gamma=0.004, v=0.9, seed=sd, inject="local"),
    }
    refs = {
        "LDN (ref)":     lambda sd: R.LDNField(H=H, theta=60.0),
        "cascade (ref)": lambda sd: R.CascadeField(H=H, a=0.92),
    }

    print("=== (1) field dynamics: total MC and (B)-mix NARMA-20 ===")
    rows = []
    for name, mk in {**fields, **refs}.items():
        mc, na = eval_field(mk)
        rows.append((name, mc, na))
        print(f"  {name:30s}  MC={mc:5.1f}   NARMA20(B-mix)={na:.3f}")

    print("\n=== (2) eigenbasis reading in DIAGONAL (B): raw nodes vs eigenmodes ===")
    diag_rows = []
    for name, mk, Lap in (("diff-1D best", lambda sd: DiffZOH(L1, D=0.8, gamma=0.008, seed=sd), L1),
                          ("diff-2D grid", lambda sd: DiffZOH(L2, D=0.3, gamma=0.008, seed=sd), L2)):
        _, V = np.linalg.eigh(Lap)
        _, na_raw = eval_field(mk, mix=False, Ho=Lap.shape[0])
        _, na_eig = eval_field(mk, mix=False, Ho=Lap.shape[0], transform=V)
        # mix (absorbs basis) control:
        _, mx_raw = eval_field(mk, mix=True)
        _, mx_eig = eval_field(mk, mix=True, transform=V)
        diag_rows.append((name, na_raw, na_eig, mx_raw, mx_eig))
        print(f"  {name:16s}  DIAG raw={na_raw:.3f} eig={na_eig:.3f} | "
              f"MIX raw={mx_raw:.3f} eig={mx_eig:.3f}")

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    names = [r[0] for r in rows]; mc = [r[1] for r in rows]; na = [r[2] for r in rows]
    y = np.arange(len(names)); col = ["#55A868"] * 4 + ["#C44E52", "#DD8452"]
    a.barh(y, na, color=col)
    for i, (m, n) in enumerate(zip(mc, na)):
        a.text(n + 0.01, i, f"MC={m:.0f}", va="center", fontsize=8)
    a.set_yticks(y); a.set_yticklabels(names, fontsize=8); a.invert_yaxis()
    a.set(xlabel="(B)-mix NARMA-20 NRMSE (lower=better)",
          title="(1) diffusion dynamics vs references")
    a.grid(alpha=0.25, axis="x")

    a = ax[1]
    dn = [r[0] for r in diag_rows]; x = np.arange(len(dn)); w = 0.2
    a.bar(x - 1.5 * w, [r[1] for r in diag_rows], w, color="#8172B3", label="DIAG raw nodes")
    a.bar(x - 0.5 * w, [r[2] for r in diag_rows], w, color="#C44E52", label="DIAG eigenmodes")
    a.bar(x + 0.5 * w, [r[3] for r in diag_rows], w, color="#B0B0B0", label="MIX raw")
    a.bar(x + 1.5 * w, [r[4] for r in diag_rows], w, color="#4C72B0", label="MIX eigenmodes")
    a.set_xticks(x); a.set_xticklabels(dn, fontsize=9)
    a.set(ylabel="NARMA-20 NRMSE",
          title="(2) eigenbasis reading does NOT help (nodes already pre-mixed)")
    a.legend(fontsize=8); a.grid(alpha=0.25, axis="y")
    fig.suptitle("Diffusion field: only advection raises MC/perf; 2D & eigenbasis "
                 "do not (memory is basis-invariant, nodes already entangled)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fp = "out/reservoir_diffusion/diffusion.png"
    fig.savefig(fp, dpi=130)
    print(f"\nsaved -> {fp}")


if __name__ == "__main__":
    main()
