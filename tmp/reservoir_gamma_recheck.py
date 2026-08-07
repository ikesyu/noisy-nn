"""gamma-matched re-verification of the activation comparisons.

WHY.  reservoir_lambda_gauge.py showed that comparing activation SHAPES at an
unoptimised effective gain confounds optimisation conditioning with activation
properties: the same task sat at NRMSE 0.55-0.96 under BatchNorm(affine init 1,
i.e. gamma ~ 1) and at 0.09 once gamma was chosen properly.

gamma_k = ||a_k|| / sigma_k  (idea_core sec1.1 / sec4.8) is the gauge-INVARIANT
"how sharply does this unit respond" knob, and s = (d-h)/sigma = gamma * d_perp.
Activation shapes therefore may only be compared at each shape's OWN best gamma.

Claims re-checked here, all from the parity setting of sec10.24 / sec10.27:
  (i)  "the crossing/bump activation COLLAPSES for depth >= 2"   <- core of the
       design axis; previously measured at gamma ~ 1 only.
  (ii) "at a single layer the bump BEATS tanh"                   <- previously
       the bump's width WAS swept while tanh's gain was NOT (asymmetric).
  (iii)"the threshold (monotone) activation rescues deep parity" <- same frame.

Design: BatchNorm(affine=False) fixes the scale, then a FROZEN scalar gamma
multiplies the standardised pre-activation, then the activation.  gamma is thus
exactly the effective gain and is swept identically for every activation,
including tanh.  Compare min_gamma per (activation, depth).
"""
import numpy as np
import torch
import torch.nn as nn

import reservoir as R
from reservoir.readout import standardize_fit, nrmse

from reservoir.moment import masks, LambdaAct, SignField, parity_task

# crossing turned out to be sharply peaked near gamma ~ 1 (0.5:1.00, 1:0.54,
# 2:0.81, 3:1.33), so the grid is refined there rather than log-spaced coarsely.
GAMMAS = (0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
ACTS = ("threshold", "crossing", "tanh")


class Net(nn.Module):
    """BN(affine=False) -> gamma_FROZEN * (p - c_LEARNABLE) -> activation.

    IMPORTANT.  idea_core sec4.8: the complete invariant system per unit is
    (direction a_hat, effective gain gamma, threshold ratio h/sigma).  Freezing
    gamma is the intent; the per-unit shift c must STAY LEARNABLE because it is
    the threshold ratio, a SEPARATE invariant.  A first version of this harness
    used BatchNorm(affine=False) with no shift, which zero-means every unit and
    therefore pins h/sigma ~ 0 for all of them -- that cripples a BUMP (whose
    position is its whole point) while barely touching a monotone activation,
    and it produced a spurious "crossing is terrible at every gamma" (1.07-1.32
    at depth 1, vs 0.55 in sec10.24).  Keep c learnable.
    """
    def __init__(self, Hin, H, depth, act, gamma, numT=64, seed=0):
        super().__init__(); torch.manual_seed(seed)
        d = [Hin] + [H] * depth
        self.ls = nn.ModuleList([nn.Linear(d[i], d[i + 1]) for i in range(depth)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(d[i + 1], affine=False) for i in range(depth)])
        self.cs = nn.ParameterList([nn.Parameter(torch.zeros(d[i + 1])) for i in range(depth)])
        self.out = nn.Linear(d[-1], 1)
        self.act, self.gamma, self.numT = act, gamma, numT

    def _f(self, p, c):
        p = (p - c) * self.gamma
        if self.act == "tanh":
            return torch.tanh(p)
        lam = 0.0 if self.act == "threshold" else 1.0
        return LambdaAct.apply(p, lam, 0.0, self.numT)

    def forward(self, x):
        for L, bn, c in zip(self.ls, self.bns, self.cs):
            x = self._f(bn(L(x)), c)
        return self.out(x).squeeze(-1)


def train_eval(A, y, tr, te, depth, act, gamma, numT=64, H=48, steps=900,
               bs=256, lr=3e-3, seed=0):
    mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
    X = torch.tensor(As, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    net = Net(A.shape[1], H, depth, act, gamma, numT=numT, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        b = torch.tensor(rng.choice(tr_idx, size=bs, replace=False))
        net.train(); opt.zero_grad()
        loss = ((net(X[b]) - yt[b]) ** 2).mean()
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pr = [net(X[torch.tensor(te_idx[i:i+1000])]).numpy()
              for i in range(0, len(te_idx), 1000)]
    return nrmse(y[te_idx], np.concatenate(pr))


def main():
    T = 3000; u, y = parity_task(T); tr, te = masks(T)
    A = SignField(H=32, gain=8.0).run(u)
    SEEDS = 2
    print("=" * 76)
    print("gamma-matched re-check: 3-way sign parity, sample numT=64, 2 seeds")
    print("gamma = effective gain (||a||/sigma), FROZEN and swept identically")
    print("for every activation.  Compare min_gamma per (act, depth).")
    print("=" * 76)
    best = {}
    for depth in (1, 2, 3, 4):
        print(f"\n--- depth {depth} ---")
        print("  act        " + "  ".join(f"g={g:<4g}" for g in GAMMAS) + "  | min_g  argmin")
        for act in ACTS:
            row = [np.mean([train_eval(A, y, tr, te, depth, act, g, seed=s)
                            for s in range(SEEDS)]) for g in GAMMAS]
            i = int(np.argmin(row)); best[(act, depth)] = (row[i], GAMMAS[i])
            print(f"  {act:<10s} " + "  ".join(f"{v:5.2f} " for v in row)
                  + f"| {row[i]:5.2f}  g={GAMMAS[i]:g}", flush=True)

    print("\n" + "=" * 76)
    print("VERDICTS (min_gamma values)")
    print("=" * 76)
    print("  depth   threshold   crossing   tanh")
    for depth in (1, 2, 3, 4):
        t, c, h = (best[(a, depth)][0] for a in ACTS)
        print(f"  {depth:5d}   {t:9.2f}   {c:8.2f}   {h:4.2f}")
    c1 = best[("crossing", 1)][0]
    print("\n (i) does the CROSSING still collapse with depth once gamma is tuned?")
    for depth in (2, 3, 4):
        c = best[("crossing", depth)][0]
        print(f"     depth{depth}: {c:.2f}  ({'collapsed' if c > 0.9 else 'HEALTHY'}"
              f"; vs depth1 {c1:.2f})")
    print("\n (ii) single layer: bump vs tanh, BOTH gain-tuned")
    print(f"     crossing {best[('crossing',1)][0]:.2f} (g={best[('crossing',1)][1]:g})"
          f"  vs tanh {best[('tanh',1)][0]:.2f} (g={best[('tanh',1)][1]:g})"
          f"  -> {'bump wins' if best[('crossing',1)][0] < best[('tanh',1)][0] else 'tanh wins/ties'}")
    print("\n (iii) does the THRESHOLD (monotone) still rescue deep parity?")
    for depth in (2, 3, 4):
        t = best[("threshold", depth)][0]; h = best[("tanh", depth)][0]
        print(f"     depth{depth}: threshold {t:.2f} vs tanh {h:.2f}")


if __name__ == "__main__":
    main()
