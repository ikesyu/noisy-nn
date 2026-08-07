"""Per-layer gamma schedule + NARMA re-check (follow-up to sec10.33).

sec10.33 established that the crossing (2nd moment / bump) does NOT collapse with
depth once the effective gain gamma = ||a||/sigma is matched; it just needs a
gentler gain (gamma* ~ 0.5 for depth 2-4 vs ~1 for depth 1) and it has a much
NARROWER usable gamma window than the monotone readings.

Two open items are closed here.

(A) PER-LAYER SCHEDULE.  sec10.33 used ONE gamma for every layer.  Since the
    residual handicap of the bump might be a per-layer information loss that
    compounds with depth, a geometric schedule gamma_l = gamma_0 * r^l is swept
    (r<1 gentler when deeper, r=1 uniform, r>1 sharper when deeper).  If some
    r != 1 beats the best uniform gamma, the deep bump improves further and the
    "2nd moment is simply worse" conclusion weakens.

    NOTE on the mechanism.  BatchNorm re-standardises every layer, so the bump's
    handicap is NOT a growing pre-activation scale -- it is INFORMATION loss: the
    bump is even-symmetric, so it maps +s and -s to the same output and discards
    sign.  Larger gamma = stronger many-to-one folding = more loss per layer.
    Small gamma keeps 2Q(1-Q) near its quadratic regime (mildly nonlinear), so
    more information survives the composition.  Under that reading the schedule
    should prefer r <= 1.  A measured r > 1 would refute it.

(B) NARMA.  sec10.33 was parity-on-a-sign-field only.  Re-run the gamma-matched
    activation comparison on NARMA-10 with the LDN field -- the setting of
    sec10.20, which had concluded "bump vs monotone ~= no difference".  That
    conclusion was itself reached without matching gamma.
"""
import numpy as np
import torch
import torch.nn as nn

import reservoir as R
from reservoir.tasks import narma_x
from reservoir.fields import LDNField
from reservoir.readout import standardize_fit, nrmse

from reservoir.moment import masks, LambdaAct, SignField, parity_task

ACTS = ("threshold", "crossing", "tanh")


class SchedNet(nn.Module):
    """Same as sec10.33's Net but gamma is a PER-LAYER list (frozen)."""
    def __init__(self, Hin, H, depth, act, gammas, numT=64, seed=0):
        super().__init__(); torch.manual_seed(seed)
        d = [Hin] + [H] * depth
        self.ls = nn.ModuleList([nn.Linear(d[i], d[i + 1]) for i in range(depth)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(d[i + 1], affine=False) for i in range(depth)])
        self.cs = nn.ParameterList([nn.Parameter(torch.zeros(d[i + 1])) for i in range(depth)])
        self.out = nn.Linear(d[-1], 1)
        self.act, self.gammas, self.numT = act, list(gammas), numT

    def _f(self, p, c, g):
        p = (p - c) * g
        if self.act == "tanh":
            return torch.tanh(p)
        lam = 0.0 if self.act == "threshold" else 1.0
        return LambdaAct.apply(p, lam, 0.0, self.numT)

    def forward(self, x):
        for L, bn, c, g in zip(self.ls, self.bns, self.cs, self.gammas):
            x = self._f(bn(L(x)), c, g)
        return self.out(x).squeeze(-1)


def train_eval(A, y, tr, te, depth, act, gammas, numT=64, H=48, steps=900,
               bs=256, lr=3e-3, seed=0):
    mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
    X = torch.tensor(As, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    net = SchedNet(A.shape[1], H, depth, act, gammas, numT=numT, seed=seed)
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


# ------------------------------------------------------------------ (A)
def exp_schedule(A, y, tr, te, depth=4, seeds=2):
    G0 = (0.5, 0.7, 1.0, 1.5)
    RS = (0.6, 0.8, 1.0, 1.25, 1.6)
    print("=" * 78)
    print(f"(A) PER-LAYER gamma SCHEDULE  gamma_l = g0 * r^l,  depth={depth}, "
          f"{seeds} seeds, parity")
    print("    r<1 = gentler when deeper (predicted better for the bump)")
    for act in ("crossing", "threshold"):
        print(f"\n  --- {act} ---")
        print("   g0\\r  " + "  ".join(f"r={r:<5g}" for r in RS))
        grid = np.zeros((len(G0), len(RS)))
        for i, g0 in enumerate(G0):
            for j, r in enumerate(RS):
                gam = [g0 * r ** l for l in range(depth)]
                grid[i, j] = np.mean([train_eval(A, y, tr, te, depth, act, gam, seed=s)
                                      for s in range(seeds)])
            print(f"   {g0:4.1f}  " + "  ".join(f"{v:5.2f}  " for v in grid[i]), flush=True)
        i, j = np.unravel_index(int(grid.argmin()), grid.shape)
        uni = grid[:, RS.index(1.0)].min()
        print(f"   best: g0={G0[i]} r={RS[j]} -> {grid[i, j]:.3f}   "
              f"best UNIFORM (r=1) -> {uni:.3f}   "
              f"{'schedule HELPS' if grid[i, j] < uni * 0.95 else 'no real gain from scheduling'}")


# ------------------------------------------------------------------ (B)
def exp_narma(seeds=2):
    GAMMAS = (0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0)
    T = 3000
    u, y = narma_x(T, 10, seed=0)
    y = y - y.mean()
    tr, te = masks(T)
    A = LDNField(H=48, theta=60.0).run(u)
    print("\n" + "=" * 78)
    print(f"(B) NARMA-10 + LDN field, gamma-matched activation comparison "
          f"({seeds} seeds)")
    print("    sec10.20 had concluded 'bump vs monotone ~= no difference' WITHOUT")
    print("    matching gamma; this re-checks it at each activation's own best gamma.")
    best = {}
    for depth in (1, 2, 3):
        print(f"\n  --- depth {depth} ---")
        print("   act        " + "  ".join(f"g={g:<5g}" for g in GAMMAS) + "| min_g  argmin")
        for act in ACTS:
            row = [np.mean([train_eval(A, y, tr, te, depth, act, [g] * depth, seed=s)
                            for s in range(seeds)]) for g in GAMMAS]
            i = int(np.argmin(row)); best[(act, depth)] = (row[i], GAMMAS[i])
            print(f"   {act:<10s} " + "  ".join(f"{v:5.2f}  " for v in row)
                  + f"| {row[i]:5.2f}  g={GAMMAS[i]:g}", flush=True)
    print("\n  NARMA verdict (min_gamma):")
    print("   depth   threshold   crossing   tanh")
    for depth in (1, 2, 3):
        t, c, h = (best[(a, depth)][0] for a in ACTS)
        print(f"   {depth:5d}   {t:9.2f}   {c:8.2f}   {h:4.2f}"
              + ("   <- bump competitive" if c <= min(t, h) * 1.05 else ""))


def main():
    T = 3000; u, y = parity_task(T); tr, te = masks(T)
    A = SignField(H=32, gain=8.0).run(u)
    exp_schedule(A, y, tr, te, depth=4)
    exp_narma()


if __name__ == "__main__":
    main()
