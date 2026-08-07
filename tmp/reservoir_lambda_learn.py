"""Is lambda LEARNABLE, and what makes it settle away from 0?

lambda mixes the two counters sec2.5 already produces:
    z_lambda = (1-lambda)*rate + lambda*flip,     rate = mean_t b_t,
                                                  flip = mean_t |b_{t+1}-b_t|
so  d z / d lambda = flip - rate  -- available at ZERO extra hardware cost, and
lambda can be per-unit.

Two forces act on it.

 (1) TASK LOSS pushes lambda -> 0.  With the effective gain gamma matched, the
     1st moment (monotone) beats the 2nd on both tasks tested (sec10.32, sec10.33).

 (2) ON-BRAND ("no noise => no output") pushes lambda -> 1, and EXACTLY so.
     As sigma -> 0 the unit becomes deterministic: b = 1[d>h] never fluctuates,
     so flip = 0 and rate = 1[d>h], giving

         z_lambda |_{sigma->0}  =  (1 - lambda) * 1[d > h]

     i.e. the residual noise-free activity is exactly (1-lambda) -- the PLATEAU
     of the "bump on a step" decomposition (sec2.7.1a).  Exact at sample level.
     So an on-brand penalty is LINEAR in lambda with constant gradient, and its
     second factor P(d>h) means the net can also satisfy it by pushing units
     below threshold (which costs function) -- a genuine trade-off, not a
     lambda-only freebie.

Experiment: per-unit learnable lambda, sweep the on-brand weight beta, and watch
lambda* move from 0 towards 1.  gamma is swept too (lesson of sec10.33: never
compare activation shapes at an unmatched effective gain) and the best gamma per
beta is reported.
"""
import numpy as np
import torch
import torch.nn as nn

import reservoir as R
from reservoir.readout import standardize_fit, nrmse

from reservoir.moment import masks, SignField, parity_task, _Phi, _phi

EPS_FREE = 0.05          # sigma used to smooth the sigma->0 limit for the penalty


class RateAct(torch.autograd.Function):
    """forward = MC mean of b_t ; backward = analytic density phi(p)."""
    @staticmethod
    def forward(ctx, p, numT):
        n = torch.randn(numT, *p.shape)
        b = ((p.unsqueeze(0) + n) > 0).float()
        ctx.save_for_backward(_phi(p))
        return b.mean(0)
    @staticmethod
    def backward(ctx, g):
        (ph,) = ctx.saved_tensors
        return g * ph, None


class FlipAct(torch.autograd.Function):
    """forward = MC mean of |b_{t+1}-b_t| ; backward = 2(1-2Phi)phi."""
    @staticmethod
    def forward(ctx, p, numT):
        n = torch.randn(numT, *p.shape)
        b = ((p.unsqueeze(0) + n) > 0).float()
        ctx.save_for_backward(_Phi(p), _phi(p))
        return (b - b.roll(-1, 0)).abs().mean(0)
    @staticmethod
    def backward(ctx, g):
        P, ph = ctx.saved_tensors
        return g * 2.0 * (1 - 2 * P) * ph, None


class LamNet(nn.Module):
    """per-unit learnable lambda = sigmoid(rho); frozen gamma (matched)."""
    def __init__(self, Hin, H, depth, gamma, numT=64, lam0=0.5, seed=0):
        super().__init__(); torch.manual_seed(seed)
        d = [Hin] + [H] * depth
        self.ls = nn.ModuleList([nn.Linear(d[i], d[i + 1]) for i in range(depth)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(d[i + 1], affine=False) for i in range(depth)])
        self.cs = nn.ParameterList([nn.Parameter(torch.zeros(d[i + 1])) for i in range(depth)])
        r0 = float(np.log(lam0 / (1 - lam0)))
        self.rhos = nn.ParameterList([nn.Parameter(torch.full((d[i + 1],), r0))
                                      for i in range(depth)])
        self.out = nn.Linear(d[-1], 1)
        self.gamma, self.numT = gamma, numT

    def lams(self):
        return [torch.sigmoid(r) for r in self.rhos]

    def forward(self, x, want_free=False):
        free = []
        for L, bn, c, rho in zip(self.ls, self.bns, self.cs, self.rhos):
            p = (bn(L(x)) - c) * self.gamma
            lam = torch.sigmoid(rho)
            z = (1 - lam) * RateAct.apply(p, self.numT) + lam * FlipAct.apply(p, self.numT)
            if want_free:
                # sigma -> 0 limit: (1-lam) * 1[p>0], smoothed for differentiability
                free.append(((1 - lam) * _Phi(p / EPS_FREE)).mean())
            x = z
        y = self.out(x).squeeze(-1)
        return (y, torch.stack(free).mean()) if want_free else y


def run(A, y, tr, te, depth, gamma, beta, numT=64, H=48, steps=900, bs=256,
        lr=3e-3, lam0=0.5, seed=0):
    mu, sd = standardize_fit(A[tr]); As = (A - mu) / sd
    X = torch.tensor(As, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    net = LamNet(A.shape[1], H, depth, gamma, numT=numT, lam0=lam0, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        b = torch.tensor(rng.choice(tr_idx, size=bs, replace=False))
        net.train(); opt.zero_grad()
        pred, free = net(X[b], want_free=True)
        loss = ((pred - yt[b]) ** 2).mean() + beta * free
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pr = [net(X[torch.tensor(te_idx[i:i+1000])]).numpy()
              for i in range(0, len(te_idx), 1000)]
        e = nrmse(y[te_idx], np.concatenate(pr))
        lam = torch.cat([l.detach() for l in net.lams()]).numpy()
        _, free = net(X[torch.tensor(te_idx[:1000])], want_free=True)
    return e, float(lam.mean()), float(lam.std()), float(free)


def main():
    T = 3000; u, y = parity_task(T); tr, te = masks(T)
    A = SignField(H=32, gain=8.0).run(u)
    DEPTH, SEEDS = 2, 3
    GAMMAS = (0.5, 1.0, 2.0)
    BETAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)
    print("=" * 78)
    print(f"LEARNABLE per-unit lambda, parity depth={DEPTH}, {SEEDS} seeds, init lam=0.5")
    print("  beta = weight on the on-brand penalty  (1-lam)*P(d>h)  [= noise-free activity]")
    print("  task loss pushes lam->0 ; on-brand pushes lam->1")
    print("=" * 78)
    print(f"  {'beta':>5}  {'gamma':>5}  {'NRMSE':>6}  {'mean lam':>9}  {'sd lam':>7}  {'free act':>9}")
    for beta in BETAS:
        rows = []
        for g in GAMMAS:
            r = [run(A, y, tr, te, DEPTH, g, beta, seed=s) for s in range(SEEDS)]
            rows.append((np.mean([x[0] for x in r]), np.mean([x[1] for x in r]),
                         np.mean([x[2] for x in r]), np.mean([x[3] for x in r]), g))
        # report the gamma that minimises the FULL objective actually optimised
        k = int(np.argmin([r[0] + beta * r[3] for r in rows]))
        e, lm, ls, fr, g = rows[k]
        print(f"  {beta:5.2f}  {g:5.1f}  {e:6.3f}  {lm:9.3f}  {ls:7.3f}  {fr:9.4f}", flush=True)

    print("\n  control: does lambda go to 0 from a HIGH start when beta=0?")
    for lam0 in (0.1, 0.5, 0.9):
        r = [run(A, y, tr, te, DEPTH, 1.0, 0.0, lam0=lam0, seed=s) for s in range(SEEDS)]
        print(f"    lam init {lam0:.1f} -> final mean lam {np.mean([x[1] for x in r]):.3f}"
              f"   NRMSE {np.mean([x[0] for x in r]):.3f}", flush=True)


if __name__ == "__main__":
    main()
