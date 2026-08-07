"""Caveat-1 test: is the lambda interior optimum a narrow-gamma accident (H1),
or universal once the effective gain is chosen properly (H2)?

Caveat 1 (from reservoir_lambda_local3b.py): lam=0.6 beat lam=0 by 5 sigma at
BN-gain init 1.0, but the effect VANISHED at gain init 4.0 (all 0.82-0.90).
Since that gain is only an INIT (BN affine weight is learnable), two hypotheses:

  H1 (bad)  : the lambda gain exists only in a narrow gamma band -> fragile.
  H2 (good) : with gamma chosen per lambda the gain is universal; gain0=4 merely
              lands in a bad basin.

Four experiments, run together:

 (1) DIAGNOSTIC - where does the learnable gamma actually go?  Measure the
     post-training gamma (gauge-invariant) for gain0 in {1,4}.  If gain0=1 walks
     to a lobe-matched gamma* and gain0=4 stays put, that alone supports H2.

 (2) DECISIVE - (lambda x gamma) grid with gamma FROZEN (not learnable), then
     compare min_gamma NRMSE(lambda).  Common-gamma comparison is unfair because
     the lobe depth of zbar_lambda differs by lambda (0.133 at 0.6, 0.5 at 1.0),
     so the best gamma differs by lambda too.  Readout weight decay is switched
     OFF here so "deeper lobe needs smaller readout weights" cannot masquerade
     as a lambda effect.
       min_gamma still favours mid lambda -> H2, caveat 1 dissolved.
       gain only in a narrow gamma band   -> H1, caveat 1 is real.

 (3) SCALING (the real prediction) - the "banister" (monotone step) has gradient
     support of width ~1/gamma in input units.  Raising gamma shortens the
     banister until it disappears -> that IS why gain0=4 killed the effect.
     Prediction: the critical gamma at which the lambda advantage dies should
     scale with the LOBE SPACING (wider spacing needs a wider banister = smaller
     gamma).  If the critical gamma tracks spacing, caveat 1 flips from "fragile
     result" into "quantitative fingerprint of the mechanism".

 (4) MECHANISM (independent of NRMSE) - where do the unit thresholds land?
     Banister hypothesis: at lam=1 (bump only, symmetric gradient) units cannot
     migrate and stay near init; at mid lambda they migrate ONTO the lobes.
     Measured as the mean distance from each unit's threshold location to the
     nearest true lobe centre, versus the same statistic at init.
"""
import numpy as np
import torch
import torch.nn as nn

import reservoir as R
from reservoir.readout import standardize_fit, nrmse

from reservoir_lambda_local import masks, DelayField, LambdaAct

LAMS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


# ---------------------------------------------------------------- shape utils
def lobe_depth(lam, n=20001):
    """peak - plateau of zbar_lambda: the 'bump on a step' decomposition."""
    Q = np.linspace(1e-4, 1 - 1e-4, n)
    z = (1 - lam) * Q + lam * 2 * Q * (1 - Q)
    return float(z.max() - (1 - lam))


# ---------------------------------------------------------------- model
class GaugeNet(nn.Module):
    """z = LambdaAct(gamma * (x - c)); gamma frozen or learnable.

    Parameterised as gamma*(x - c) rather than BN so that (i) the effective gain
    gamma is explicit and gauge-interpretable, and (ii) c is literally the unit's
    threshold LOCATION in input units -> directly comparable to lobe centres (4).
    """
    def __init__(self, Hin, H, lam, gamma0=1.0, learn_gamma=True, numT=100, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.c = nn.Parameter(torch.randn(H, Hin) * 0.5)      # threshold location
        self.w = nn.Parameter(torch.randn(H, Hin) / np.sqrt(Hin))
        self.log_gamma = nn.Parameter(torch.full((H,), float(np.log(gamma0))),
                                      requires_grad=learn_gamma)
        self.out = nn.Linear(H, 1)
        self.lam, self.numT = lam, numT

    def gamma(self):
        return self.log_gamma.exp()

    def forward(self, x):
        # p_k = gamma_k * w_k . (x - c_k)
        d = torch.einsum("bi,hi->bh", x, self.w) - (self.w * self.c).sum(-1)
        p = d * self.gamma()
        z = LambdaAct.apply(p, self.lam, 0.0, self.numT)
        return self.out(z).squeeze(-1)

    def thresh_loc(self):
        """1-D case: the input value where the unit's pre-activation crosses 0."""
        with torch.no_grad():
            return (self.c[:, 0]).cpu().numpy()


def train_eval(X_np, y, tr, te, H, lam, gamma0=1.0, learn_gamma=True, numT=100,
               steps=3000, bs=256, lr=3e-3, wd=0.0, seed=0, want_diag=False):
    mu, sd = standardize_fit(X_np[tr]); Xs = (X_np - mu) / sd
    X = torch.tensor(Xs, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    net = GaugeNet(X.shape[1], H, lam, gamma0=gamma0, learn_gamma=learn_gamma,
                   numT=numT, seed=seed)
    loc0 = net.thresh_loc().copy()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        b = torch.tensor(rng.choice(tr_idx, size=bs, replace=False))
        net.train(); opt.zero_grad()
        loss = ((net(X[b]) - yt[b]) ** 2).mean()
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pr = [net(X[torch.tensor(te_idx[i:i+1000])]).numpy()
              for i in range(0, len(te_idx), 1000)]
    e = nrmse(y[te_idx], np.concatenate(pr))
    if not want_diag:
        return e
    return e, dict(gamma=net.gamma().detach().cpu().numpy(),
                   loc0=loc0, loc1=net.thresh_loc(), mu=mu[0], sd=sd[0])


# ---------------------------------------------------------------- task
def lobe_task(T, lag=2, spacing=0.6, w=0.18, seed=0):
    """3 Gaussian lobes at {-spacing, 0, +spacing} of one lagged input."""
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    x = np.concatenate([np.zeros(lag), u[:T - lag]])
    centers = (-spacing, 0.0, spacing)
    y = sum(np.exp(-((x - c) / w) ** 2) for c in centers)
    return u, y - y.mean(), np.array(centers)


def get_xy(T, spacing, lag=2):
    u, y, centers = lobe_task(T, lag=lag, spacing=spacing)
    X1 = DelayField(H=6).run(u)[:, lag:lag+1]
    return X1, y, centers


# ---------------------------------------------------------------- experiments
def exp1_diagnostic(X1, y, tr, te, H=16):
    print("=" * 78)
    print("(1) DIAGNOSTIC: where does the LEARNABLE gamma go?  (H=16, 3 seeds)")
    print("    H2 predicts: gain0=1 walks toward a lobe-matched gamma*,")
    print("                 gain0=4 stays high (bad basin) -> effect dies there.")
    print(f"    {'lam':>5} {'g0':>4} {'gamma_final (mean/med)':>26} {'NRMSE':>7}")
    for lam in (0.0, 0.6, 1.0):
        for g0 in (1.0, 4.0):
            gs, es = [], []
            for s in range(3):
                e, d = train_eval(X1, y, tr, te, H, lam, gamma0=g0,
                                  learn_gamma=True, seed=s, want_diag=True)
                gs.append(d["gamma"]); es.append(e)
            g = np.concatenate(gs)
            print(f"    {lam:5.1f} {g0:4.1f} {g.mean():12.2f} /{np.median(g):9.2f} "
                  f"{np.mean(es):9.3f}", flush=True)


def exp2_decisive(X1, y, tr, te, H=16, gammas=(1, 2, 3, 5, 8, 12), seeds=3):
    print("=" * 78)
    print(f"(2) DECISIVE: (lambda x FROZEN gamma) grid, wd=0.  H={H}, {seeds} seeds")
    print("    Compare min_gamma per lambda (fair: best gamma differs by lambda).")
    print("    lobe depth of zbar_lam:", "  ".join(f"{l:.1f}:{lobe_depth(l):.3f}"
                                                   for l in LAMS))
    print("    gam\\lam " + "  ".join(f"{l:5.2f}" for l in LAMS))
    grid = np.zeros((len(gammas), len(LAMS)))
    for i, g in enumerate(gammas):
        for j, lam in enumerate(LAMS):
            grid[i, j] = np.mean([train_eval(X1, y, tr, te, H, lam, gamma0=g,
                                             learn_gamma=False, wd=0.0, seed=s)
                                  for s in range(seeds)])
        print(f"    {g:5.1f}  " + "  ".join(f"{v:5.2f}" for v in grid[i]), flush=True)
    best = grid.min(axis=0); arg = [gammas[k] for k in grid.argmin(axis=0)]
    print("    ---")
    print("    min_g  " + "  ".join(f"{v:5.2f}" for v in best))
    print("    argmin " + "  ".join(f"{v:5.1f}" for v in arg))
    jb = int(np.argmin(best))
    print(f"\n    => best lambda under min_gamma = {LAMS[jb]:.1f} "
          f"(NRMSE {best[jb]:.3f});  lam=0: {best[0]:.3f}, lam=1: {best[-1]:.3f}")
    if best[jb] < min(best[0], best[-1]) * 0.95 and 0 < jb < len(LAMS) - 1:
        print("    => INTERIOR optimum survives gamma optimisation  ==> H2 (caveat dissolved)")
    else:
        print("    => no interior optimum after gamma optimisation  ==> H1 (caveat real)")
    return grid, best


def exp3_scaling(tr, te, T, H=16, spacings=(0.4, 0.6, 0.9), seeds=2,
                 gammas=(1, 2, 3, 5, 8, 12)):
    print("=" * 78)
    print("(3) SCALING: does the critical gamma (where lam-advantage dies)")
    print("    track LOBE SPACING?  banister width ~ 1/gamma  =>  gamma_c ~ 1/spacing.")
    for sp in spacings:
        X1, y, _ = get_xy(T, sp)
        adv = []
        for g in gammas:
            e0 = np.mean([train_eval(X1, y, tr, te, H, 0.0, gamma0=g,
                                     learn_gamma=False, wd=0.0, seed=s) for s in range(seeds)])
            e6 = np.mean([train_eval(X1, y, tr, te, H, 0.6, gamma0=g,
                                     learn_gamma=False, wd=0.0, seed=s) for s in range(seeds)])
            adv.append((e0 - e6) / max(e0, 1e-9))
        # critical gamma = largest gamma where advantage still > 5%
        alive = [g for g, a in zip(gammas, adv) if a > 0.05]
        gc = max(alive) if alive else float("nan")
        print(f"    spacing={sp:4.2f} (sd-units ~{sp/np.std(np.random.default_rng(0).uniform(-1,1,10000)):.2f})"
              f"  adv(lam.6 vs lam0) per gamma: "
              + " ".join(f"{g}:{a*100:+5.1f}%" for g, a in zip(gammas, adv))
              + f"   -> gamma_c={gc}", flush=True)


def exp4_mechanism(X1, y, tr, te, centers, H=16, gamma=3.0, seeds=3):
    print("=" * 78)
    print("(4) MECHANISM: do unit thresholds MIGRATE onto the lobes?")
    print("    banister hypothesis: lam=1 (bump only, symmetric gradient) cannot")
    print("    migrate; mid lambda can.  Distance to nearest lobe centre (std units).")
    print(f"    {'lam':>5} {'dist@init':>10} {'dist@end':>9} {'improvement':>12}")
    for lam in (0.0, 0.6, 1.0):
        d0s, d1s = [], []
        for s in range(seeds):
            _, d = train_eval(X1, y, tr, te, H, lam, gamma0=gamma,
                              learn_gamma=False, wd=0.0, seed=s, want_diag=True)
            # thresholds live in standardised input units; map centres likewise
            cz = (centers - d["mu"]) / d["sd"]
            for loc, acc in ((d["loc0"], d0s), (d["loc1"], d1s)):
                acc.append(np.mean(np.min(np.abs(loc[:, None] - cz[None, :]), axis=1)))
        m0, m1 = np.mean(d0s), np.mean(d1s)
        print(f"    {lam:5.1f} {m0:10.3f} {m1:9.3f} {100*(m0-m1)/m0:11.1f}%", flush=True)


def main():
    T = 3000
    tr, te = masks(T)
    X1, y, centers = get_xy(T, spacing=0.6)
    exp1_diagnostic(X1, y, tr, te)
    exp2_decisive(X1, y, tr, te)
    exp4_mechanism(X1, y, tr, te, centers)
    exp3_scaling(tr, te, T)


if __name__ == "__main__":
    main()
