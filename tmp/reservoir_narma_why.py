"""Why does the CROSSING win on NARMA at depth 1?  (follow-up to sec10.34)

sec10.34 found the first setting where the 2nd moment beats the 1st at matched
gamma: NARMA-10 + LDN field, depth 1 -> crossing 0.34 vs tanh 0.37, threshold 0.39.

The proposed mechanism was: NARMA contains PRODUCTS (u[t-x]*u[t-1], y*mean(y)),
and 2Q(1-Q) is a quadratic form near its centre, so the bump supplies products
and squares directly, whereas a monotone unit must build them by superposition.

That mechanism makes a sharp, falsifiable prediction: the crossing should win on
tasks whose nonlinearity is PRODUCT/SQUARE-like and lose on tasks that are
monotone in each lagged input.  Isolate this with synthetic tasks on the SAME
field, SAME depth, SAME gamma sweep:

  product : y = u(t-2) * u(t-5)          <- product of two lags   (predict: bump wins)
  square  : y = u(t-3)^2                 <- pure square           (predict: bump wins)
  sum     : y = u(t-2) + u(t-5)          <- linear control        (predict: no one wins)
  absum   : y = |u(t-2)| + |u(t-5)|      <- even but SEPARABLE    (bump should still help)
  monot   : y = tanh(2*u(t-2)) + tanh(2*u(t-5))  <- monotone, separable
                                                    (predict: monotone wins)
  narma10 : the real thing                        (reference)

If the bump's advantage tracks "product/square-ness" the mechanism is supported;
if it tracks something else (e.g. it wins everywhere at depth 1) it is not.
Also sweeps lambda so we can see whether the OPTIMUM is at lambda=1 or interior.
"""
import numpy as np

from reservoir.tasks import narma_x
from reservoir.fields import LDNField
from reservoir_lambda_local import masks
from reservoir_gamma_schedule import train_eval as ge_train   # per-layer gamma net

GAMMAS = (0.3, 0.5, 0.7, 1.0, 1.5, 2.0)
ACTS = ("threshold", "crossing", "tanh")
SEEDS = 3
T = 3000


def lags(u, T):
    return lambda L: np.concatenate([np.zeros(L), u[:T - L]])


def make_tasks():
    rng = np.random.default_rng(0)
    u = rng.uniform(0.0, 0.5, size=T)          # same drive statistics as NARMA
    g = lags(u, T)
    tasks = {
        "product": g(2) * g(5),
        "square":  g(3) ** 2,
        "sum":     g(2) + g(5),
        "absum":   np.abs(g(2) - 0.25) + np.abs(g(5) - 0.25),
        "monot":   np.tanh(4 * (g(2) - 0.25)) + np.tanh(4 * (g(5) - 0.25)),
    }
    out = {k: (u, v - v.mean()) for k, v in tasks.items()}
    un, yn = narma_x(T, 10, seed=0)
    out["narma10"] = (un, yn - yn.mean())
    return out


def main():
    tr, te = masks(T)
    tasks = make_tasks()
    print("=" * 78)
    print(f"Why does the bump win on NARMA depth1?  LDN field, depth 1, "
          f"{SEEDS} seeds, min over gamma")
    print("  mechanism claim: 2Q(1-Q) is quadratic near centre -> supplies "
          "PRODUCTS/SQUARES directly")
    print("=" * 78)
    print(f"  {'task':<9} {'threshold':>10} {'crossing':>9} {'tanh':>7} "
          f"| {'winner':>10}  {'bump adv vs best monotone':>10}")
    for name, (u, y) in tasks.items():
        A = LDNField(H=48, theta=60.0).run(u)
        best = {}
        for act in ACTS:
            vals = [np.mean([ge_train(A, y, tr, te, 1, act, [g], seed=s)
                             for s in range(SEEDS)]) for g in GAMMAS]
            i = int(np.argmin(vals)); best[act] = (vals[i], GAMMAS[i])
        t, c, h = (best[a][0] for a in ACTS)
        mono = min(t, h)
        win = min(best, key=lambda a: best[a][0])
        adv = (mono - c) / max(mono, 1e-9) * 100
        print(f"  {name:<9} {t:10.3f} {c:9.3f} {h:7.3f} | {win:>10}  {adv:+9.1f}%",
              flush=True)


if __name__ == "__main__":
    main()
