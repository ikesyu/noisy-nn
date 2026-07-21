"""rl_multimode_field -- continuous noise-field center learned by reward (§7.3, §19; 本丸).

Reward moves a continuous field centre mu_c[context] (a Gaussian recruitment bump) while
the body learns the behaviors.  Shows (1) reward breaks symmetry and self-organizes two
distinct continuous field centres, (2) the body reaches each target under its centre, and
(3) the DISTINCTIVE continuous-option result: sweeping the centre between the two learnt
values moves the endpoint smoothly through untrained intermediate behaviors (§14.2).

    .venv/bin/python tmp/rl_multimode_field.py --episodes 5000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from rl.multimode_field import train_field, endpoint_at_center


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--tau", type=float, default=0.15)
    ap.add_argument("--lr_field", type=float, default=0.15)
    ap.add_argument("--init_spread", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT = Path(__file__).resolve().parent / "out"
    OUT.mkdir(exist_ok=True)

    policy, env, mu_c, muc_hist, hist = train_field(
        seed=args.seed, H=args.H, sigma=args.sigma, episodes=args.episodes,
        tau=args.tau, lr_field=args.lr_field, init_spread=args.init_spread, verbose=True)
    print(f"\nfinal return(last200): {np.mean(hist[-200:]):.2f}")
    print(f"learnt field centres mu_c: {np.round(mu_c, 3)}")
    for ctx in (0, 1):
        m, s = endpoint_at_center(policy, env, mu_c[ctx], args.sigma, args.tau)
        ok = "OK" if abs(m - env.targets[ctx]) < 0.3 else "XX"
        print(f"  ctx {ctx} (target {env.targets[ctx]:+.0f})  centre={mu_c[ctx]:.3f} "
              f"-> endpoint {m:+.2f} +/- {s:.2f}  [{ok}]")

    # interpolation sweep: endpoint vs field centre
    cs = np.linspace(0, 1, 21)
    ends = [endpoint_at_center(policy, env, c, args.sigma, args.tau)[0] for c in cs]
    print("\ninterpolation (centre -> endpoint):")
    for c, e in zip(cs[::4], ends[::4]):
        print(f"  c={c:.2f} -> x={e:+.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # mu_c trajectory (symmetry breaking)
    axes[0].plot(muc_hist[:, 0], color="C0", label="ctx 0 centre")
    axes[0].plot(muc_hist[:, 1], color="C1", label="ctx 1 centre")
    axes[0].axhline(0.5, color="gray", ls=":", lw=1)
    axes[0].set_xlabel("episode"); axes[0].set_ylabel("field centre c")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("reward moves the field centres apart\n(from a symmetric start)")
    axes[0].legend(fontsize=8)
    # interpolation curve
    axes[1].plot(cs, ends, "o-", color="C2")
    for ctx, col in zip((0, 1), ("C0", "C1")):
        axes[1].axvline(mu_c[ctx], color=col, ls="--", lw=1,
                        label=f"learnt centre ctx{ctx}")
    axes[1].axhline(env.targets[0], color="gray", ls=":", lw=1)
    axes[1].axhline(env.targets[1], color="gray", ls=":", lw=1)
    axes[1].set_xlabel("field centre c (swept)"); axes[1].set_ylabel("endpoint x")
    axes[1].set_ylim(-1.6, 1.6)
    axes[1].set_title("continuous option: interpolating the centre\ngives smooth intermediate behavior")
    axes[1].legend(fontsize=8)
    fig.suptitle("Reward learns the noise field as a CONTINUOUS option coordinate (§7.3, §19)")
    fig.tight_layout()
    fig.savefig(OUT / "rl_multimode_field.png", dpi=130)
    print(f"saved {OUT / 'rl_multimode_field.png'}")


if __name__ == "__main__":
    main()
