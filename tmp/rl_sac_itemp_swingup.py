"""rl_sac_itemp_swingup (旧 rl_sac_itemp) -- §25.7 SAC revisit: fully-NNN SAC v6 with internal exploration noise,
measured-variance accounting and (optionally) automatic entropy-temperature alpha.

Task: NO-STOPPER swing-up (§23.13 env: terminal walls + alive bonus + barrier) -- the
same benchmark the canonical NNN-PPO (§26) fully solves (12/12 tail=1.0, ~360k steps).
SAC v5 (§23.11) ran on the easier stop-wall task with external sigma_e; here:

  (i)   action sampling is INTERNAL (readout noise field sigma_out, §25.6);
  (ii)  every log pi / score / importance ratio uses the per-state MEASURED variance
        (the decisive §25.6 mechanism);
  (iii) --alpha-auto: dual ascent on alpha toward a target entropy -- Stage 3 (§25.4)
        realized in its natural SAC form (entropy is now a measured physical quantity).

    .venv/bin/python tmp/rl_sac_itemp_swingup.py --episodes 600 --seed 0 [--alpha-auto]
Output: tmp/out/swingup_sacit_{tag}_s{seed}.pt, tmp/out/sacit_{tag}_curves.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from rl.sac import train_sac_nnn
from rl_ppo_external_swingup import WALL_PENALTY, X_BARRIER, ALIVE_BONUS
from rl_ppo_itemp_swingup import eval_strict as eval_nowall   # field-aware strict eval

OUT = Path(__file__).resolve().parent / "out"


def plot_curves(stats, evals, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epi = [s["epi"] for s in stats]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(epi, [s["ret_step"] for s in stats], color="C0", lw=0.9)
    axes[0].set_title("training reward / step")
    axes[0].set_xlabel("episode")
    axes[0].grid(alpha=0.3)
    if evals:
        axes[1].plot([e["upd"] for e in evals], [e["last100_up"] for e in evals],
                     "s-", color="C3", label="last100_up")
        axes[1].plot([e["upd"] for e in evals], [e["mean_cos"] for e in evals],
                     "o-", color="C2", label="mean cos")
        axes[1].axhline(1.0, color="0.7", ls=":", lw=1)
        axes[1].legend(fontsize=8)
    axes[1].set_title("greedy eval (terminal walls)")
    axes[1].set_xlabel("episode (checkpoint)")
    axes[1].grid(alpha=0.3)
    axes[2].plot(epi, [s["alpha"] for s in stats], color="C4", lw=0.9, label="alpha")
    axes[2].plot(epi, [s["temp"] for s in stats], color="C1", lw=0.9,
                 label="temperature")
    if any(s.get("rho_T", 1.0) != 1.0 for s in stats):
        axes[2].plot(epi, [s.get("rho_T", 1.0) for s in stats], color="C6", lw=0.9,
                     label="rho_T (dial)")
    axes[2].set_title("entropy price / physical temperature")
    axes[2].set_xlabel("episode")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    fig.suptitle("SAC v6 (fully-NNN, internal noise, measured-variance accounting)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--alpha-auto", action="store_true")
    ap.add_argument("--h-target", type=float, default=0.0)
    ap.add_argument("--sigma-out", type=float, default=0.35)
    ap.add_argument("--temp-reg", action="store_true",
                    help="v6.1: close the loop on the physical temperature dial "
                         "(body field x sigma_out) toward --temp-target")
    ap.add_argument("--temp-target", type=float, default=0.35)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    tag = "auto" if args.alpha_auto else "fixa"
    if args.temp_reg:
        tag = "reg" + tag

    policy, qs, norm, cks, hist, stats = train_sac_nnn(
        seed=args.seed, episodes=args.episodes, horizon=args.horizon,
        actor_mode="replay", verbose=True,
        bottom_frac=0.5, top_frac=0.2,
        wall_mode="end", wall_penalty=WALL_PENALTY, x_barrier=X_BARRIER,
        alive_bonus=ALIVE_BONUS,
        internal_noise=True, sigma_out=args.sigma_out,
        alpha=args.alpha, alpha_auto=args.alpha_auto, h_target=args.h_target,
        temp_reg=args.temp_reg, temp_target=args.temp_target,
        checkpoint_every=50)
    save = {"checkpoints": cks, "hist": hist, "stats": stats, "seed": args.seed,
            "alpha_auto": args.alpha_auto, "h_target": args.h_target,
            "sigma_out": args.sigma_out,
            # twin-Q nets: consumed by rl_swingup_visualize_raster.py (value trace + Q raster)
            "q1_net": {k: v.detach().clone() for k, v in qs[0].net.state_dict().items()},
            "q2_net": {k: v.detach().clone() for k, v in qs[1].net.state_dict().items()},
            "q_hidden": qs[0].hidden, "q_t": qs[0].t,
            "wall_penalty": WALL_PENALTY, "x_barrier": X_BARRIER,
            "alive_bonus": ALIVE_BONUS}
    path = OUT / f"swingup_sacit_{tag}_s{args.seed}.pt"
    torch.save(save, path)
    print(f"saved {path}")

    print(f"=== [sacit-{tag}] eval from BOTTOM (greedy, horizon 500, terminal walls) ===")
    evals = []
    for epi, st in cks:
        res = eval_nowall(st, horizon=500)
        agg = {"upd": epi,
               "mean_cos": float(np.mean([r["mean_cos"] for r in res])),
               "last100_up": float(np.mean([r["last100_up"] for r in res])),
               "wall_hits": int(np.sum([r["wall_hits"] for r in res])),
               "survived": int(np.sum([r["survived"] for r in res]))}
        evals.append(agg)
        print(f"  epi {epi:4d}  mean cos {agg['mean_cos']:+.3f}  "
              f"last100_up {agg['last100_up']:.3f}  walls {agg['wall_hits']}  "
              f"survived {agg['survived']}/3", flush=True)
    save["evals"] = evals
    torch.save(save, path)
    if evals:
        tails = np.array([e["last100_up"] for e in evals])
        print(f"[sacit-{tag}] best last100_up {tails.max():.3f}   "
              f"late-half mean {tails[len(tails) // 2:].mean():.3f}   "
              f"(PPO §26: 1.000; SAC v5 on the easier stop-wall task: 1.000 at ~400k)")
    plot_curves(stats, evals, OUT / f"sacit_{tag}_curves.png")
    print(f"saved {OUT / f'sacit_{tag}_curves.png'}")


if __name__ == "__main__":
    main()
