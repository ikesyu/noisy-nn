"""rl_ppo_external_swingup (旧 rl_ppo_nowall) -- PPO v4 (idea_rl.md §23.10) on the NO-STOPPER swing-up task.

The historical swing-up env has physical stoppers at |x| = x_threshold, and the learned
PPO v4 policy exploits them (measured: 64-88 wall-contact steps per greedy episode).
Here the task is made strict: wall_mode="end" terminates the episode with a penalty on
any stopper contact, plus a soft quadratic barrier beyond 70% of the track.  A successful
swing-up + balance must therefore be achieved WITHOUT ever touching the bounds.

Everything else is canonical PPO v4: fully-NNN actor-critic (cov_jac actor + learned NNN
critic, both persistent EMA mirrors + KP), noise-deadband clip, KL early stop,
sigma_e annealed 0.4 -> 0.2.

    .venv/bin/python tmp/rl_ppo_external_swingup.py [--updates 300] [--seed 0]
Output: tmp/out/swingup_ppo_nowall_s{seed}.pt  (checkpoints + critic + stats)
        tmp/out/ppo_nowall_curves.png          (training curves + eval summary)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from rl.ppo import train_ppo_nnn
from rl.envs_swingup import CartPoleSwingUp
from rl.a2c_swingup import build_policy

OUT = Path(__file__).resolve().parent / "out"

WALL_PENALTY = 3.0
X_BARRIER = 0.5
ALIVE_BONUS = 2.2   # survival must beat suicide-by-stopper (see envs_swingup.alive_bonus)


def eval_nowall(state, horizon=500, seeds=(0, 1, 2)):
    """Greedy from-bottom eval on the terminal-wall env.  Returns per-seed dicts with
    mean cos, frac_up, last100_up (0 if the episode died at a wall before the tail),
    wall-contact steps, max |x| and survival."""
    p, mean, std = build_policy(state)
    out = []
    for s in seeds:
        env = CartPoleSwingUp(horizon=horizon, random_start=False, seed=s,
                              force_mag=state["force_mag"], x_threshold=4.0,
                              continuous=True, wall_mode="end",
                              wall_penalty=WALL_PENALTY, x_barrier=X_BARRIER)
        obs, _ = env.reset(seed=s)
        cs, hits, xmax = [], 0, 0.0
        for _ in range(horizon):
            on = torch.clamp((torch.tensor(obs, dtype=torch.float32) - mean) / std, -5, 5)
            step = p.rollout_step(on.unsqueeze(0), greedy=True)
            obs, r, te, tr, info = env.step(float(step.action.item()))
            hits += int(bool(info.get("wall", False)))
            xmax = max(xmax, abs(float(env.state[0])))
            cs.append(env.cos_theta())
            if te or tr:
                break
        cs = np.array(cs)
        full = len(cs) == horizon
        out.append({"seed": s, "mean_cos": float(cs.mean()),
                    "frac_up": float((cs > 0.9).mean()),
                    "last100_up": float((cs[-100:] > 0.9).mean()) if full else 0.0,
                    "wall_hits": hits, "max_x": xmax, "survived": full})
    return out


def plot_curves(stats, evals, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    upd = [s["upd"] for s in stats]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    ax.plot(upd, [s["ret_step"] for s in stats], color="C0", lw=1.2)
    ax.set_xlabel("update")
    ax.set_ylabel("training reward / step")
    ax.set_title("training reward (exploration on)")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if evals:
        eu = [e["upd"] for e in evals]
        ax.plot(eu, [e["mean_cos"] for e in evals], "o-", color="C2",
                label="mean cos(theta)")
        ax.plot(eu, [e["last100_up"] for e in evals], "s-", color="C3",
                label="last100_up")
        ax.axhline(1.0, color="0.7", ls=":", lw=1)
        ax.legend()
    ax.set_xlabel("update (checkpoint)")
    ax.set_title("greedy eval from bottom (3 seeds, terminal-wall env)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(upd, [s["wall_hits"] for s in stats], color="C1", lw=1.0,
            label="training (exploration)")
    if evals:
        ax.plot([e["upd"] for e in evals], [e["wall_hits"] for e in evals],
                "s-", color="C3", label="greedy eval")
    ax.set_xlabel("update")
    ax.set_ylabel("wall contacts / update")
    ax.set_title("stopper contacts (must reach 0)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(upd, [s["r2"] for s in stats], color="C4", lw=1.0, label="value R$^2$")
    ax.plot(upd, [s["ep_len"] / 400.0 for s in stats], color="C5", lw=1.0,
            label="episode length / horizon")
    ax.set_xlabel("update")
    ax.set_ylim(-0.1, 1.15)
    ax.set_title("critic fit and episode survival")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("PPO v4 (fully-NNN actor-critic, no backprop) -- swing-up WITHOUT stoppers",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--sigma-end", type=float, default=0.2)   # v4 floor
    ap.add_argument("--bottom-frac", type=float, default=0.4)
    ap.add_argument("--top-frac", type=float, default=0.25)   # wall-free hold curriculum
    ap.add_argument("--sigma-start", type=float, default=0.4)
    ap.add_argument("--top-center", type=float, default=0.0)  # hold-centering shaping
    ap.add_argument("--lr-actor", type=float, default=0.01)
    ap.add_argument("--kl-target", type=float, default=0.02)
    ap.add_argument("--wall-mode", choices=("end", "stop"), default="end",
                    help="'stop': walls exist physically (proven v4 dynamics) but a "
                         "heavy per-step wall penalty makes the OPTIMAL policy wall-"
                         "free; 'end': touching a stopper terminates the episode. "
                         "Greedy eval is always the strict terminal-wall test.")
    ap.add_argument("--wall-penalty", type=float, default=None,
                    help="per-step (stop) / terminal (end) wall cost; default 2.5/3.0")
    ap.add_argument("--alive-bonus", type=float, default=None,
                    help="reward shift; default 0 (stop) / 2.2 (end, anti-suicide)")
    ap.add_argument("--lr-var-scale", action="store_true",
                    help="scale actor lr with (sig_e/sigma_start)^2 so sigma_e can "
                         "anneal below 0.2 without the v4 terminal collapse")
    ap.add_argument("--init-from", default=None,
                    help=".pt of an earlier run: warm-start policy/critic/norm from its "
                         "final checkpoint (curriculum phase 2)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    init_policy = init_critic = None
    if args.init_from:
        prev = torch.load(args.init_from, weights_only=False)
        init_policy = prev["checkpoints"][-1][1]
        init_critic = prev.get("critic_net")
        print(f"warm start from {args.init_from} "
              f"(checkpoint upd {prev['checkpoints'][-1][0]})")

    wall_penalty = (args.wall_penalty if args.wall_penalty is not None
                    else (2.5 if args.wall_mode == "stop" else WALL_PENALTY))
    alive_bonus = (args.alive_bonus if args.alive_bonus is not None
                   else (0.0 if args.wall_mode == "stop" else ALIVE_BONUS))
    policy, critic, norm, cks, hist, stats = train_ppo_nnn(
        seed=args.seed, updates=args.updates, horizon=args.horizon,
        sigma_explore=args.sigma_start, sigma_explore_end=args.sigma_end, verbose=True,
        bottom_frac=args.bottom_frac, top_frac=args.top_frac,
        lr_actor=args.lr_actor, kl_target=args.kl_target,
        wall_mode=args.wall_mode, wall_penalty=wall_penalty, x_barrier=X_BARRIER,
        alive_bonus=alive_bonus, top_center=args.top_center, fill_batch=True,
        init_policy=init_policy, init_critic=init_critic,
        lr_var_scale=args.lr_var_scale)
    save = {"checkpoints": cks, "hist": hist, "stats": stats, "seed": args.seed,
            "critic_net": {k: v.detach().clone()
                           for k, v in critic.net.state_dict().items()},
            "critic_hidden": critic.hidden, "critic_t": critic.t,
            "wall_penalty": wall_penalty, "x_barrier": X_BARRIER,
            "alive_bonus": alive_bonus, "top_center": args.top_center,
            "wall_mode": args.wall_mode}
    path = OUT / f"swingup_ppo_nowall_s{args.seed}.pt"
    torch.save(save, path)
    print(f"saved {path}")

    print("=== eval swing-up from BOTTOM (greedy, horizon 500, terminal walls) ===")
    evals = []
    for upd, st in cks:
        res = eval_nowall(st, horizon=500)
        agg = {"upd": upd,
               "mean_cos": float(np.mean([r["mean_cos"] for r in res])),
               "frac_up": float(np.mean([r["frac_up"] for r in res])),
               "last100_up": float(np.mean([r["last100_up"] for r in res])),
               "wall_hits": int(np.sum([r["wall_hits"] for r in res])),
               "max_x": float(np.max([r["max_x"] for r in res])),
               "survived": int(np.sum([r["survived"] for r in res]))}
        evals.append(agg)
        print(f"  upd {upd:4d}  mean cos {agg['mean_cos']:+.3f}  "
              f"frac_up {agg['frac_up']:.3f}  last100_up {agg['last100_up']:.3f}  "
              f"walls {agg['wall_hits']}  max|x| {agg['max_x']:.2f}  "
              f"survived {agg['survived']}/3", flush=True)
    save["evals"] = evals
    torch.save(save, path)
    if not evals:
        print("(no checkpoints -- updates < checkpoint_every)")
        return

    tails = np.array([e["last100_up"] for e in evals])
    late = tails[len(tails) // 2:]
    ok = [e for e in evals if e["last100_up"] >= 1.0 and e["wall_hits"] == 0]
    print(f"best last100_up {tails.max():.3f}   late-half mean {late.mean():.3f}   "
          f"checkpoints with (last100_up=1.0 AND zero wall contact): {len(ok)}")

    plot_curves(stats, evals, OUT / "ppo_nowall_curves.png")
    print(f"saved {OUT / 'ppo_nowall_curves.png'}")


if __name__ == "__main__":
    main()
