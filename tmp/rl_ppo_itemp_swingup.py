"""rl_ppo_itemp_swingup (旧 rl_itemp_swingup) -- §25 Stage 1: INTERNAL exploration temperature on the
no-stopper swing-up (§23.13 setup).

The external sigma_e is removed: the executed action is one of the policy NNN's own
T readout samples, so the exploration temperature is the physical ensemble spread,
and the noise FIELD controls it.  Two arms:

  --cold 0.3   gated: hot field (sigma x1) while pumping, cold field (sigma x0.3)
               near upright, blended by g = sigmoid(6*cos(theta))  [the §25.3(b) claim]
  --cold 0     constant: internal noise at the hot level everywhere  [ablation:
               internalization alone, no context-dependent temperature]

Baseline for go/no-go: §23.13 external-sigma_e run (tail mean 0.52, full success ~1/12).

    .venv/bin/python tmp/rl_ppo_itemp_swingup.py --updates 300 --seed 0 --cold 0.3
Output: tmp/out/swingup_itemp_{tag}_s{seed}.pt, tmp/out/itemp_{tag}_curves.png
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
from rl.a2c_swingup import build_policy, _set_field
from rl_ppo_external_swingup import plot_curves, WALL_PENALTY, X_BARRIER, ALIVE_BONUS

OUT = Path(__file__).resolve().parent / "out"
SIGMA0 = 0.6
H = 128


def eval_strict(state, horizon=500, seeds=(0, 1, 2)):
    """Greedy from-bottom eval on the terminal-wall env, temperature field applied."""
    p, mean, std = build_policy(state)
    out = []
    for s in seeds:
        env = CartPoleSwingUp(horizon=horizon, random_start=False, seed=s,
                              force_mag=state["force_mag"], x_threshold=4.0,
                              continuous=True, wall_mode="end")
        obs, _ = env.reset(seed=s)
        cs, hits, xmax = [], 0, 0.0
        for _ in range(horizon):
            if p._opt_fields is not None:
                _set_field(p, p._opt_fields, float(obs[2]), p._gate_k, p._gate_c)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--hot", type=float, default=1.0)
    ap.add_argument("--cold", type=float, default=0.3,
                    help="cold-field sigma multiplier near upright; <=0 disables the "
                         "gate (constant hot field = internalization-only ablation)")
    ap.add_argument("--gate-k", type=float, default=6.0)
    ap.add_argument("--gate-c", type=float, default=0.0)
    ap.add_argument("--hot-out", type=float, default=0.0,
                    help="readout-unit noise while pumping (calibrated temperature "
                         "lever; the body-sigma field saturates).  >0 switches to the "
                         "sigma_out design; --cold-out sets the near-upright level")
    ap.add_argument("--cold-out", type=float, default=0.05)
    ap.add_argument("--draw", choices=("sample", "gauss"), default="sample",
                    help="gauss: magnitude-matched Gaussian execution (no structured "
                         "body component) -- decomposition Arm B")
    ap.add_argument("--fixed-var", type=float, default=0.0,
                    help=">0: constant score variance instead of per-state measured "
                         "-- decomposition Arm A")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    temp_fields = temp_out = None
    if args.hot_out > 0:
        gated = args.cold_out < args.hot_out
        temp_out = (args.hot_out, args.cold_out)
        tag = "outgate" if gated else "outconst"
        if args.draw == "gauss":
            tag = "outgauss"
        elif args.fixed_var > 0:
            tag = "outfixv"
    else:
        gated = args.cold > 0
        tag = "gate" if gated else "const"
        if gated:
            hot = [torch.full((H,), SIGMA0 * args.hot) for _ in range(2)]
            cold = [torch.full((H,), SIGMA0 * args.cold) for _ in range(2)]
            temp_fields = [hot, cold]                   # g->1 near top selects cold

    policy, critic, norm, cks, hist, stats = train_ppo_nnn(
        seed=args.seed, updates=args.updates, horizon=args.horizon,
        verbose=True, bottom_frac=0.5, top_frac=0.2,
        wall_mode="end", wall_penalty=WALL_PENALTY, x_barrier=X_BARRIER,
        alive_bonus=ALIVE_BONUS, fill_batch=True,
        internal_noise=True, temp_fields=temp_fields, temp_out=temp_out,
        gate_k=args.gate_k, gate_c=args.gate_c, draw_mode=args.draw,
        var_override=args.fixed_var if args.fixed_var > 0 else None)
    save = {"checkpoints": cks, "hist": hist, "stats": stats, "seed": args.seed,
            "critic_net": {k: v.detach().clone()
                           for k, v in critic.net.state_dict().items()},
            "critic_hidden": critic.hidden, "critic_t": critic.t,
            "wall_penalty": WALL_PENALTY, "x_barrier": X_BARRIER,
            "alive_bonus": ALIVE_BONUS, "internal_noise": True,
            "hot": args.hot, "cold": args.cold if gated else None,
            "temp_out": temp_out, "gate_k": args.gate_k, "gate_c": args.gate_c}
    path = OUT / f"swingup_itemp_{tag}_s{args.seed}.pt"
    torch.save(save, path)
    print(f"saved {path}")

    print(f"=== [{tag}] eval from BOTTOM (greedy, horizon 500, terminal walls) ===")
    evals = []
    for upd, st in cks:
        res = eval_strict(st, horizon=500)
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
    print(f"[{tag}] best last100_up {tails.max():.3f}   late-half mean {late.mean():.3f}"
          f"   (baseline §23.13: best 0.647, 12-episode tail mean 0.52)")
    plot_curves(stats, evals, OUT / f"itemp_{tag}_curves.png")
    print(f"saved {OUT / f'itemp_{tag}_curves.png'}")


if __name__ == "__main__":
    main()
