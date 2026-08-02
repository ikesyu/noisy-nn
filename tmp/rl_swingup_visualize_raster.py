"""rl_swingup_visualize_raster (旧 rl_ppo_nowall_raster) -- raster-plot video of the NNN
populations while a trained swing-up policy performs one greedy episode.

For every step of one greedy from-bottom episode we record the T-averaged crossing
activity z of every hidden unit in the actor NNN, plus (when available) the value
network: the PPO critic V(s) or the SAC twin-Q min(Q1,Q2)(s,a).  The video shows the
cart-pole (left), the rasters filling in as the episode unfolds (right, units sorted
per layer by peak-activity time), and the aligned behavior traces.

NOISE-SOURCE AGNOSTIC: playback is greedy (a = mu), identical for external-sigma_e and
internal-temperature checkpoints.  Checkpoints carrying a noise FIELD (outgate sigma
fields, SAC v6.1 rho_T fields...) are replayed with the field re-applied per step
(mu depends on it), mirroring eval_strict / eval_from_bottom.

SAVE-FORMAT REQUIREMENTS (the .pt passed as --run):
  required  "checkpoints": [(step:int, snapshot:dict)]  -- snapshot from
            a2c_swingup._snapshot or a2c_nnncritic._snap (build_policy-compatible;
            fields/gate_k/gate_c/rho_mode optional inside)
  optional  "critic_net" + "critic_hidden" + "critic_t"   -- PPO/A2C value NNN
            -> V raster + V trace
  optional  "q1_net"/"q2_net" + "q_hidden" + "q_t"        -- SAC twin-Q NNNs
            -> Q1 raster + min(Q1,Q2)(s, a_greedy) trace
  (neither present -> actor-only layout, no value trace)
  optional  "wall_penalty", "x_barrier" -- terminal-wall env shaping for playback
All current drivers (rl_ppo_external_swingup, rl_ppo_itemp_swingup,
rl_sac_itemp_swingup) write this format.

    .venv/bin/python tmp/rl_swingup_visualize_raster.py [--run tmp/out/swingup_itemp_outconst_s0.pt]
Output: tmp/out/rl_swingup_raster.gif  (animation; --out to override)
        tmp/out/<out-stem>.png         (static full-episode figure)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from rl.envs_swingup import CartPoleSwingUp
from rl.a2c_swingup import build_policy, _set_field
from rl.critic import NNNCritic

OUT = Path(__file__).resolve().parent / "out"
X_THR = 4.0


def _build_value_net(data):
    """Returns (kind, net(s)) for whichever value network the run saved:
    ("critic", NNNCritic) | ("twinq", (QNNN, QNNN)) | ("none", None)."""
    if "critic_net" in data:
        critic = NNNCritic(obs_dim=5, hidden=int(data["critic_hidden"]),
                           t=int(data["critic_t"]))
        critic.net.load_state_dict(data["critic_net"])
        critic.eval()
        return "critic", critic
    if "q1_net" in data:
        from rl.sac import QNNN
        qs = []
        for key in ("q1_net", "q2_net"):
            q = QNNN(5, hidden=int(data["q_hidden"]), t=int(data["q_t"]))
            q.net.load_state_dict(data[key])
            q.eval()
            qs.append(q)
        return "twinq", tuple(qs)
    return "none", None


def record_episode(data, horizon=500, seed=0, ck_index=-1):
    state = data["checkpoints"][ck_index][1]
    policy, mean, std = build_policy(state)
    vkind, vnet = _build_value_net(data)
    env = CartPoleSwingUp(horizon=horizon, random_start=False, seed=seed,
                          force_mag=state["force_mag"], x_threshold=X_THR,
                          continuous=True, wall_mode="end",
                          wall_penalty=data.get("wall_penalty", 3.0),
                          x_barrier=data.get("x_barrier", 0.5))
    obs, _ = env.reset(seed=seed)
    n_alayers = len(policy.crossings)
    rec = {"az": [[] for _ in range(n_alayers)], "cz": None, "cos": [], "x": [],
           "theta": [], "a": [], "v": None, "wall_hits": 0, "vkind": vkind}
    if vkind != "none":
        rec["v"] = []
        n_clayers = len((vnet if vkind == "critic" else vnet[0]).crossings)
        rec["cz"] = [[] for _ in range(n_clayers)]
    for _ in range(horizon):
        if policy._opt_fields is not None:      # field-dependent mu: re-apply per step
            _set_field(policy, policy._opt_fields, float(obs[2]),
                       policy._gate_k, policy._gate_c,
                       policy._rho_mode, policy._sigma0, policy._h0)
        on = torch.clamp((torch.tensor(obs, dtype=torch.float32) - mean) / std, -5, 5)
        step = policy.rollout_step(on.unsqueeze(0), greedy=True)
        a_norm = float(step.action.item()) / state["force_mag"]
        for l in range(len(rec["az"])):
            rec["az"][l].append(step.z[l][0].mean(dim=0).numpy())   # [H] mean over T
        if vkind == "critic":
            v, cstep = vnet.value_step(on.unsqueeze(0))
        elif vkind == "twinq":
            sa = torch.cat([on, torch.tensor([a_norm], dtype=torch.float32)]
                           ).unsqueeze(0)
            q1v, cstep = vnet[0].q_step(sa)                          # raster from Q1
            v = float(torch.minimum(q1v, vnet[1].q_eval(sa)).item())
        if vkind != "none":
            rec["v"].append(v)
            for l in range(len(rec["cz"])):
                rec["cz"][l].append(cstep.z[l][0].mean(dim=0).numpy())
        rec["a"].append(a_norm)
        obs, r, te, tr, info = env.step(float(step.action.item()))
        rec["wall_hits"] += int(bool(info.get("wall", False)))
        rec["cos"].append(env.cos_theta())
        rec["x"].append(float(env.state[0]))
        rec["theta"].append(float(env.state[2]))
        if te or tr:
            break
    rec["az"] = [np.stack(v, axis=1) for v in rec["az"]]            # [H, steps]
    if rec["cz"] is not None:
        rec["cz"] = [np.stack(v, axis=1) for v in rec["cz"]]
    return rec


def sort_by_peak(z):
    return z[np.argsort(np.argmax(z, axis=1)), :]


def build_rasters(rec):
    """Per-network raster [units, steps], layers stacked, units sorted by peak time."""
    A = np.concatenate([sort_by_peak(z) for z in rec["az"]], axis=0)
    C = (np.concatenate([sort_by_peak(z) for z in rec["cz"]], axis=0)
         if rec["cz"] is not None else None)
    return A, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(OUT / "swingup_itemp_outconst_s0.pt"))
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default="rl_swingup_raster.gif")
    ap.add_argument("--ck-index", type=int, default=-1)
    ap.add_argument("--torch-seed", type=int, default=0)
    ap.add_argument("--tries", type=int, default=1,
                    help="record N episodes (the NNN ensemble mean makes even greedy "
                         "episodes stochastic) and keep the one with the best hold")
    args = ap.parse_args()

    data = torch.load(args.run, weights_only=False)
    torch.manual_seed(args.torch_seed)
    best = None
    for k in range(args.tries):
        r = record_episode(data, horizon=args.horizon, seed=args.seed,
                           ck_index=args.ck_index)
        full = len(r["cos"]) == args.horizon
        tail = float((np.array(r["cos"])[-100:] > 0.9).mean()) if full else -1.0
        score = (tail, float(np.mean(r["cos"])))
        print(f"  try {k}: steps {len(r['cos'])}, walls {r['wall_hits']}, "
              f"last100_up {max(tail, 0.0):.3f}")
        if best is None or score > best[0]:
            best = (score, r)
    rec = best[1]
    n = len(rec["cos"])
    A, C = build_rasters(rec)
    has_v = rec["v"] is not None
    vlab = "V (std. units)" if rec["vkind"] == "critic" else "min Q (scaled)"
    tail_up = float((np.array(rec["cos"])[-100:] > 0.9).mean()) if n == args.horizon else 0.0
    print(f"episode: {n} steps, wall contacts {rec['wall_hits']}, "
          f"max|x| {max(abs(v) for v in rec['x']):.2f}, last100_up {tail_up:.3f}, "
          f"value net: {rec['vkind']}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.animation import FuncAnimation, PillowWriter

    def int_ticks(lo, hi, max_n=5):
        ticks = np.arange(math.ceil(lo), math.floor(hi) + 1)
        return ticks[::2] if len(ticks) > max_n else ticks

    plt.rcParams.update({
        "font.size":       13,
        "axes.titlesize":  14,
        "axes.labelsize":  13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })

    nA1 = rec["az"][0].shape[0]                          # layer-1 size (divider line)
    a_title = f"Actor NNN ({' + '.join(str(z.shape[0]) for z in rec['az'])} units)"
    vA = np.percentile(A, 98)
    if has_v:
        nC1 = rec["cz"][0].shape[0]
        c_title = (f"{'Critic' if rec['vkind'] == 'critic' else 'Q1'} NNN "
                   f"({' + '.join(str(z.shape[0]) for z in rec['cz'])} units)")
        vC = np.percentile(C, 98)

    def setup(fig):
        gs = fig.add_gridspec(3, 2, width_ratios=[0.9, 1.35],
                              height_ratios=[1.35, 1.0, 0.75],
                              hspace=0.5, wspace=0.26,
                              left=0.075, right=0.97, top=0.95, bottom=0.1)
        axc = fig.add_subplot(gs[0, 0])                 # cart-pole
        axt = fig.add_subplot(gs[1:, 0])                # behavior traces
        if has_v:
            axa = fig.add_subplot(gs[0:2, 1])           # actor raster
            axk = fig.add_subplot(gs[2, 1], sharex=axa)  # value-net raster
        else:
            axa = fig.add_subplot(gs[:, 1])             # actor raster, full column
            axk = None
        return axc, axt, axa, axk

    fig = plt.figure(figsize=(12.0, 7.0), dpi=85)
    axc, axt, axa, axk = setup(fig)

    # cart-pole
    axc.set_xlim(-X_THR - 0.8, X_THR + 0.8)
    axc.set_ylim(-1.45, 1.7)
    axc.set_aspect("equal")
    axc.axis("off")
    axc.plot([-X_THR - 0.6, X_THR + 0.6], [0, 0], color="0.6", lw=1)
    for sx in (-X_THR, X_THR):
        axc.plot([sx, sx], [-0.28, 0.28], color="#c23b3b", lw=3)
    axc.plot(0, 1.0, marker="*", color="goldenrod", markersize=11, zorder=1)
    cart = Rectangle((-0.2, -0.1), 0.4, 0.2, color="0.2", zorder=3)
    axc.add_patch(cart)
    (pole,) = axc.plot([], [], color="C1", lw=4, zorder=4)
    (bob,) = axc.plot([], [], "o", color="C3", markersize=7, zorder=5)
    head = axc.text(0.5, 1.02, "", transform=axc.transAxes, ha="center",
                    fontsize=14, family="monospace")

    # behavior traces (all dimensionless: cos(theta), a in [-1,1], value standardized)
    axt.set_xlim(0, n)
    axt.set_xticks([0, n // 2, n])
    axt.set_xlabel("episode step")
    axt.grid(alpha=0.25)
    axt.axhline(0.0, color="0.85", lw=0.8)
    (l_cos,) = axt.plot([], [], color="C2", lw=1.3, label="cos(theta)")
    (l_act,) = axt.plot([], [], color="C0", lw=0.9, label="action a")
    l_val = None
    lo, hi = -1.1, 1.1
    if has_v:
        (l_val,) = axt.plot([], [], color="C4", lw=1.1, label=vlab)
        lo = min(lo, np.min(rec["v"]) - 0.2)
        hi = max(hi, np.max(rec["v"]) + 0.2)
    axt.set_ylim(lo, hi)
    axt.set_yticks(int_ticks(lo, hi))
    axt.legend(loc="lower right", ncol=1, framealpha=0.85)
    cur_t = axt.axvline(0, color="#c23b3b", lw=1.0)

    def raster(ax, Z, vmax, title, n_top):
        im = ax.imshow(np.full_like(Z, np.nan), aspect="auto", origin="lower",
                       cmap="Greys", vmin=0.0, vmax=vmax,
                       extent=(0, n, 0, Z.shape[0]), interpolation="nearest")
        ax.axhline(n_top, color="C0", lw=0.8, ls=":")
        ax.set_ylabel("unit")
        ax.set_title(title)
        ax.set_xticks([0, n // 2, n])
        ax.set_yticks([0, n_top, Z.shape[0]])
        cur = ax.axvline(0, color="#c23b3b", lw=1.0)
        return im, cur

    imA, curA = raster(axa, A, vA, a_title, nA1)
    cursors = [cur_t, curA]
    imK = None
    if has_v:
        imK, curK = raster(axk, C, vC, c_title, nC1)
        axk.set_xlabel("episode step")
        plt.setp(axa.get_xticklabels(), visible=False)
        cursors.append(curK)
    else:
        axa.set_xlabel("episode step")

    ks = list(range(0, n, args.stride))
    if ks[-1] != n - 1:
        ks.append(n - 1)
    ks.extend([n - 1] * int(args.fps * 1.0))

    Amask = np.full_like(A, np.nan)
    Cmask = np.full_like(C, np.nan) if has_v else None

    def _upd(fi):
        k = ks[fi]
        x, th, c = rec["x"][k], rec["theta"][k], rec["cos"][k]
        tipx, tipy = x + math.sin(th), math.cos(th)
        cart.set_xy((x - 0.2, -0.1))
        pole.set_data([x, tipx], [0.05, tipy + 0.05])
        bob.set_data([tipx], [tipy + 0.05])
        pole.set_color("C2" if c > 0.9 else "C1")
        head.set_text(f"step {k + 1:3d}/{n}  cos {c:+.2f}  "
                      f"walls {rec['wall_hits']}")
        for line, key in ((l_cos, "cos"), (l_act, "a"), (l_val, "v")):
            if line is not None:
                line.set_data(range(k + 1), rec[key][:k + 1])
        Amask[:, :k + 1] = A[:, :k + 1]
        imA.set_data(Amask)
        if has_v:
            Cmask[:, :k + 1] = C[:, :k + 1]
            imK.set_data(Cmask)
        for cur in cursors:
            cur.set_xdata([k, k])
        return ()

    anim = FuncAnimation(fig, _upd, frames=len(ks), interval=1000 / args.fps)
    out = OUT / args.out
    anim.save(out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"saved {out}  ({len(ks)} frames)")

    # static full-episode figure
    fig = plt.figure(figsize=(12.0, 7.0), dpi=120)
    axc, axt, axa, axk = setup(fig)
    axc.axis("off")
    axc.plot(rec["x"], np.cos(rec["theta"]), lw=0.8, color="0.4")
    axc.plot(rec["x"][0], math.cos(rec["theta"][0]), "o", color="C0", label="start")
    axc.plot(rec["x"][-1], math.cos(rec["theta"][-1]), "o", color="C2", label="end")
    for sx in (-X_THR, X_THR):
        axc.axvline(sx, color="#c23b3b", lw=2)
    axc.set_title("tip trajectory (x vs cos(theta)); red = stoppers")
    axc.legend(loc="lower right")
    axt.set_xlim(0, n)
    axt.set_xticks([0, n // 2, n])
    axt.set_yticks(int_ticks(lo, hi))
    axt.grid(alpha=0.25)
    axt.set_xlabel("episode step")
    axt.plot(rec["cos"], color="C2", lw=1.2, label="cos(theta)")
    axt.plot(rec["a"], color="C0", lw=0.8, label="action a")
    if has_v:
        axt.plot(rec["v"], color="C4", lw=1.0, label=vlab)
    axt.legend(loc="lower right")
    panels = [(axa, A, vA, a_title, nA1)]
    if has_v:
        panels.append((axk, C, vC, c_title, nC1))
    for ax, Z, vmax, title, ntop in panels:
        ax.imshow(Z, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=vmax,
                  extent=(0, n, 0, Z.shape[0]), interpolation="nearest")
        ax.axhline(ntop, color="C0", lw=0.8, ls=":")
        ax.set_ylabel("unit")
        ax.set_title(title)
        ax.set_xticks([0, n // 2, n])
        ax.set_yticks([0, ntop, Z.shape[0]])
    (axk if has_v else axa).set_xlabel("episode step")
    png = OUT / (Path(args.out).stem + ".png")
    fig.savefig(png)
    plt.close(fig)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
