"""Every drawing, animation, and file save for the neuromod benchmark.

Nothing here knows how the network was trained; everything takes plain arrays and
callables.  Figures are saved when `save_path` is given and shown otherwise, so a
headless batch run and an interactive look use the same code path.

Typography follows the repo's animation convention: no figure suptitle, sparse
ticks, one shared font scale.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from . import world

OUT_DIR = Path(__file__).resolve().parents[1] / "out"

FONT = {
    "font.size":       13,
    "axes.titlesize":  14,
    "axes.labelsize":  13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
}

MARKERS = {
    "food":    dict(marker="o", color="tab:green", label="food"),
    "threat":  dict(marker="^", color="tab:red",   label="threat"),
    "shelter": dict(marker="s", color="tab:blue",  label="shelter"),
}

LEGEND_HANDLES = [
    Line2D([0], [0], marker=MARKERS[k]["marker"], color="w",
           markerfacecolor=MARKERS[k]["color"], markeredgecolor="k",
           markersize=10, label=k)
    for k in world.CATEGORIES
]


def use_headless(save_path) -> None:
    """Switch to the Agg backend when the caller only wants files."""
    if save_path is not None:
        matplotlib.use("Agg", force=True)


def resolve_out(name: str, out_dir=None) -> Path:
    """Absolute path under `tmp/out` (or `out_dir`), creating the directory."""
    directory = Path(out_dir) if out_dir is not None else OUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def save_or_show(fig, save_path, dpi=140):
    """Save to `save_path` or show; returns the path when a file was written."""
    if save_path is None:
        plt.show()
        return None
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved {path}")
    return path


def draw_objects(ax, objs, food_strengths=None,
                 shelter_region_radius=world.SHELTER_REGION_RADIUS,
                 legend=True):
    """Draw the scene; depleted food fades, shelters are circular regions."""
    food = objs["food"]
    fs = food_strengths if food_strengths is not None else np.ones(food.shape[0])
    for j in range(food.shape[0]):
        if fs[j] > 0.0:
            ax.scatter(food[j, 0], food[j, 1], marker="o", s=110,
                       color="tab:green", edgecolor="k", zorder=5)
        else:
            ax.scatter(food[j, 0], food[j, 1], marker="o", s=45,
                       color="tab:green", edgecolor="none", alpha=0.22, zorder=4)
    thr = objs["threat"]
    ax.scatter(thr[:, 0], thr[:, 1], marker="^", s=110,
               color="tab:red", edgecolor="k", zorder=5)
    for c in objs["shelter"]:
        ax.add_patch(Circle((c[0], c[1]), shelter_region_radius, facecolor="tab:blue",
                            alpha=0.18, edgecolor="tab:blue", lw=1.2, zorder=2))
    ax.scatter(objs["shelter"][:, 0], objs["shelter"][:, 1], marker="s", s=45,
               color="tab:blue", edgecolor="k", zorder=5)
    if legend:
        ax.legend(handles=LEGEND_HANDLES, loc="upper left", fontsize=9)


def _square_axes(ax, title=None):
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)


def draw_loss_panel(ax, history):
    """Static training-loss curves, one per recruited behaviour."""
    for state in world.STATES:
        color = MARKERS[world.STATE_TO_FIELD[state]]["color"]
        ax.plot(history[state], color=color, lw=1.2, label=world.EPISODE_LABEL[state])
    ax.set_yscale("log")
    ax.set_title("Training loss (shared weights)")
    ax.set_xlabel("update step")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(alpha=0.3)


def pure_panels(predict, objects, fields, save_path=None, q_side=15):
    """The three recruited vector fields side by side.

    Same weights, same objects, same observation grid; only the noise field
    differs, so the panels differing at all is the addressing result.
    `predict(obs, field) -> [N, 2]` keeps this module free of model details.
    """
    use_headless(save_path)
    plt.rcParams.update(FONT)

    q_axis = np.linspace(-1.0, 1.0, q_side, dtype=np.float32)
    qx, qy = np.meshgrid(q_axis, q_axis)
    q_positions = np.stack([qx.ravel(), qy.ravel()], axis=1).astype(np.float32)
    q_obs = world.encode_observations(q_positions, objects)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for ax, cat in zip(axes, world.CATEGORIES):
        draw_objects(ax, objects)
        vq = predict(q_obs, fields[cat])
        ax.quiver(q_positions[:, 0], q_positions[:, 1], vq[:, 0], vq[:, 1],
                  color="0.25", angles="xy", scale=22, width=0.004)
        _square_axes(ax, f"'{cat}' field  ->  {world.FIELD_EPISODE[cat]}")
    fig.tight_layout()
    return save_or_show(fig, save_path)


def field_sheet(fields, hidden_dim, save_path=None):
    """The three noise fields drawn on the unit sheet, with a shared colour scale."""
    use_headless(save_path)
    plt.rcParams.update(FONT)
    side = int(round(np.sqrt(hidden_dim)))
    vmax = float(max(f.max() for f in fields.values()))

    fig, axes = plt.subplots(1, len(fields), figsize=(4.2 * len(fields), 4.0))
    for ax, (key, field) in zip(np.atleast_1d(axes), fields.items()):
        im = ax.imshow(field.numpy().reshape(side, side), origin="lower",
                       extent=[0, 1, 0, 1], cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(f"'{key}' field")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), label="noise std", shrink=0.85)
    return save_or_show(fig, save_path)


def animate(predict, objects, fields, history, alphas, *,
            demo_mode="scripted", layout="scripted", anim_frames=360,
            eat_radius=0.10, shelter_radius=0.08, food_respawn=False,
            speed_mode="learned", speed_gain=0.9, hunger_rate=0.006,
            need_rate=0.006, eat_amount=0.6, rest_frames=50,
            threat_gain=1.7, threat_range=0.40, neuromod_smoothing=0.12,
            threat_motion="moving", threat_speed=0.01, seed=0,
            dynamic=False, velocities=None, show_reference=False,
            target_gamma=2.0, velocity_smoothing=0.2, dt=0.04,
            hidden_dim=64, save_path=None, fps=20):
    """Closed-loop animation of the trained system.

    `demo_mode="scripted"` runs the reactive loop: the graded drives produce
    continuous, smoothed weights that BLEND the three fields, and the recruited
    network produces the movement.  `demo_mode="cycle"` sweeps the blend around
    the three states instead, which is the continuous-control view.

    Panels: world and trajectory, recruited vector field, the current noise field
    on the unit sheet, and the agent's speed.  The speed panel exists because with
    `speed_mode="learned"` the network's output magnitude carries the collapse
    that behaviour-level stochastic resonance is about.
    """
    use_headless(save_path)
    plt.rcParams.update(FONT)

    q_side = 13
    q_axis = np.linspace(-1.0, 1.0, q_side, dtype=np.float32)
    qx, qy = np.meshgrid(q_axis, q_axis)
    q_positions = np.stack([qx.ravel(), qy.ravel()], axis=1).astype(np.float32)
    side = int(round(np.sqrt(hidden_dim)))

    state = world.initialize_demo_state(objects, layout)
    if dynamic and velocities is not None:
        state["vels"] = {k: velocities[k].copy() for k in world.CATEGORIES}
    threats_move = (demo_mode == "scripted" and threat_motion == "moving")
    if threats_move:
        state["threat_vels"] = world.make_threat_velocities(
            state["objects"], threat_speed, seed)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.6))
    ax_env, ax_quiver = axes[0]
    ax_field, ax_speed = axes[1]

    vmax = float(max(f.max() for f in fields.values()))
    speed_hist: list[float] = []

    def update(frame):
        objs = state["objects"]
        if demo_mode == "cycle" and dynamic and "vels" in state:
            world.update_dynamic_objects(objs, state["vels"])

        if demo_mode == "scripted":
            if threats_move:
                world.step_threats(objs, state["threat_vels"],
                                   shelter_keepout=world.THREAT_KEEPOUT_RADIUS,
                                   bounds=world.THREAT_BOUNDS)
            world.advance_drives(state, shelter_radius, hunger_rate, need_rate,
                                 rest_frames)
            state["w"], state["goal"], _ = world.neuromod_weights(
                state["pos"], objs, state["hunger"], state["shelter_need"],
                state["goal"], state["w"], threat_gain, threat_range,
                neuromod_smoothing)
            weights = state["w"]
        else:
            phase = 2.0 * np.pi * frame / anim_frames
            weights = world.cyclic_weights(phase)

        from . import fields as fields_mod
        field = fields_mod.blend_fields(fields, weights, world.CATEGORIES)
        current_alpha = world.blend_alpha(weights, alphas)
        dom = int(np.argmax(weights))
        state_name = world.STATES[dom]
        field_name = world.STATE_TO_FIELD[state_name]
        episode_label = world.EPISODE_LABEL[state_name]

        obs = world.encode_observation(state["pos"], objs,
                                       food_strengths=state["food_strengths"])
        v_pred = predict(obs[None, :], field).ravel()

        resting = (demo_mode == "scripted" and state["rest"] > 0)
        speed = world.step_agent(state, v_pred, dt, speed_gain, mode=speed_mode,
                                 smoothing=velocity_smoothing, resting=resting)
        speed_hist.append(speed)

        ate = world.apply_food_depletion(
            state["pos"], objs, state["food_strengths"], eat_radius,
            respawn=(food_respawn or demo_mode == "scripted"))
        if ate:
            state["hunger"] = max(0.0, state["hunger"] - eat_amount)

        state["trail"].append(state["pos"].copy())
        state["trail"] = state["trail"][-160:]
        trail = np.array(state["trail"])
        n_left = int(np.sum(state["food_strengths"] > 0.0))
        n_food = state["food_strengths"].shape[0]

        # world + trajectory
        ax_env.clear()
        draw_objects(ax_env, objs, state["food_strengths"])
        if len(trail) > 1:
            ax_env.plot(trail[:, 0], trail[:, 1], "-", color="0.4", lw=1.5, alpha=0.85)
        ax_env.scatter(state["pos"][0], state["pos"][1], s=150, color="black", zorder=6)
        _square_axes(ax_env, f"{episode_label}   food {n_left}/{n_food}")

        # recruited vector field
        ax_quiver.clear()
        draw_objects(ax_quiver, objs, state["food_strengths"], legend=False)
        q_obs = world.encode_observations(q_positions, objs,
                                          food_strengths=state["food_strengths"])
        vq = predict(q_obs, field)
        ax_quiver.quiver(q_positions[:, 0], q_positions[:, 1], vq[:, 0], vq[:, 1],
                         color="0.25", angles="xy", scale=22, width=0.004)
        if show_reference:
            vref = world.make_mixed_behavior_targets(q_obs, current_alpha,
                                                     gamma=target_gamma)
            ax_quiver.quiver(q_positions[:, 0], q_positions[:, 1], vref[:, 0], vref[:, 1],
                             color="tab:orange", angles="xy", scale=22, width=0.003,
                             alpha=0.5)
        _square_axes(ax_quiver, "Recruited vector field"
                     if demo_mode == "scripted" else f"'{field_name}' field")

        # noise field on the unit sheet
        ax_field.clear()
        ax_field.imshow(field.numpy().reshape(side, side), origin="lower",
                        extent=[0, 1, 0, 1], cmap="magma", vmin=0.0, vmax=vmax)
        ax_field.set_title(f"Noise field   F={weights[0]:.2f} "
                           f"T={weights[1]:.2f} S={weights[2]:.2f}")
        ax_field.set_xticks([])
        ax_field.set_yticks([])

        # speed: the channel that carries a low-noise collapse
        ax_speed.clear()
        ax_speed.plot(speed_hist[-300:], color="0.2", lw=1.4)
        ax_speed.set_xlabel("frame")
        ax_speed.set_ylabel("|v| (network output)")
        ax_speed.set_ylim(0, 1.15)
        ax_speed.set_yticks([0, 0.5, 1.0])
        ax_speed.grid(alpha=0.3)
        ax_speed.set_title(f"speed {speed:.2f}   hunger {state['hunger']:.2f}   "
                           f"shelter need {state['shelter_need']:.2f}")
        return []

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=anim_frames, interval=1000 / fps,
                         blit=False, repeat=True)

    if save_path is None:
        plt.show()
        return anim
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {path}  ({anim_frames} frames)")
    return path


def loss_figure(history, save_path=None):
    """Standalone training-loss figure."""
    use_headless(save_path)
    plt.rcParams.update(FONT)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    draw_loss_panel(ax, history)
    fig.tight_layout()
    return save_or_show(fig, save_path)
