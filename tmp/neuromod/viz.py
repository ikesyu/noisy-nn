"""Every drawing, animation, and file save for the neuromod benchmark.

Nothing here knows how the network was trained; everything takes plain arrays and
callables.  Figures are saved when `save_path` is given and shown otherwise, so a
headless batch run and an interactive look use the same code path.

Typography follows the repo's animation convention: no figure suptitle, sparse
ticks, one shared font scale.
"""
from __future__ import annotations

import itertools
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
    """Static training-loss curves: per recruited behaviour, or whatever
    series the training protocol recorded (e.g. one 'blended' curve)."""
    for key, series in history.items():
        color = (MARKERS[world.STATE_TO_FIELD[key]]["color"]
                 if key in world.STATE_TO_FIELD else "0.3")
        label = world.EPISODE_LABEL.get(key, key)
        ax.plot(series, color=color, lw=1.2, label=label)
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
        im = ax.imshow(field.cpu().numpy().reshape(side, side), origin="lower",
                       extent=[0, 1, 0, 1], cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(f"'{key}' field")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), label="noise std", shrink=0.85)
    return save_or_show(fig, save_path)


def animate(predict, objects, fields, history, alphas, *,
            params: world.LoopParams = None,
            demo_mode="scripted", anim_frames=360,
            dynamic=False, velocities=None, show_reference=False,
            target_gamma=2.0, hidden_dim=64, save_path=None, fps=20):
    """Closed-loop animation of the trained system.

    `params` is THE loop configuration -- one `world.LoopParams`, the same
    object a headless `world.rollout` would take.  This function used to
    re-declare every loop knob as a keyword with its own defaults, and those
    defaults drifted from `LoopParams` (`speed_ref` was missing entirely, so
    the ANIMATED animal ran ~3x slower than the measured one; `threat_range`,
    `refuge_range` and `risk_hunger` had stale values).  A single shared
    dataclass is what guarantees the SR curve describes the animal on screen.
    None means the standard benchmark settings.

    `demo_mode="scripted"` runs the reactive loop: the graded drives produce
    continuous, smoothed weights that BLEND the three fields, and the recruited
    network produces the movement.  `demo_mode="cycle"` sweeps the blend around
    the three states instead, which is the continuous-control view.

    Panels: world and trajectory, recruited vector field, the current noise field
    on the unit sheet, and the agent's speed.  The speed panel exists because the
    network's output magnitude carries the collapse that behaviour-level
    stochastic resonance is about.
    """
    if params is None:
        params = world.LoopParams()
    use_headless(save_path)
    plt.rcParams.update(FONT)

    q_side = 13
    q_axis = np.linspace(-1.0, 1.0, q_side, dtype=np.float32)
    qx, qy = np.meshgrid(q_axis, q_axis)
    q_positions = np.stack([qx.ravel(), qy.ravel()], axis=1).astype(np.float32)
    side = int(round(np.sqrt(hidden_dim)))

    state = world.initialize_demo_state(objects)
    if dynamic and velocities is not None:
        state["vels"] = {k: velocities[k].copy() for k in world.CATEGORIES}
    threats_move = (demo_mode == "scripted" and params.threat_motion == "moving")
    if threats_move:
        state["threat_vels"] = world.make_threat_velocities(
            state["objects"], params.threat_speed, params.threat_seed)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.6))
    ax_env, ax_quiver = axes[0]
    ax_field, ax_speed = axes[1]

    vmax = float(max(f.max() for f in fields.values()))
    speed_hist: list[float] = []

    def update(frame):
        objs = state["objects"]
        if demo_mode == "cycle" and dynamic and "vels" in state:
            world.update_dynamic_objects(objs, state["vels"])

        rec = world.advance_frame(state, predict, fields, params,
                                  demo_mode=demo_mode, frame=frame,
                                  n_frames=anim_frames)
        weights, field, speed = rec["weights"], rec["field"], rec["speed"]
        field_name, episode_label = rec["field_name"], rec["label"]
        current_alpha = world.blend_alpha(weights, alphas)
        speed_hist.append(speed)
        del speed_hist[:-300]        # only the last 300 are drawn; endless-safe
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
        ax_field.imshow(field.cpu().numpy().reshape(side, side), origin="lower",
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
        ph = world.circadian_phase(state["clock"], params.circadian_period)
        clock_txt = (f"{'day' if ph < params.day_fraction else 'NIGHT'} "
                     f"{ph:.2f}   ")
        ax_speed.set_title(f"speed {speed:.2f}   hunger {state['hunger']:.2f}   "
                           f"{clock_txt}goal {state['goal']}")
        return []

    fig.tight_layout()
    if save_path is None:
        # Live view: run the closed loop FOREVER.  itertools.count() never wraps
        # the frame index (cycle mode keeps its phase continuous anyway), and
        # cache_frame_data=False stops matplotlib from accumulating every frame.
        # `anim_frames` only sets the length of a SAVED animation.
        anim = FuncAnimation(fig, update, frames=itertools.count(),
                             interval=1000 / fps, blit=False,
                             cache_frame_data=False)
        plt.show()
        return anim
    anim = FuncAnimation(fig, update, frames=anim_frames, interval=1000 / fps,
                         blit=False, repeat=False)
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {path}  ({anim_frames} frames)")
    return path


def concentration_slider(predict, objects, fields, params, schedule, curve,
                         *, metric_label="foods eaten per 1000 frames",
                         hidden_dim=64, save_path=None, fps=20, trail_speed=300,
                         stride=1, figsize=(11, 8.8)):
    """Watch one animal behave while the neuromodulator concentration is swept.

    `schedule` is the per-frame concentration; `curve` is (x, y, y_std) of the
    behavioural measure over concentration, precomputed from headless rollouts of
    the SAME dynamics.  The cursor on that curve is what makes the point: you see
    where the animal currently sits on the inverted U while watching what it can
    and cannot do there.

    Panels: world and trajectory, the noise field at the current concentration, the
    speed trace (silence at low concentration shows up here first), and the
    behavioural curve with a cursor.
    """
    use_headless(save_path)
    plt.rcParams.update(FONT)
    side = int(round(np.sqrt(hidden_dim)))

    state = world.initialize_demo_state(objects, "scripted")
    if params.threat_motion == "moving":
        state["threat_vels"] = world.make_threat_velocities(
            state["objects"], params.threat_speed, params.threat_seed)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_env, ax_curve = axes[0]
    ax_field, ax_speed = axes[1]

    x_curve, y_curve, y_std = curve
    ax_curve.plot(x_curve, y_curve, "-o", color="0.25", lw=1.8)
    if y_std is not None:
        ax_curve.fill_between(x_curve, y_curve - y_std, y_curve + y_std,
                              color="0.25", alpha=0.18)
    ax_curve.set_xscale("log")
    ax_curve.set_xlabel("neuromodulator concentration  (field scale)")
    ax_curve.set_ylabel(metric_label)
    ax_curve.set_title("acute dose response of this one animal")
    ax_curve.grid(alpha=0.3)
    cursor = ax_curve.axvline(schedule[0], color="#c23b3b", lw=2.0)
    marker, = ax_curve.plot([], [], "o", color="#c23b3b", ms=11, zorder=6)

    # Scale the field panel to a bit above the useful band rather than to the top of
    # the schedule: the extreme concentrations are the point of the sweep, but if the
    # colour scale is set by them the interesting middle renders almost black.
    peak = float(max(f.max() for f in fields.values()))
    vmax = peak * float(min(max(schedule), 2.5 * np.median(schedule)))
    speed_hist, eaten = [], [0]

    # `stride` advances the dynamics several times per drawn frame.  The episode is
    # unchanged; only the number of rendered frames drops, which is what the file
    # size is made of.
    def update(drawn):
        for k in range(stride):
            frame = min(drawn * stride + k, len(schedule) - 1)
            conc = float(schedule[frame])
            rec = world.advance_frame(state, predict, fields, params,
                                      concentration=conc, demo_mode="scripted",
                                      frame=frame, n_frames=len(schedule))
            eaten[0] += int(rec["ate"])
            speed_hist.append(rec["speed"])
        trail = np.array(state["trail"])

        ax_env.clear()
        draw_objects(ax_env, state["objects"], state["food_strengths"])
        if len(trail) > 1:
            ax_env.plot(trail[:, 0], trail[:, 1], "-", color="0.4", lw=1.5, alpha=0.85)
        ax_env.scatter(state["pos"][0], state["pos"][1], s=150, color="black", zorder=6)
        _square_axes(ax_env, f"{rec['label']}   eaten {eaten[0]}")

        ax_field.clear()
        ax_field.imshow(rec["field"].cpu().numpy().reshape(side, side), origin="lower",
                        extent=[0, 1, 0, 1], cmap="magma", vmin=0.0, vmax=vmax)
        ax_field.set_title(f"noise field at concentration {conc:.2f}")
        ax_field.set_xticks([])
        ax_field.set_yticks([])

        ax_speed.clear()
        ax_speed.plot(speed_hist[-trail_speed:], color="0.2", lw=1.4)
        ax_speed.set_xlabel("frame")
        ax_speed.set_ylabel("|v| (network output)")
        ax_speed.set_ylim(0, 1.15)
        ax_speed.set_yticks([0, 0.5, 1.0])
        ax_speed.grid(alpha=0.3)
        ax_speed.set_title(f"speed {rec['speed']:.2f}")

        cursor.set_xdata([conc, conc])
        marker.set_data([conc], [float(np.interp(conc, x_curve, y_curve))])
        return []

    fig.tight_layout()
    n_drawn = int(np.ceil(len(schedule) / stride))
    anim = FuncAnimation(fig, update, frames=n_drawn,
                         interval=1000 / fps, blit=False, repeat=True)
    if save_path is None:
        plt.show()
        return anim
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {path}  ({n_drawn} drawn frames, {len(schedule)} dynamics steps)")
    return path


def loss_figure(history, save_path=None):
    """Standalone training-loss figure."""
    use_headless(save_path)
    plt.rcParams.update(FONT)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    draw_loss_panel(ax, history)
    fig.tight_layout()
    return save_or_show(fig, save_path)
