"""
examples/neuromodulated_behavior_modes.py

Neuromodulator-like noise fields continuously re-weight always-present behavioral
drives -- in a MULTI-OBJECT environment with a distance-decayed salience input.
=====================================================================

Idea (paper demo)
-----------------
A single Noise-modulated Neural Network (NNN) with ONE fixed set of weights is
trained to produce a family of *mixed* behavioral vector fields.  What selects the
behavior at run time is not the weights but the *noise field* -- a per-hidden-unit
standard-deviation vector.  In the analytic NNN a unit that receives zero noise
emits exactly zero (it is effectively detached), while a unit with positive noise
is active.  A spatial noise field therefore *recruits* a particular functional
subnetwork out of the shared weights.

We interpret the three noise fields as neuromodulator-like concentration patterns
(think dopamine / serotonin / etc.).  Critically, the three behavioral drives --
approach food, avoid threat, approach shelter -- are NOT separate hard modes.
They are always present simultaneously; the noise field only changes their
relative weighting.  We therefore train three *biased* neuromodulatory states
rather than one-hot modes:

    food-biased state    : strongly approach food, still avoid/consider the rest
    threat-biased state  : strongly avoid threat, still weakly seek food/shelter
    shelter-biased state : strongly approach shelter, still avoid/consider the rest

Sensory encoding
----------------
The network does NOT receive the animal's absolute position.  It receives a
distance-decayed salience vector per object category (a 6D input): each category
contributes the sum over its objects of (unit vector to the object) weighted by
exp(-distance / decay_length).  Far objects are naturally ignored because their
influence decays exponentially, and nearby objects dominate -- a smooth,
biologically plausible alternative to picking only the single nearest object.
Because the input is object-relative, the learned policy depends on relative
sensory structure rather than memorized coordinates, so it still works when the
objects move.

Behavioral target
-----------------
The supervised target is a bounded mixed velocity built from the salience vectors:

    z = alpha_food * u_food - alpha_threat * u_threat + alpha_shelter * u_shelter
    v = tanh(gamma * ||z||) * z / (||z|| + eps)

The tanh speed rule means the animal moves slowly when all objects are far away
rather than always at unit speed.  This is *supervised* vector-field regression;
there is no reward, no policy gradient, no reinforcement learning.

At visualization time we smoothly interpolate the three trained noise fields AND
the corresponding alpha weights with cyclic softmax coefficients, modeling a
gradual transition of the internal neuromodulatory state; the behavior morphs
continuously between the biased states even though the weights never change.

Run
---
    python examples/neuromodulated_behavior_modes.py
    python examples/neuromodulated_behavior_modes.py --epochs 300 --anim-frames 200
    python examples/neuromodulated_behavior_modes.py --epochs 3000 --dynamic-objects

Requires only numpy, torch, matplotlib.  Uses plt.show() -- no files are written,
no ffmpeg is needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from nnn import model


CATEGORIES = ("food", "threat", "shelter")
STATES = ("food_biased", "threat_biased", "shelter_biased")

# Each biased neuromodulatory state weights the three drives [food, threat, shelter].
# The non-dominant weights stay non-zero (drives are never fully switched off, only
# re-weighted), but the dominant component is clearly stronger so the behavioral
# difference between states is easy to see.
ALPHA_STATES = {
    "food_biased":    np.array([1.60, 0.25, 0.25], dtype=np.float32),
    "threat_biased":  np.array([0.25, 2.00, 0.25], dtype=np.float32),
    "shelter_biased": np.array([0.25, 0.35, 1.60], dtype=np.float32),
}

# Which noise field drives which biased state.
STATE_TO_FIELD = {
    "food_biased": "food",
    "threat_biased": "threat",
    "shelter_biased": "shelter",
}


# ============================================================
# Environment: multiple objects of each category in [-1, 1]^2
# ============================================================

def make_objects(rng: np.random.Generator, n_food: int, n_threat: int,
                 n_shelter: int) -> dict[str, np.ndarray]:
    """Random object positions for each category, each of shape [n_k, 2]."""
    counts = {"food": n_food, "threat": n_threat, "shelter": n_shelter}
    return {
        key: rng.uniform(-1.0, 1.0, size=(counts[key], 2)).astype(np.float32)
        for key in CATEGORIES
    }


def make_object_velocities(rng: np.random.Generator, objects: dict[str, np.ndarray],
                           speed: float) -> dict[str, np.ndarray]:
    """A small constant velocity (random direction) per object, for the dynamic demo."""
    vels = {}
    for key in CATEGORIES:
        n = objects[key].shape[0]
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
        vels[key] = (speed * np.stack([np.cos(theta), np.sin(theta)], axis=1)
                     ).astype(np.float32)
    return vels


def update_dynamic_objects(objects: dict[str, np.ndarray],
                           velocities: dict[str, np.ndarray]) -> None:
    """Advance objects and bounce them off the walls of [-1, 1]^2 (in place).

    Dynamic objects demonstrate that the trained policy depends on the relative
    salience structure, not on memorized absolute coordinates: the animal keeps
    behaving sensibly as the objects drift.
    """
    for key in CATEGORIES:
        objects[key] += velocities[key]
        # Reflect the velocity component that left the box, then clip back in.
        out = (objects[key] > 1.0) | (objects[key] < -1.0)
        velocities[key][out] *= -1.0
        np.clip(objects[key], -1.0, 1.0, out=objects[key])


# ============================================================
# Gaussian relative-vector salience sensory encoding
# ============================================================
# The input is a Gaussian distance-weighted RELATIVE-VECTOR salience signal.
# Crucially it sums the raw relative vectors r_j (not unit vectors), so an
# object's contribution shrinks to zero as the animal reaches it.  This is what
# stops the animal oscillating around, or sticking to, a food/shelter object: at
# the object the pull vanishes.  The Gaussian weight makes far objects negligible.

def distance_decayed_category_vector(position: np.ndarray, object_positions: np.ndarray,
                                     decay_length: float, eps: float = 1e-8) -> np.ndarray:
    """Gaussian distance-decayed relative-vector salience signal for one category.

    This intentionally uses the raw relative vector r_j, not the unit vector, so
    the contribution of an object goes to zero when the animal reaches it.

    Args:
        position: np.ndarray, shape [2]
        object_positions: np.ndarray, shape [n_objects, 2]
        decay_length: float  -- interpreted as the Gaussian salience sigma.
        eps: float

    Returns:
        u: np.ndarray, shape [2]

    Definition:
        r_j = object_positions[j] - position
        d_j = ||r_j||
        weight_j = exp(-d_j^2 / (2 * sigma^2))
        u = sum_j weight_j * r_j            (raw r_j, NOT normalized)
    """
    rel = object_positions - position[None, :]                    # [n, 2]
    dist = np.linalg.norm(rel, axis=1, keepdims=True)             # [n, 1]
    weight = np.exp(-(dist ** 2) / (2.0 * decay_length ** 2))     # [n, 1]
    u = np.sum(weight * rel, axis=0)                              # [2]
    return u.astype(np.float32)


def nearest_relative_vector(position: np.ndarray, object_positions: np.ndarray,
                            eps: float = 1e-8):
    """Relative vector to the nearest object of one category.

    Kept only for the visualization highlight ("nearest example"); the network
    input is the salience-weighted sum over ALL objects, not this nearest one.
    """
    diffs = object_positions - position[None, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1) + eps)
    idx = int(np.argmin(dists))
    return diffs[idx].astype(np.float32), idx, float(dists[idx])


def encode_observation(position: np.ndarray, objects: dict[str, np.ndarray],
                       decay_lengths: dict[str, float]) -> np.ndarray:
    """6D salience-weighted sensory input for a single animal position.

    [u_food, u_threat, u_shelter] -- the distance-decayed salience vector of each
    category.  This is what the network sees -- never the absolute position.
    """
    parts = []
    for key in CATEGORIES:
        u = distance_decayed_category_vector(position, objects[key], decay_lengths[key])
        parts.append(u)
    return np.concatenate(parts).astype(np.float32)       # [6]


def encode_observations(positions: np.ndarray, objects: dict[str, np.ndarray],
                        decay_lengths: dict[str, float]) -> np.ndarray:
    """Batch encoder: [N, 2] positions -> [N, 6] observations.

    A simple loop is used deliberately to keep the example readable.
    """
    return np.stack([encode_observation(p, objects, decay_lengths) for p in positions],
                    axis=0)


# ============================================================
# Mixed behavior target (supervised)
# ============================================================
# The target is a single bounded velocity that mixes all three drives at once.
# Changing alpha re-weights the drives; it does not switch between hard modes.

def make_mixed_behavior_targets(observations: np.ndarray, alpha: np.ndarray,
                                gamma: float = 2.0, eps: float = 1e-8) -> np.ndarray:
    """Bounded mixed-velocity targets from salience observations and drive weights.

    Args:
        observations: np.ndarray, shape [N, 6]
            [u_food_x, u_food_y, u_threat_x, u_threat_y, u_shelter_x, u_shelter_y]
        alpha: np.ndarray, shape [3]  -- [alpha_food, alpha_threat, alpha_shelter]
        gamma: float                  -- speed saturation gain

    Returns:
        targets: np.ndarray, shape [N, 2]
    """
    u_food    = observations[:, 0:2]
    u_threat  = observations[:, 2:4]
    u_shelter = observations[:, 4:6]

    # Approach food, flee threat (note the minus), approach shelter -- all at once.
    z = alpha[0] * u_food - alpha[1] * u_threat + alpha[2] * u_shelter
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    speed = np.tanh(gamma * norm)                 # bounded: slow when objects far
    targets = speed * z / (norm + eps)
    return targets.astype(np.float32)


def compute_component_contributions(observation: np.ndarray, alpha: np.ndarray) -> dict:
    """Magnitude of each weighted behavioral component for a single observation.

    Diagnostic: reveals which drive currently dominates the animal's behavior, so
    we can tell whether behavior matches the intended neuromodulatory state.

    Args:
        observation: np.ndarray, shape [6]
            [u_food_x, u_food_y, u_threat_x, u_threat_y, u_shelter_x, u_shelter_y]
        alpha: np.ndarray, shape [3]  -- [alpha_food, alpha_threat, alpha_shelter]

    Returns:
        dict with keys "food", "threat", "shelter" -> scalar magnitudes
        ||alpha_food * u_food||, ||alpha_threat * u_threat||, ||alpha_shelter * u_shelter||
    """
    u_food    = observation[0:2]
    u_threat  = observation[2:4]
    u_shelter = observation[4:6]
    return {
        "food":    float(np.linalg.norm(alpha[0] * u_food)),
        "threat":  float(np.linalg.norm(alpha[1] * u_threat)),
        "shelter": float(np.linalg.norm(alpha[2] * u_shelter)),
    }


def make_training_grid(grid_side: int) -> np.ndarray:
    """A regular grid of animal positions over [-1, 1]^2, shape [grid_side**2, 2]."""
    axis = np.linspace(-1.0, 1.0, grid_side, dtype=np.float32)
    gx, gy = np.meshgrid(axis, axis)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


# ============================================================
# Neuromodulator-like noise fields
# ============================================================
# The hidden layer has 64 units, laid out as an 8 x 8 virtual sheet.  A state's
# noise field is a localized Gaussian bump on that sheet: units near the bump
# center get strong noise (are recruited), units far away get zero noise (are
# detached).  Different bump centers therefore recruit different, but partially
# overlapping, functional subnetworks.  The three fields represent distinct
# neuromodulator-like internal states that re-weight the behavioral drives.

GRID_SIZE = 8                       # 8 x 8 = 64 hidden units
FIELD_CENTERS = {                   # bump centers on the [0,1] x [0,1] unit sheet
    "food":    (0.25, 0.75),
    "threat":  (0.75, 0.25),
    "shelter": (0.25, 0.25),
}


def virtual_grid() -> np.ndarray:
    """Coordinates of the 8 x 8 hidden-unit sheet, flattened to [64, 2] in [0,1]."""
    axis = np.linspace(0.0, 1.0, GRID_SIZE, dtype=np.float32)
    gx, gy = np.meshgrid(axis, axis)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


def make_noise_field(center, base_std: float, sigma: float, theta: float) -> torch.Tensor:
    """Build one localized noise-std vector of length 64 for a neuromodulatory state.

    intensity[i] = exp(-||grid_i - center||^2 / (2 sigma^2)); small values below
    theta are truncated to zero, then stdvec = base_std * intensity.  A zero entry
    means the corresponding hidden unit is detached for this state.
    """
    grid = virtual_grid()
    center = np.asarray(center, dtype=np.float32)
    d2 = np.sum((grid - center) ** 2, axis=1)
    intensity = np.exp(-d2 / (2.0 * sigma ** 2))
    intensity[intensity < theta] = 0.0
    stdvec = base_std * intensity
    return torch.tensor(stdvec, dtype=torch.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def neuromodulatory_weights(phase: float, beta: float = 3.0) -> np.ndarray:
    """Smooth cyclic softmax weights over the three states as a function of phase.

    The three phase-shifted cosines make each state dominate in turn, and softmax
    guarantees a smooth, normalized transition -- a gradual change of the internal
    neuromodulatory state rather than a hard switch.  Order: [food, threat, shelter].
    """
    logits = beta * np.array([
        np.cos(phase),
        np.cos(phase - 2.0 * np.pi / 3.0),
        np.cos(phase - 4.0 * np.pi / 3.0),
    ])
    return softmax(logits)


def blend_fields(fields: dict[str, torch.Tensor], weights: np.ndarray) -> torch.Tensor:
    """Convex combination of the three noise fields (order: food, threat, shelter)."""
    return (weights[0] * fields["food"]
            + weights[1] * fields["threat"]
            + weights[2] * fields["shelter"])


def blend_alpha(weights: np.ndarray) -> np.ndarray:
    """Interpolated behavioral drive weights matching the blended noise field."""
    return (weights[0] * ALPHA_STATES["food_biased"]
            + weights[1] * ALPHA_STATES["threat_biased"]
            + weights[2] * ALPHA_STATES["shelter_biased"])


# ============================================================
# Model evaluation
# ============================================================

def evaluate_vector_field(net: nn.Module, obs: torch.Tensor,
                          field: torch.Tensor) -> torch.Tensor:
    """Predicted velocity vectors for a batch of observations under one noise field.

    The SAME field is used for both hidden layers, so a single neuromodulatory
    pattern gates the whole network.
    """
    return net(obs, stds=[field, field])


def diagnose_noise_field_separation(net: nn.Module, sample_observations: torch.Tensor,
                                    noise_fields: dict[str, torch.Tensor],
                                    device: torch.device) -> None:
    """Check that the SAME observations produce DIFFERENT outputs per pure field.

    If the trained network really uses the noise field to switch policy, the mean
    output distance between any two pure fields should be clearly above zero.
    Distances near zero would mean the network ignores the noise field.
    """
    obs = sample_observations.to(device)
    with torch.no_grad():
        y_food    = evaluate_vector_field(net, obs, noise_fields["food"])
        y_threat  = evaluate_vector_field(net, obs, noise_fields["threat"])
        y_shelter = evaluate_vector_field(net, obs, noise_fields["shelter"])

    def mean_dist(a, b):
        return float(torch.linalg.norm(a - b, dim=1).mean().item())

    print("\nNoise-field separation (mean ||y_i - y_j|| over "
          f"{obs.shape[0]} observations):")
    print(f"  food   vs threat : {mean_dist(y_food, y_threat):.4f}")
    print(f"  food   vs shelter: {mean_dist(y_food, y_shelter):.4f}")
    print(f"  threat vs shelter: {mean_dist(y_threat, y_shelter):.4f}")
    print("  (near zero => the network is NOT using the noise field)")


# ============================================================
# Training
# ============================================================
# Same observations, same network weights, different noise fields, different
# (mixed) target behaviors.  Every epoch we present all three biased states in a
# randomized order and take one Adam step per state.  The NNN thus learns to
# express different mixed behavior policies under different noise fields.

def train(net: nn.Module, obs: torch.Tensor, targets: dict[str, torch.Tensor],
          state_fields: dict[str, torch.Tensor], epochs: int, lr: float):
    """Supervised training of the shared weights across all biased states."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    history = {s: [] for s in STATES}

    log_step = max(1, epochs // 20)
    for epoch in range(epochs):
        for state in np.random.permutation(STATES):
            optimizer.zero_grad()
            pred = evaluate_vector_field(net, obs, state_fields[state])
            loss = criterion(pred, targets[state])
            loss.backward()
            optimizer.step()
            history[state].append(loss.item())

        if epoch % log_step == 0 or epoch == epochs - 1:
            msg = "  ".join(f"{s.split('_')[0]}={history[s][-1]:.4f}" for s in STATES)
            print(f"  epoch {epoch:5d}   {msg}")

    return history


def train_minibatch(net, obs, targets, state_fields, epochs, lr, chunk):
    """Same as train() but takes Adam steps over random chunks of the data."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    history = {s: [] for s in STATES}
    n = obs.shape[0]

    log_step = max(1, epochs // 20)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for state in np.random.permutation(STATES):
            last = None
            for start in range(0, n, chunk):
                idx = perm[start:start + chunk]
                optimizer.zero_grad()
                pred = evaluate_vector_field(net, obs[idx], state_fields[state])
                loss = criterion(pred, targets[state][idx])
                loss.backward()
                optimizer.step()
                last = loss.item()
            history[state].append(last)
        if epoch % log_step == 0 or epoch == epochs - 1:
            msg = "  ".join(f"{s.split('_')[0]}={history[s][-1]:.4f}" for s in STATES)
            print(f"  epoch {epoch:5d}   {msg}")
    return history


# ============================================================
# Visualization
# ============================================================

MARKERS = {  # how each object category is drawn
    "food":    dict(marker="o", color="tab:green", label="food"),
    "threat":  dict(marker="^", color="tab:red",   label="threat"),
    "shelter": dict(marker="s", color="tab:blue",  label="shelter"),
}


def animate(net: nn.Module, objects: dict[str, np.ndarray],
            velocities: dict[str, np.ndarray], fields: dict[str, torch.Tensor],
            history: dict[str, list], decay_lengths: dict[str, float],
            gamma: float, anim_frames: int, dynamic: bool,
            show_reference: bool, velocity_smoothing: float = 0.2,
            dt: float = 0.04):
    """Animate the trained system while the neuromodulatory state is interpolated.

    Panel 1  multi-object world + animal trajectory driven by the current field
    Panel 2  predicted velocity field (trained NNN), optional analytic reference
    Panel 3  the current 8 x 8 interpolated noise field as a heatmap
    Panel 4  training loss curves for the three biased states
    """
    # Coarse grid of candidate animal positions for the quiver panel.
    q_side = 13
    q_axis = np.linspace(-1.0, 1.0, q_side, dtype=np.float32)
    qx, qy = np.meshgrid(q_axis, q_axis)
    q_positions = np.stack([qx.ravel(), qy.ravel()], axis=1).astype(np.float32)

    # Mutable state captured by the update closure.
    state = {
        "pos": np.array([0.0, 0.0], dtype=np.float32),
        "vel": np.array([0.0, 0.0], dtype=np.float32),   # smoothed velocity
        "trail": [],
        "objects": {k: objects[k].copy() for k in CATEGORIES},
        "vels": {k: velocities[k].copy() for k in CATEGORIES},
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax_env, ax_quiver = axes[0]
    ax_field, ax_loss = axes[1]

    # Panel 4 is static -- draw the loss curves once.
    for s in STATES:
        color = MARKERS[STATE_TO_FIELD[s]]["color"]
        ax_loss.plot(history[s], color=color, lw=1.2, label=s.replace("_", "-"))
    ax_loss.set_yscale("log")
    ax_loss.set_title("Training loss (shared weights, per biased state)")
    ax_loss.set_xlabel("update step")
    ax_loss.set_ylabel("MSE")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(alpha=0.3)

    vmax = float(max(f.max() for f in fields.values()))
    suptitle = fig.suptitle("", fontsize=12)

    def draw_objects(ax, objs):
        for key in CATEGORIES:
            style = MARKERS[key]
            pts = objs[key]
            ax.scatter(pts[:, 0], pts[:, 1], marker=style["marker"], s=110,
                       color=style["color"], edgecolor="k", zorder=5,
                       label=style["label"])

    def update(frame):
        objs = state["objects"]
        if dynamic:
            update_dynamic_objects(objs, state["vels"])

        phase = 2.0 * np.pi * frame / anim_frames
        weights = neuromodulatory_weights(phase)      # [food, threat, shelter]
        field = blend_fields(fields, weights)
        current_alpha = blend_alpha(weights)          # reference drive weights

        # --- advance the animal one step using salience sensing + current field ---
        obs = encode_observation(state["pos"], objs, decay_lengths)      # [6]
        with torch.no_grad():
            v_pred = evaluate_vector_field(
                net, torch.tensor(obs[None, :], dtype=torch.float32), field
            ).numpy().ravel()
        # Velocity smoothing (animation-only): low-pass the predicted velocity to
        # damp visual jitter. s=1 => no smoothing, smaller s => more inertia.
        state["vel"] = ((1.0 - velocity_smoothing) * state["vel"]
                        + velocity_smoothing * v_pred).astype(np.float32)
        v = state["vel"]
        state["pos"] = np.clip(state["pos"] + dt * v, -1.0, 1.0).astype(np.float32)
        state["trail"].append(state["pos"].copy())
        if len(state["trail"]) > 120:
            state["trail"] = state["trail"][-120:]
        trail = np.array(state["trail"])

        # --- Panel 1: world + trajectory + nearest-example highlights ---
        ax_env.clear()
        draw_objects(ax_env, objs)
        # Dashed lines mark only the nearest example of each type; the actual
        # input is salience-weighted over ALL objects, not nearest-only.
        for key in CATEGORIES:
            _, idx, _ = nearest_relative_vector(state["pos"], objs[key])
            tgt = objs[key][idx]
            ax_env.plot([state["pos"][0], tgt[0]], [state["pos"][1], tgt[1]],
                        "--", color=MARKERS[key]["color"], lw=1.0, alpha=0.5)
        if len(trail) > 1:
            ax_env.plot(trail[:, 0], trail[:, 1], "-", color="0.4", lw=1.5, alpha=0.8)
        ax_env.scatter(state["pos"][0], state["pos"][1], s=140, color="black", zorder=6)
        ax_env.set_xlim(-1, 1)
        ax_env.set_ylim(-1, 1)
        ax_env.set_aspect("equal")
        ax_env.set_title("Multi-object world (dashed = nearest example only)")
        ax_env.legend(loc="upper left", fontsize=8)
        ax_env.grid(alpha=0.3)

        # --- Panel 2: predicted vector field over candidate positions ---
        # Each grid position is encoded through the CURRENT objects (salience sum).
        ax_quiver.clear()
        draw_objects(ax_quiver, objs)
        q_obs = encode_observations(q_positions, objs, decay_lengths)
        with torch.no_grad():
            vq = evaluate_vector_field(
                net, torch.tensor(q_obs, dtype=torch.float32), field
            ).numpy()
        ax_quiver.quiver(q_positions[:, 0], q_positions[:, 1], vq[:, 0], vq[:, 1],
                         color="0.25", angles="xy", scale=22, width=0.004,
                         label="trained NNN")
        if show_reference:
            # Optional diagnostic: analytic target field for the current alpha.
            vref = make_mixed_behavior_targets(q_obs, current_alpha, gamma=gamma)
            ax_quiver.quiver(q_positions[:, 0], q_positions[:, 1], vref[:, 0], vref[:, 1],
                             color="tab:orange", angles="xy", scale=22, width=0.003,
                             alpha=0.5, label="analytic reference")
            ax_quiver.legend(loc="upper left", fontsize=7)
        ax_quiver.set_xlim(-1, 1)
        ax_quiver.set_ylim(-1, 1)
        ax_quiver.set_aspect("equal")
        ax_quiver.set_title("Predicted mixed-behavior vector field")
        ax_quiver.grid(alpha=0.3)

        # --- Panel 3: current interpolated noise field ---
        ax_field.clear()
        heat = field.numpy().reshape(GRID_SIZE, GRID_SIZE)
        ax_field.imshow(heat, origin="lower", extent=[0, 1, 0, 1],
                        cmap="magma", vmin=0.0, vmax=vmax)
        ax_field.set_title(
            f"Noise field  F={weights[0]:.2f}, T={weights[1]:.2f}, S={weights[2]:.2f}"
        )
        ax_field.set_xticks([])
        ax_field.set_yticks([])

        # --- Dynamic title: noise weights, drive weights, and live contributions ---
        # Component contributions show which drive currently dominates the animal's
        # behavior at its present location -- useful to confirm behavior matches the
        # neuromodulatory state (and to spot the wrong component dominating).
        contrib = compute_component_contributions(obs, current_alpha)
        suptitle.set_text(
            "Neuromodulatory state continuously re-weights always-present drives\n"
            f"state weights: F={weights[0]:.2f}, T={weights[1]:.2f}, S={weights[2]:.2f}"
            f"    alpha: food={current_alpha[0]:.2f}, "
            f"threat={current_alpha[1]:.2f}, shelter={current_alpha[2]:.2f}\n"
            f"contribution: food={contrib['food']:.2f}, "
            f"threat={contrib['threat']:.2f}, shelter={contrib['shelter']:.2f}"
        )
        return []

    anim = FuncAnimation(fig, update, frames=anim_frames,
                         interval=60, blit=False, repeat=True)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    plt.show()
    return anim


# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Neuromodulator-like noise fields continuously re-weight "
                    "always-present behavioral drives in a shared-weight NNN."
    )
    p.add_argument("--epochs",      type=int,   default=3000,
                   help="Training epochs [3000]")
    p.add_argument("--lr",          type=float, default=3e-4,
                   help="Adam learning rate [3e-4]")
    p.add_argument("--grid-side",   type=int,   default=31,
                   help="Training grid resolution per axis [31]")
    p.add_argument("--hidden-dim",  type=int,   default=64,
                   help="Hidden units per layer; must be 64 (8x8 sheet) [64]")
    p.add_argument("--anim-frames", type=int,   default=360,
                   help="Number of animation frames per cycle [360]")
    p.add_argument("--train-chunk", type=int,   default=0,
                   help="If > 0, minibatch the training data into chunks of this "
                        "many points per step [0 = full-batch]")
    p.add_argument("--seed",        type=int,   default=0,
                   help="Random seed [0]")
    # Environment.
    p.add_argument("--n-food",      type=int,   default=5,
                   help="Number of food objects [5]")
    p.add_argument("--n-threat",    type=int,   default=3,
                   help="Number of threat objects [3]")
    p.add_argument("--n-shelter",   type=int,   default=2,
                   help="Number of shelter objects [2]")
    p.add_argument("--dynamic-objects", action="store_true",
                   help="Slowly move objects during the animation (demo only)")
    p.add_argument("--object-speed", type=float, default=0.002,
                   help="Per-frame speed of dynamic objects [0.002]")
    # Gaussian relative-vector salience encoding (lambda = Gaussian salience sigma).
    p.add_argument("--lambda-food",    type=float, default=0.65,
                   help="Gaussian salience sigma for food [0.65]")
    p.add_argument("--lambda-threat",  type=float, default=0.35,
                   help="Gaussian salience sigma for threat (short = local) [0.35]")
    p.add_argument("--lambda-shelter", type=float, default=0.75,
                   help="Gaussian salience sigma for shelter [0.75]")
    p.add_argument("--target-gamma",   type=float, default=2.0,
                   help="Speed saturation gain in the tanh target rule [2.0]")
    # Noise-field shape (slightly sharper => more distinct subnetworks).
    p.add_argument("--base-std",    type=float, default=0.8,
                   help="Peak noise std of a recruited unit [0.8]")
    p.add_argument("--sigma",       type=float, default=0.22,
                   help="Gaussian width of a noise bump on the 8x8 sheet [0.22]")
    p.add_argument("--theta",       type=float, default=0.15,
                   help="Truncation threshold for the intensity field [0.15]")
    # Visualization.
    p.add_argument("--show-reference", action="store_true",
                   help="Overlay the analytic target vector field as a diagnostic")
    p.add_argument("--velocity-smoothing", type=float, default=0.2,
                   help="Animation-only velocity smoothing in [0, 1]: "
                        "v = (1-s)*v_prev + s*v_pred. 1.0 = no smoothing, "
                        "smaller = more inertia (less jitter) [0.2]")
    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)   # seeded, reproducible objects

    if args.hidden_dim != GRID_SIZE * GRID_SIZE:
        raise ValueError(
            f"--hidden-dim must be {GRID_SIZE * GRID_SIZE} to match the "
            f"{GRID_SIZE}x{GRID_SIZE} noise sheet; got {args.hidden_dim}."
        )

    # CPU by default: avoids Matplotlib/CUDA backend friction for the animation.
    device = torch.device("cpu")

    decay_lengths = {
        "food": args.lambda_food,
        "threat": args.lambda_threat,
        "shelter": args.lambda_shelter,
    }

    # --- Multi-object environment (static during training) ---
    objects = make_objects(rng, args.n_food, args.n_threat, args.n_shelter)
    velocities = make_object_velocities(rng, objects, args.object_speed)
    print("Objects: " + ", ".join(f"{k}={objects[k].shape[0]}" for k in CATEGORIES))

    # --- Data: positions -> salience observations -> mixed behavior targets ---
    positions_np = make_training_grid(args.grid_side)
    obs_np = encode_observations(positions_np, objects, decay_lengths)    # [N, 6]
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
    # One mixed-velocity target set per biased state (same obs, different alpha).
    targets = {
        s: torch.tensor(
            make_mixed_behavior_targets(obs_np, ALPHA_STATES[s], gamma=args.target_gamma),
            dtype=torch.float32, device=device)
        for s in STATES
    }

    # --- Neuromodulator-like noise fields (one per state) ---
    food_noise_field    = make_noise_field(FIELD_CENTERS["food"],    args.base_std, args.sigma, args.theta)
    threat_noise_field  = make_noise_field(FIELD_CENTERS["threat"],  args.base_std, args.sigma, args.theta)
    shelter_noise_field = make_noise_field(FIELD_CENTERS["shelter"], args.base_std, args.sigma, args.theta)
    fields = {"food": food_noise_field.to(device),
              "threat": threat_noise_field.to(device),
              "shelter": shelter_noise_field.to(device)}
    state_fields = {s: fields[STATE_TO_FIELD[s]] for s in STATES}
    for cat in CATEGORIES:
        active = int((fields[cat] > 0).sum().item())
        print(f"noise field '{cat:7s}': {active:2d}/64 units recruited, "
              f"peak std={fields[cat].max().item():.3f}")

    # --- Model: analytic NNN, 6D salience input, weights shared ---
    H = args.hidden_dim
    net = model.SimpleNNNAnalytic(structure=[6, H, H, 2], std=args.base_std).to(device)

    print(f"\nModel : SimpleNNNAnalytic  structure=[6, {H}, {H}, 2]")
    print(f"Train : epochs={args.epochs}  lr={args.lr}  grid={args.grid_side}x{args.grid_side}")
    print("        same obs, same weights, different noise fields -> re-weighted drives.\n")

    # --- Train ---
    if args.train_chunk > 0:
        history = train_minibatch(net, obs, targets, state_fields,
                                  args.epochs, args.lr, args.train_chunk)
    else:
        history = train(net, obs, targets, state_fields, args.epochs, args.lr)

    # --- Final per-state MSE ---
    criterion = nn.MSELoss()
    print("\nFinal per-state MSE:")
    with torch.no_grad():
        for s in STATES:
            pred = evaluate_vector_field(net, obs, state_fields[s])
            print(f"  {s:15s}: {criterion(pred, targets[s]).item():.5f}")

    # --- Diagnostic: does the network actually use the noise field? ---
    n_sample = min(256, obs.shape[0])
    sample_idx = torch.randperm(obs.shape[0])[:n_sample]
    diagnose_noise_field_separation(net, obs[sample_idx], fields, device)

    # --- Animate the trained system ---
    print("\nOpening animation window (close it to exit)...")
    if args.dynamic_objects:
        print("  dynamic objects ON: behavior tracks the moving salience structure.")
    animate(net, objects, velocities, fields, history, decay_lengths,
            args.target_gamma, args.anim_frames, args.dynamic_objects,
            args.show_reference, args.velocity_smoothing)


if __name__ == "__main__":
    main()
