"""
examples/neuromodulated_behavior_modes.py

Neuromodulator-like noise fields act as RECRUITMENT VARIABLES: each field recruits
a different functional subnetwork out of ONE shared set of weights, producing a
clearly recognizable behavioral tendency -- Foraging, Avoidance, or Sheltering.
=====================================================================

Idea (paper demo)
-----------------
A single Noise-modulated Neural Network (NNN) with ONE fixed set of weights is
trained to produce three behavioral vector fields.  What selects the behavior at
run time is not the weights but the *noise field* -- a per-hidden-unit standard-
deviation vector.  In the analytic NNN a unit that receives zero noise emits
exactly zero (it is effectively detached), while a unit with positive noise is
active.  A spatial noise field therefore *recruits* a particular functional
subnetwork out of the shared weights.

We interpret the three noise fields as neuromodulator-like concentration patterns
(think dopamine / serotonin / etc.).  Each recruits a distinct, visually
classifiable behavior:

    food-biased field    -> Foraging   : approach food
    threat-biased field  -> Avoidance  : move away from threat
    shelter-biased field -> Sheltering : approach shelter

The three behavioral weights are near-one-hot (the off-drives are kept slightly
non-zero only so the model is not literally a one-hot switch), so under each field
the recruited behavior is unambiguous rather than a compromise between drives.

Sensory encoding
----------------
The network does NOT receive the animal's absolute position.  It receives a
Gaussian distance-decayed RELATIVE-VECTOR salience per object category (a 6D
input): each category contributes the sum over its objects of the raw relative
vector r_j weighted by exp(-d_j^2 / (2 sigma^2)).  Using the raw r_j (not the
unit vector) makes an object's contribution vanish as the animal reaches it, so
the animal does not oscillate around or stick to a reached object.  The Gaussian
weight makes far objects negligible.  Because the input is object-relative, the
learned policy depends on relative sensory structure, not memorized coordinates.

Behavioral target
-----------------
The supervised target is a bounded velocity built from the salience vectors:

    z = alpha_food * u_food - alpha_threat * u_threat + alpha_shelter * u_shelter
    v = tanh(gamma * ||z||) * z / (||z|| + eps)

With near-one-hot alpha this is a near-pure approach/avoid field.  The tanh speed
rule means the animal moves slowly when all objects are far away.  This is
*supervised* vector-field regression; there is no reward, no policy gradient, no
reinforcement learning.

Perception
----------
The 6D input is the RAW relative vector to the NEAREST food / threat / shelter
(never absolute position).  Nearest-object sensing scales to arbitrarily many
objects and avoids the centroid attractor a multi-object sum would create.

Closed-loop demo dynamics (animation only, never in training)
-------------------------------------------------------------
The default 'scripted' demo is a REACTIVE closed loop: perception -> a simple
context/need rule picks which noise field to recruit -> the shared-weight NNN
produces the movement.  A nearby threat recruits Avoidance; otherwise the more
urgent internal drive (hunger vs shelter need, both rising over time and reset by
eating / sheltering) recruits Foraging or Sheltering.  The controller only SELECTS
the field -- all approach/avoid geometry is computed by the NNN, mirroring how real
neuromodulator release is gated by sensory and internal state.  Threats wander by
default (--threat-motion static to freeze them); food respawns so the loop persists.

Demo modes
----------
  scripted    : the reactive closed-loop behavior demo (default);
  cycle       : smooth cyclic interpolation between the three fields/alphas;
  pure-panels : a static side-by-side of the three recruited vector fields.

Run
---
    python examples/neuromodulated_behavior_modes.py
    python examples/neuromodulated_behavior_modes.py --epochs 300 --anim-frames 240
    python examples/neuromodulated_behavior_modes.py --demo-mode cycle --epochs 1000
    python examples/neuromodulated_behavior_modes.py --layout random --epochs 1000

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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from nnn import model


CATEGORIES = ("food", "threat", "shelter")
STATES = ("food_biased", "threat_biased", "shelter_biased")

# Each neuromodulatory state weights the three drives [food, threat, shelter].
# These are NEAR-ONE-HOT: the dominant drive is strong and the off-drives are kept
# only slightly non-zero (so the model is not literally a one-hot switch), which
# makes the recruited behavior under each field cleanly classifiable as Foraging,
# Avoidance, or Sheltering rather than a compromise between drives.
ALPHA_STATES = {
    "food_biased":    np.array([1.80, 0.03, 0.03], dtype=np.float32),
    "threat_biased":  np.array([0.03, 2.20, 0.03], dtype=np.float32),
    "shelter_biased": np.array([0.03, 0.03, 1.80], dtype=np.float32),
}

# Human-readable behavior label recruited by each state / noise field.
EPISODE_LABEL = {
    "food_biased": "Foraging",
    "threat_biased": "Avoidance",
    "shelter_biased": "Sheltering",
}
FIELD_EPISODE = {"food": "Foraging", "threat": "Avoidance", "shelter": "Sheltering"}

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


def make_threat_velocities(objects: dict[str, np.ndarray], speed: float,
                           seed: int = 0) -> np.ndarray:
    """A constant random-direction velocity per threat, [n_threat, 2]."""
    rng = np.random.default_rng(seed)
    n = objects["threat"].shape[0]
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return (speed * np.stack([np.cos(theta), np.sin(theta)], axis=1)).astype(np.float32)


def step_threats(objects: dict[str, np.ndarray], threat_vels: np.ndarray,
                 shelter_keepout: float = 0.0, bounds: float = 1.0) -> None:
    """Advance ONLY the threats and bounce them inside [-bounds, bounds]^2 (in place).

    Moving threats make the reactive demo livelier and dodge the deadlock a threat
    parked on the sole food<->shelter corridor could cause: a wandering threat
    leaves the path on its own, so the agent's flee<->approach loop resolves.

    `bounds` < 1 keeps threats away from the map edges so they do not push the agent
    into a wall.  If shelter_keepout > 0, threats are also kept OUT of the shelter
    regions (pushed to the boundary and reflected), so a resting agent is undisturbed.
    """
    t = objects["threat"]
    t += threat_vels
    out = (t > bounds) | (t < -bounds)
    threat_vels[out] *= -1.0
    np.clip(t, -bounds, bounds, out=t)

    if shelter_keepout > 0.0:
        for i in range(t.shape[0]):
            for s in objects["shelter"]:
                d = t[i] - s
                dist = float(np.linalg.norm(d))
                if dist < shelter_keepout:
                    n = d / dist if dist > 1e-6 else np.array([1.0, 0.0], np.float32)
                    t[i] = (s + n * shelter_keepout).astype(np.float32)   # to boundary
                    vn = float(np.dot(threat_vels[i], n))
                    if vn < 0.0:                                          # reflect inward vel
                        threat_vels[i] = (threat_vels[i] - 2.0 * vn * n).astype(np.float32)
        np.clip(t, -1.0, 1.0, out=t)


# ============================================================
# Scripted multi-object scene + reactive neuromodulatory control
# ============================================================
# The closed-loop demo runs a full loop: perception -> neuromodulatory state ->
# recruited subnetwork -> behavior.  The neuromodulatory state is a CONTINUOUS
# 3-vector of weights (food/threat/shelter) produced by a simple graded rule
# (neuromod_weights) and used to BLEND the three noise fields -- like an overlapping
# mixture of neuromodulator concentrations rather than a hard switch.  It does NOT
# compute any movement: all approach/avoid geometry is produced by the shared-weight
# NNN under the blended noise field.  All dynamics below are animation-only and
# never touch training.

# Agent start position for the scripted demo (near the shelter side, on the left).
SCRIPTED_START = np.array([-0.55, 0.05], dtype=np.float32)

# Radius of the drawn circular shelter region (diameter ~0.25).  The arrival-
# detection radius (--shelter-radius) is kept clearly SMALLER than this, and threats
# are reflected at 1.5x this radius so they never touch the region.
SHELTER_REGION_RADIUS = 0.125
THREAT_KEEPOUT_RADIUS = 1.5 * SHELTER_REGION_RADIUS

# Wandering threats bounce inside this inner box (< 1.0) so they stay central and do
# not lure/push the agent onto the map edges (reduces wall-hugging behaviour).
THREAT_BOUNDS = 0.72

# Food/shelter goal commitment thresholds: the agent forages until hunger falls to
# HUNGER_LO (having eaten enough -- possibly several foods), then shelters until
# rested to SHELTER_LO.  This 1-bit commitment prevents food<->shelter dithering and
# yields "food after food" and "stay at shelter"; threat blends over it continuously.
HUNGER_LO = 0.35
SHELTER_LO = 0.15


def make_scripted_objects() -> dict[str, np.ndarray]:
    """Fixed multi-object scene: food on the right, shelter on the left, threats in
    the middle band the agent must cross.  With nearest-object perception this needs
    no careful spacing, and the central threats make the agent detour naturally.
    """
    return {
        # Two foods on the right, held away from the edges (|coord| <= ~0.62) so the
        # agent does not hug a wall to reach them; spaced so each is individually
        # reachable yet close enough for "food after food".
        "food":    np.array([[0.42, 0.45], [0.62, 0.10]], dtype=np.float32),
        # Threats guard the middle band but sit OFF the direct food<->shelter route,
        # so the agent detours past them without being trapped in a flee<->approach
        # oscillation (which happens if a threat blocks the only corridor).
        "threat":  np.array([[0.05, 0.20], [0.15, -0.30], [-0.10, -0.05]],
                            dtype=np.float32),
        # Two shelters far apart on the left (>3x their former separation) but kept
        # off the edges (|coord| <= ~0.64), at roughly equal distance from the food.
        "shelter": np.array([[-0.64, 0.52], [-0.28, -0.58]], dtype=np.float32),
    }


def initialize_demo_state(objects: dict[str, np.ndarray], layout: str) -> dict:
    """Initialize agent state and the internal drives that gate the noise field."""
    start = SCRIPTED_START.copy() if layout == "scripted" \
        else np.zeros(2, dtype=np.float32)
    return {
        "pos": start.copy(),
        "start": start.copy(),
        "vel": np.zeros(2, dtype=np.float32),               # smoothed heading
        "trail": [start.copy()],
        "objects": {k: objects[k].copy() for k in CATEGORIES},
        "food_strengths": np.ones(objects["food"].shape[0], dtype=np.float32),
        "hunger": 1.0,          # rises with time, reduced by eating
        "shelter_need": 0.0,    # rises with time, cleared after a rest at shelter
        "goal": "food",         # committed food/shelter goal (1 bit, hysteretic)
        "rest": 0,              # rest-at-shelter countdown (frames remaining)
        # Continuous neuromodulatory weights [food, threat, shelter] (start: forage).
        "w": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }


def apply_food_depletion(position: np.ndarray, objects: dict[str, np.ndarray],
                         food_strengths: np.ndarray, eat_radius: float,
                         respawn: bool = False) -> bool:
    """Deplete any food the agent reaches (in place); return True if one was eaten.

    With respawn enabled, a depleted food recovers once the agent has moved well
    clear of it, so the world stays populated for a continuous reactive demo.
    """
    ate = False
    food = objects["food"]
    for j in range(food.shape[0]):
        d = float(np.linalg.norm(food[j] - position))
        if d < eat_radius and food_strengths[j] > 0.0:
            food_strengths[j] = 0.0
            ate = True
        elif respawn and food_strengths[j] <= 0.0 and d > 2.5 * eat_radius:
            food_strengths[j] = 1.0
    return ate


def apply_shelter_satisfaction(position: np.ndarray, objects: dict[str, np.ndarray],
                               shelter_radius: float) -> bool:
    """Return True if the agent is currently inside any shelter's radius."""
    shelter = objects["shelter"]
    if shelter.shape[0] == 0:
        return False
    nearest = float(np.linalg.norm(shelter - position[None, :], axis=1).min())
    return nearest < shelter_radius


def neuromod_weights(position: np.ndarray, objects: dict[str, np.ndarray],
                     hunger: float, shelter_need: float, goal: str,
                     prev_w: np.ndarray, threat_gain: float, threat_range: float,
                     smoothing: float) -> tuple:
    """Continuous neuromodulatory weights [food, threat, shelter] and updated goal.

    Two levels, kept deliberately simple:
      * a committed food/shelter GOAL (1 bit) switches only at the HUNGER_LO /
        SHELTER_LO thresholds -- decisive, so the agent does not dither between food
        and shelter, and it forages through several foods / rests at a shelter;
      * a distance-graded threat urgency continuously CROSS-FADES the threat field
        over that goal, and the whole weight vector is low-pass filtered over time.
    The graded + smoothed threat blend prevents the threat<->food / threat<->shelter
    chattering a hard rule makes, and blends the recruited subnetworks like an
    overlapping mixture of neuromodulator concentrations.  Motion is still all NNN.
    """
    # 1-bit goal commitment (hysteresis by satiation, not by distance).
    if goal == "food" and hunger <= HUNGER_LO:
        goal = "shelter"
    elif goal == "shelter" and shelter_need <= SHELTER_LO:
        goal = "food"
    base = np.array([1.0, 0.0, 0.0] if goal == "food" else [0.0, 0.0, 1.0],
                    dtype=np.float32)

    # Graded threat urgency -> threat weight a in [0, 1].
    threat = objects["threat"]
    if threat.shape[0] > 0:
        d_threat = float(np.linalg.norm(threat - position[None, :], axis=1).min())
        g_threat = float(np.exp(-(d_threat / threat_range) ** 2))   # 1 near, 0 far
    else:
        g_threat = 0.0
    a = float(np.clip(threat_gain * g_threat, 0.0, 1.0))

    w_target = (1.0 - a) * base + a * np.array([0.0, 1.0, 0.0], dtype=np.float32)
    w = ((1.0 - smoothing) * prev_w + smoothing * w_target).astype(np.float32)
    return w, goal, g_threat


# ============================================================
# Nearest-object relative-vector sensory encoding
# ============================================================
# The 6D input is the RAW relative vector to the NEAREST object of each category:
# [r_food, r_threat, r_shelter].  Using the nearest object (not a sum over all
# objects) keeps the input well-defined and free of the centroid attractor that a
# multi-object sum creates between objects; it also scales naturally to arbitrarily
# many objects.  Using the raw vector r (not the unit vector) means the signal ->0
# exactly at the object, so the agent decelerates onto its target instead of
# oscillating around it.  The network never sees absolute position.

def nearest_relative_vector(position: np.ndarray, object_positions: np.ndarray,
                            available: np.ndarray = None) -> np.ndarray:
    """Raw relative vector to the nearest (available) object of one category.

    Args:
        position: np.ndarray, shape [2]
        object_positions: np.ndarray, shape [n_objects, 2]
        available: optional bool/0-1 mask, shape [n_objects] -- objects that are
            NOT available (e.g. eaten food) are ignored.  If every object is
            unavailable (or there are none), a zero vector is returned.

    Returns:
        r: np.ndarray, shape [2]  -- object_nearest - position (zeros if none).
    """
    if object_positions.shape[0] == 0:
        return np.zeros(2, dtype=np.float32)
    rel = object_positions - position[None, :]                    # [n, 2]
    dist = np.linalg.norm(rel, axis=1)                            # [n]
    if available is not None:
        dist = np.where(np.asarray(available) > 0.0, dist, np.inf)
    j = int(np.argmin(dist))
    if not np.isfinite(dist[j]):
        return np.zeros(2, dtype=np.float32)
    return rel[j].astype(np.float32)


def encode_observation(position: np.ndarray, objects: dict[str, np.ndarray],
                       food_strengths: np.ndarray = None) -> np.ndarray:
    """6D nearest-object sensory input for a single animal position.

    [r_food, r_threat, r_shelter] -- the raw relative vector to the nearest object
    of each category.  Perception is pure geometry: internal drives (hunger /
    shelter need) are NOT mixed in here -- they only gate WHICH noise fields are
    recruited (see neuromod_weights).  Depleted food is excluded via food_strengths
    so the agent perceives the nearest *available* food.
    """
    r_food = nearest_relative_vector(position, objects["food"], available=food_strengths)
    r_threat = nearest_relative_vector(position, objects["threat"])
    r_shelter = nearest_relative_vector(position, objects["shelter"])
    return np.concatenate([r_food, r_threat, r_shelter]).astype(np.float32)   # [6]


def encode_observations(positions: np.ndarray, objects: dict[str, np.ndarray],
                        food_strengths: np.ndarray = None) -> np.ndarray:
    """Batch encoder: [N, 2] positions -> [N, 6] nearest-object observations.

    A simple loop is used deliberately to keep the example readable.  Training
    calls this with all food available; the animation passes the live food_strengths.
    """
    return np.stack([encode_observation(p, objects, food_strengths)
                     for p in positions], axis=0)


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

# Proxy legend handles (built once) so per-object fading does not spawn duplicate
# or missing legend entries.
LEGEND_HANDLES = [
    Line2D([0], [0], marker=MARKERS[k]["marker"], color="w",
           markerfacecolor=MARKERS[k]["color"], markeredgecolor="k",
           markersize=10, label=k)
    for k in CATEGORIES
]


def draw_objects(ax, objs: dict[str, np.ndarray],
                 food_strengths: np.ndarray = None,
                 shelter_region_radius: float = SHELTER_REGION_RADIUS):
    """Draw the objects; depleted food fades out, shelters are circular regions."""
    # Food: bright while available, small and pale once eaten (shows the change).
    food = objs["food"]
    fs = food_strengths if food_strengths is not None else np.ones(food.shape[0])
    for j in range(food.shape[0]):
        if fs[j] > 0.0:
            ax.scatter(food[j, 0], food[j, 1], marker="o", s=110,
                       color="tab:green", edgecolor="k", zorder=5)
        else:
            ax.scatter(food[j, 0], food[j, 1], marker="o", s=45,
                       color="tab:green", edgecolor="none", alpha=0.22, zorder=4)
    # Threat: always fully drawn (avoidance is always active).
    thr = objs["threat"]
    ax.scatter(thr[:, 0], thr[:, 1], marker="^", s=110,
               color="tab:red", edgecolor="k", zorder=5)
    # Shelter: a filled circular REGION (the agent rests anywhere inside it).
    for c in objs["shelter"]:
        ax.add_patch(Circle((c[0], c[1]), shelter_region_radius, facecolor="tab:blue",
                            alpha=0.18, edgecolor="tab:blue", lw=1.2, zorder=2))
    ax.scatter(objs["shelter"][:, 0], objs["shelter"][:, 1], marker="s", s=45,
               color="tab:blue", edgecolor="k", zorder=5)
    ax.legend(handles=LEGEND_HANDLES, loc="upper left", fontsize=8)


def _draw_loss_panel(ax_loss, history):
    """Static training-loss curves (shared weights, one curve per state)."""
    for s in STATES:
        color = MARKERS[STATE_TO_FIELD[s]]["color"]
        ax_loss.plot(history[s], color=color, lw=1.2, label=EPISODE_LABEL[s])
    ax_loss.set_yscale("log")
    ax_loss.set_title("Training loss (shared weights, per recruited behavior)")
    ax_loss.set_xlabel("update step")
    ax_loss.set_ylabel("MSE")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(alpha=0.3)


def show_pure_panels(net: nn.Module, objects: dict[str, np.ndarray],
                     fields: dict[str, torch.Tensor], gamma: float) -> None:
    """Static diagnostic: the three recruited vector fields side by side.

    Same weights, same objects, same observation grid -- only the noise field
    differs.  The three panels should look qualitatively different (approach food /
    flee threat / approach shelter), making the recruitment effect obvious.
    """
    q_side = 15
    q_axis = np.linspace(-1.0, 1.0, q_side, dtype=np.float32)
    qx, qy = np.meshgrid(q_axis, q_axis)
    q_positions = np.stack([qx.ravel(), qy.ravel()], axis=1).astype(np.float32)
    q_obs = encode_observations(q_positions, objects)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, cat in zip(axes, CATEGORIES):
        draw_objects(ax, objects)
        with torch.no_grad():
            vq = evaluate_vector_field(
                net, torch.tensor(q_obs, dtype=torch.float32), fields[cat]
            ).numpy()
        ax.quiver(q_positions[:, 0], q_positions[:, 1], vq[:, 0], vq[:, 1],
                  color="0.25", angles="xy", scale=22, width=0.004)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_title(f"'{cat}' noise field  ->  {FIELD_EPISODE[cat]}")
        ax.grid(alpha=0.3)
    fig.suptitle("Pure noise-field recruitment: one shared weight set, three "
                 "fields, three behaviors", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()


def animate(net: nn.Module, objects: dict[str, np.ndarray],
            fields: dict[str, torch.Tensor], history: dict[str, list],
            gamma: float, anim_frames: int,
            demo_mode: str, layout: str, eat_radius: float, shelter_radius: float,
            food_respawn: bool, cruise_speed: float = 0.9,
            hunger_rate: float = 0.006, need_rate: float = 0.006,
            eat_amount: float = 0.6, rest_frames: int = 50,
            threat_gain: float = 1.7, threat_range: float = 0.40,
            neuromod_smoothing: float = 0.12,
            threat_motion: str = "moving", threat_speed: float = 0.01, seed: int = 0,
            dynamic: bool = False,
            velocities: dict[str, np.ndarray] = None, show_reference: bool = False,
            velocity_smoothing: float = 0.2, dt: float = 0.04):
    """Closed-loop animation of the trained system.

    demo_mode == "scripted": a REACTIVE closed loop -- each frame neuromod_weights
        turns the graded drives (hunger, distance-graded threat urgency, shelter
        need) into continuous, temporally smoothed weights that BLEND the three
        noise fields.  The recruited shared-weight NNN then produces the movement;
        the agent cruises at constant speed along the predicted DIRECTION (its
        magnitude vanishes at objects and would otherwise stall it).  Partial
        satiation per food and gradual satisfaction inside a shelter yield richer
        behavior (food-after-food, staying at shelter).
    demo_mode == "cycle": smoothly interpolate the fields/alphas around the cycle.

    Panel 1  world + agent trajectory (eaten food fades)
    Panel 2  predicted velocity field (trained NNN), optional analytic reference
    Panel 3  the current 8 x 8 noise field as a heatmap
    Panel 4  training loss curves (static)
    """
    # Coarse grid of candidate animal positions for the quiver panel.
    q_side = 13
    q_axis = np.linspace(-1.0, 1.0, q_side, dtype=np.float32)
    qx, qy = np.meshgrid(q_axis, q_axis)
    q_positions = np.stack([qx.ravel(), qy.ravel()], axis=1).astype(np.float32)

    # Mutable closed-loop state (agent + live environment/drive dynamics).
    state = initialize_demo_state(objects, layout)
    if dynamic and velocities is not None:
        state["vels"] = {k: velocities[k].copy() for k in CATEGORIES}
    threats_move = (demo_mode == "scripted" and threat_motion == "moving")
    if threats_move:
        state["threat_vels"] = make_threat_velocities(state["objects"], threat_speed, seed)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax_env, ax_quiver = axes[0]
    ax_field, ax_loss = axes[1]

    _draw_loss_panel(ax_loss, history)

    vmax = float(max(f.max() for f in fields.values()))
    suptitle = fig.suptitle("", fontsize=12)

    def update(frame):
        objs = state["objects"]
        if demo_mode == "cycle" and dynamic and "vels" in state:
            update_dynamic_objects(objs, state["vels"])

        # --- update drives and choose the (blended) noise field for this frame ---
        if demo_mode == "scripted":
            if threats_move:
                # Wander the threats within a central box (away from the edges) and
                # reflect them at 1.5x the shelter region radius so they never touch a
                # shelter.
                step_threats(objs, state["threat_vels"],
                             shelter_keepout=THREAT_KEEPOUT_RADIUS,
                             bounds=THREAT_BOUNDS)
            # Hunger always rises.  Sheltering uses a time-based REST: on first
            # arrival the agent starts a rest countdown; when it elapses the shelter
            # need is cleared (goal flips to food).  A time-based rest is robust to a
            # wandering threat briefly nudging the agent off the shelter -- unlike a
            # "decay only while inside" rule, which such a threat can stall forever.
            inside_shelter = apply_shelter_satisfaction(state["pos"], objs, shelter_radius)
            state["hunger"] = min(1.0, state["hunger"] + hunger_rate)
            if state["goal"] == "shelter":
                if inside_shelter and state["rest"] == 0:
                    state["rest"] = rest_frames          # arrived -> begin resting
                if state["rest"] > 0:
                    state["rest"] -= 1
                    state["shelter_need"] = 1.0          # hold at shelter while resting
                    if state["rest"] == 0:
                        state["shelter_need"] = 0.0      # rested -> will head out to forage
                else:
                    state["shelter_need"] = min(1.0, state["shelter_need"] + need_rate)
            else:  # foraging: shelter need accrues, no pending rest
                state["shelter_need"] = min(1.0, state["shelter_need"] + need_rate)
                state["rest"] = 0
            # Continuous, temporally smoothed neuromodulatory weights -> blended field.
            state["w"], state["goal"], _ = neuromod_weights(
                state["pos"], objs, state["hunger"], state["shelter_need"],
                state["goal"], state["w"], threat_gain, threat_range,
                neuromod_smoothing)
            weights = state["w"]
            field = blend_fields(fields, weights)
            current_alpha = blend_alpha(weights)
            dom = int(np.argmax(weights))
            state_name = STATES[dom]
            field_name = STATE_TO_FIELD[state_name]
            episode_label = EPISODE_LABEL[state_name]
        else:  # cycle
            phase = 2.0 * np.pi * frame / anim_frames
            weights = neuromodulatory_weights(phase)       # [food, threat, shelter]
            field = blend_fields(fields, weights)
            current_alpha = blend_alpha(weights)
            dom = int(np.argmax(weights))
            state_name = STATES[dom]
            episode_label = EPISODE_LABEL[state_name]
            field_name = STATE_TO_FIELD[state_name]

        # --- advance the animal one step: nearest-object sensing + current field ---
        obs = encode_observation(state["pos"], objs,
                                 food_strengths=state["food_strengths"])       # [6]
        with torch.no_grad():
            v_pred = evaluate_vector_field(
                net, torch.tensor(obs[None, :], dtype=torch.float32), field
            ).numpy().ravel()
        # While resting at a shelter the agent stays perfectly still (a clear visual
        # "at rest").  Otherwise it moves at constant speed along the NNN's predicted
        # HEADING (its magnitude vanishes near objects and would otherwise stall it);
        # velocity_smoothing low-passes the heading to damp jitter.
        resting = (demo_mode == "scripted" and state["rest"] > 0)
        if resting:
            state["vel"] = np.zeros(2, dtype=np.float32)     # stationary; drop heading
        else:
            state["vel"] = ((1.0 - velocity_smoothing) * state["vel"]
                            + velocity_smoothing * v_pred).astype(np.float32)
            heading = state["vel"]
            hn = float(np.linalg.norm(heading))
            if hn > 1e-3:                                # deadzone: hold if no signal
                step = dt * cruise_speed * heading / hn
                state["pos"] = np.clip(state["pos"] + step, -1.0, 1.0).astype(np.float32)

        # --- closed-loop demo dynamics (never used in training) ---
        # Reactive demo respawns food so the world stays populated for the loop.
        # Eating only PARTIALLY sates hunger, so after one food the agent may still
        # be hungry and move on to another food ("food after food").
        ate = apply_food_depletion(state["pos"], objs, state["food_strengths"],
                                   eat_radius,
                                   respawn=(food_respawn or demo_mode == "scripted"))
        if ate:
            state["hunger"] = max(0.0, state["hunger"] - eat_amount)

        state["trail"].append(state["pos"].copy())
        if len(state["trail"]) > 160:
            state["trail"] = state["trail"][-160:]
        trail = np.array(state["trail"])
        n_food_left = int(np.sum(state["food_strengths"] > 0.0))
        n_food = state["food_strengths"].shape[0]

        # --- Panel 1: world + trajectory ---
        ax_env.clear()
        draw_objects(ax_env, objs, state["food_strengths"],
                     shelter_region_radius=SHELTER_REGION_RADIUS)
        if len(trail) > 1:
            ax_env.plot(trail[:, 0], trail[:, 1], "-", color="0.4", lw=1.5, alpha=0.85)
        ax_env.scatter(state["pos"][0], state["pos"][1], s=150, color="black",
                       zorder=6, label="agent")
        ax_env.set_xlim(-1, 1)
        ax_env.set_ylim(-1, 1)
        ax_env.set_aspect("equal")
        ax_env.set_title(f"Behavior: {episode_label}")
        ax_env.grid(alpha=0.3)

        # --- Panel 2: predicted vector field over candidate positions ---
        # Each grid position is encoded through the CURRENT objects/strengths.
        ax_quiver.clear()
        draw_objects(ax_quiver, objs, state["food_strengths"],
                     shelter_region_radius=SHELTER_REGION_RADIUS)
        q_obs = encode_observations(q_positions, objs,
                                    food_strengths=state["food_strengths"])
        with torch.no_grad():
            vq = evaluate_vector_field(
                net, torch.tensor(q_obs, dtype=torch.float32), field
            ).numpy()
        ax_quiver.quiver(q_positions[:, 0], q_positions[:, 1], vq[:, 0], vq[:, 1],
                         color="0.25", angles="xy", scale=22, width=0.004)
        if show_reference:
            # Optional diagnostic: analytic target field for the current alpha.
            vref = make_mixed_behavior_targets(q_obs, current_alpha, gamma=gamma)
            ax_quiver.quiver(q_positions[:, 0], q_positions[:, 1], vref[:, 0], vref[:, 1],
                             color="tab:orange", angles="xy", scale=22, width=0.003,
                             alpha=0.5)
        ax_quiver.set_xlim(-1, 1)
        ax_quiver.set_ylim(-1, 1)
        ax_quiver.set_aspect("equal")
        title2 = ("Recruited vector field (blended noise field)"
                  if demo_mode == "scripted"
                  else f"Recruited vector field ('{field_name}' noise field)")
        ax_quiver.set_title(title2)
        ax_quiver.grid(alpha=0.3)

        # --- Panel 3: current noise field ---
        ax_field.clear()
        heat = field.numpy().reshape(GRID_SIZE, GRID_SIZE)
        ax_field.imshow(heat, origin="lower", extent=[0, 1, 0, 1],
                        cmap="magma", vmin=0.0, vmax=vmax)
        ax_field.set_title(
            f"Blended noise field  F={weights[0]:.2f}, "
            f"T={weights[1]:.2f}, S={weights[2]:.2f}"
        )
        ax_field.set_xticks([])
        ax_field.set_yticks([])

        # --- Dynamic title ---
        if demo_mode == "scripted":
            suptitle.set_text(
                "Perception -> blended neuromodulatory field -> recruited "
                "subnetwork -> behavior (one shared weight set)\n"
                f"dominant behavior: {episode_label}    weights "
                f"F={weights[0]:.2f}, T={weights[1]:.2f}, S={weights[2]:.2f}\n"
                f"hunger: {state['hunger']:.2f}   shelter need: "
                f"{state['shelter_need']:.2f}   food available: {n_food_left}/{n_food}"
            )
        else:
            suptitle.set_text(
                "Continuous neuromodulatory transition (cycle demo)\n"
                f"weights F={weights[0]:.2f}, T={weights[1]:.2f}, S={weights[2]:.2f}"
                f"    dominant: {episode_label}\n"
                f"food available: {n_food_left}/{n_food}"
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Neuromodulator-like noise fields recruit different functional "
                    "subnetworks from ONE shared-weight NNN, producing Foraging, "
                    "Avoidance, or Sheltering.  The default 'scripted' demo is a "
                    "reactive closed loop (perception -> neuromodulatory state -> "
                    "recruited subnetwork -> behavior) with wandering threats.",
        epilog="examples:\n"
               "  python examples/neuromodulated_behavior_modes.py\n"
               "  python examples/neuromodulated_behavior_modes.py --epochs 800\n"
               "  python examples/neuromodulated_behavior_modes.py --threat-motion static\n"
               "  python examples/neuromodulated_behavior_modes.py --demo-mode pure-panels\n"
               "  python examples/neuromodulated_behavior_modes.py --demo-mode cycle --epochs 1000\n",
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
    p.add_argument("--seed",        type=int,   default=7,
                   help="Random seed; 7 chosen to minimise wall-hugging / U-turns [7]")
    # Demo scene / schedule.
    p.add_argument("--layout", choices=("scripted", "random"), default="scripted",
                   help="Object layout: 'scripted' fixed scene (default) or 'random'")
    p.add_argument("--demo-mode", choices=("scripted", "cycle", "pure-panels"),
                   default="scripted",
                   help="'scripted' Foraging->Avoidance->Sheltering episodes "
                        "(default), 'cycle' smooth interpolation, or 'pure-panels' "
                        "static three-field diagnostic")
    # Environment (random layout only).
    p.add_argument("--n-food",      type=int,   default=5,
                   help="Number of food objects (random layout) [5]")
    p.add_argument("--n-threat",    type=int,   default=3,
                   help="Number of threat objects (random layout) [3]")
    p.add_argument("--n-shelter",   type=int,   default=2,
                   help="Number of shelter objects (random layout) [2]")
    p.add_argument("--dynamic-objects", action="store_true",
                   help="Slowly move objects during the animation (demo only)")
    p.add_argument("--object-speed", type=float, default=0.002,
                   help="Per-frame speed of dynamic objects [0.002]")
    # Closed-loop demo dynamics (animation only).
    p.add_argument("--eat-radius",    type=float, default=0.10,
                   help="Distance at which food is consumed (strength -> 0) [0.10]")
    p.add_argument("--food-respawn",  action="store_true",
                   help="Let consumed food recover after the agent moves clear "
                        "(default: consumed food stays depleted)")
    p.add_argument("--shelter-radius", type=float, default=0.08,
                   help="Arrival radius that counts as 'reached shelter' -- kept "
                        "clearly smaller than the 0.125 region radius [0.08]")
    # Reactive scripted closed loop (constant speed + context/need arbitration).
    p.add_argument("--cruise-speed", type=float, default=0.9,
                   help="Constant agent speed along the predicted heading [0.9]")
    p.add_argument("--hunger-rate", type=float, default=0.006,
                   help="Per-frame rise of hunger (reset to 0 by eating) [0.006]")
    p.add_argument("--need-rate", type=float, default=0.006,
                   help="Per-frame rise of shelter need (reset by sheltering) [0.006]")
    p.add_argument("--eat-amount", type=float, default=0.6,
                   help="Hunger removed per food eaten (<1 => visit several) [0.6]")
    p.add_argument("--rest-frames", type=int, default=50,
                   help="How long the agent rests once it reaches shelter [50]")
    p.add_argument("--threat-gain", type=float, default=1.7,
                   help="Gain mapping threat urgency to blend weight in [0,1] "
                        "(higher => flee sooner/stronger) [1.7]")
    p.add_argument("--threat-range", type=float, default=0.40,
                   help="Distance scale of the graded threat urgency [0.40]")
    p.add_argument("--neuromod-smoothing", type=float, default=0.12,
                   help="Low-pass on the neuromodulatory weights in [0,1]: smaller "
                        "= smoother field blending, more commitment [0.12]")
    p.add_argument("--threat-motion", choices=("moving", "static"), default="moving",
                   help="Scripted demo: 'moving' threats wander (default) or 'static'")
    p.add_argument("--threat-speed", type=float, default=0.01,
                   help="Per-frame speed of wandering threats (moving) [0.01]")
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

    # --- Multi-object environment (static during training) ---
    if args.layout == "scripted":
        objects = make_scripted_objects()
    else:
        objects = make_objects(rng, args.n_food, args.n_threat, args.n_shelter)
    velocities = make_object_velocities(rng, objects, args.object_speed)
    print(f"Layout '{args.layout}': "
          + ", ".join(f"{k}={objects[k].shape[0]}" for k in CATEGORIES))

    # --- Data: positions -> nearest-object observations -> behavior targets ---
    positions_np = make_training_grid(args.grid_side)
    obs_np = encode_observations(positions_np, objects)                  # [N, 6]
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

    # --- Visualize the trained system ---
    if args.demo_mode == "pure-panels":
        print("\nOpening static three-field diagnostic (close it to exit)...")
        show_pure_panels(net, objects, fields, args.target_gamma)
        return

    print("\nOpening animation window (close it to exit)...")
    if args.demo_mode == "scripted":
        print("  reactive demo: threat near -> Avoidance; else Foraging / Sheltering.")
        print(f"  threats: {args.threat_motion}"
              + (f" (speed {args.threat_speed})" if args.threat_motion == "moving" else ""))
    if args.dynamic_objects:
        print("  dynamic objects ON: behavior tracks the moving salience structure.")
    animate(net, objects, fields, history,
            args.target_gamma, args.anim_frames, args.demo_mode, args.layout,
            args.eat_radius, args.shelter_radius, args.food_respawn,
            cruise_speed=args.cruise_speed,
            hunger_rate=args.hunger_rate, need_rate=args.need_rate,
            eat_amount=args.eat_amount, rest_frames=args.rest_frames,
            threat_gain=args.threat_gain, threat_range=args.threat_range,
            neuromod_smoothing=args.neuromod_smoothing,
            threat_motion=args.threat_motion, threat_speed=args.threat_speed,
            seed=args.seed,
            dynamic=args.dynamic_objects, velocities=velocities,
            show_reference=args.show_reference,
            velocity_smoothing=args.velocity_smoothing)


if __name__ == "__main__":
    main()
