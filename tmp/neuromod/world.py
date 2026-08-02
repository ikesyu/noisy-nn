"""The standard Foraging / Avoidance / Sheltering problem.

This module owns everything that defines the benchmark and nothing that depends
on how it is solved or drawn: the scene, the sensory encoding, the supervised
behaviour targets, and the closed-loop drive dynamics that decide which
neuromodulatory state is active.  Experiments should treat it as fixed.

Sensory encoding
----------------
The 6D input is the RAW relative vector to the NEAREST available object of each
category: [r_food, r_threat, r_shelter].  Nearest-object sensing scales to
arbitrarily many objects and avoids the centroid attractor that summing over
objects creates.  Using the raw vector rather than the unit vector means the
signal goes to zero exactly at the object, so the agent decelerates onto its
target instead of orbiting it.  The network never sees absolute position, so a
policy learned on a static scene keeps working when objects move.

Behaviour target
----------------
    z = a_food * r_food - a_threat * r_threat + a_shelter * r_shelter
    v = tanh(gamma * ||z||) * z / (||z|| + eps)

Because r is the raw relative vector, ||z|| is LARGE when the relevant objects
are far, so tanh saturates and the agent travels at full speed; ||z|| shrinks on
approach, so the speed law is a deceleration-on-arrival rule.  This is
supervised vector-field regression: no reward, no policy gradient.

Closed loop
-----------
`neuromod_weights` turns the graded drives (hunger, distance-graded threat
urgency, shelter need) into a continuous 3-vector that BLENDS the three noise
fields, like an overlapping mixture of neuromodulator concentrations rather than
a hard switch.  It selects the field and computes no movement; all approach and
avoid geometry comes from the network.
"""
from __future__ import annotations

import numpy as np


CATEGORIES = ("food", "threat", "shelter")
STATES = ("food_biased", "threat_biased", "shelter_biased")

# Each neuromodulatory state weights the three drives [food, threat, shelter].
#
# NOTE (see docs/idea_neuromod.md section 4): these are near-one-hot, which makes
# each recruited behaviour cleanly classifiable but also makes the three tasks
# use nearly DISJOINT input dimensions.  Multiplexing is then almost free, which
# weakens the shared-weight claim.  `--alpha-mix` on the driver raises the
# off-drives so the tasks genuinely compete; the default preserves the original
# demo.
ALPHA_STATES = {
    "food_biased":    np.array([1.80, 0.03, 0.03], dtype=np.float32),
    "threat_biased":  np.array([0.03, 2.20, 0.03], dtype=np.float32),
    "shelter_biased": np.array([0.03, 0.03, 1.80], dtype=np.float32),
}

EPISODE_LABEL = {
    "food_biased": "Foraging",
    "threat_biased": "Avoidance",
    "shelter_biased": "Sheltering",
}
FIELD_EPISODE = {"food": "Foraging", "threat": "Avoidance", "shelter": "Sheltering"}
STATE_TO_FIELD = {
    "food_biased": "food",
    "threat_biased": "threat",
    "shelter_biased": "shelter",
}

# Agent start position for the scripted demo (near the shelter side, on the left).
SCRIPTED_START = np.array([-0.55, 0.05], dtype=np.float32)

# Radius of the drawn circular shelter region.  The arrival-detection radius is
# kept clearly smaller, and threats are reflected at 1.5x this radius so they
# never enter the region.
SHELTER_REGION_RADIUS = 0.125
THREAT_KEEPOUT_RADIUS = 1.5 * SHELTER_REGION_RADIUS

# Wandering threats bounce inside this inner box so they stay central and do not
# push the agent onto the map edges.
THREAT_BOUNDS = 0.72

# Goal commitment thresholds: forage until hunger falls to HUNGER_LO (possibly
# eating several foods), then shelter until rested to SHELTER_LO.  This 1-bit
# commitment prevents food<->shelter dithering; threat blends over it continuously.
HUNGER_LO = 0.35
SHELTER_LO = 0.15


def alpha_states(mix: float = None) -> dict[str, np.ndarray]:
    """Drive weights per state, optionally with the off-drives raised to `mix`.

    `mix` is the off-drive weight as a fraction of the dominant one.  The default
    (None) returns the original near-one-hot weights.  Larger values make the
    three behaviours share input dimensions, so storing them in one weight set
    becomes a real multiplexing feat rather than three independent maps.
    """
    if mix is None:
        return {s: a.copy() for s, a in ALPHA_STATES.items()}
    out = {}
    for state, alpha in ALPHA_STATES.items():
        dominant = float(alpha.max())
        new = np.full_like(alpha, dominant * mix)
        new[int(np.argmax(alpha))] = dominant
        out[state] = new
    return out


# ============================================================
# Scene
# ============================================================

def make_objects(rng: np.random.Generator, n_food: int, n_threat: int,
                 n_shelter: int) -> dict[str, np.ndarray]:
    """Random object positions for each category, each of shape [n_k, 2]."""
    counts = {"food": n_food, "threat": n_threat, "shelter": n_shelter}
    return {
        key: rng.uniform(-1.0, 1.0, size=(counts[key], 2)).astype(np.float32)
        for key in CATEGORIES
    }


def make_scripted_objects() -> dict[str, np.ndarray]:
    """Fixed multi-object scene: food right, shelter left, threats in the middle
    band the agent must cross.  Threats sit OFF the direct food<->shelter route so
    the agent detours past them instead of oscillating in a flee<->approach loop.
    """
    return {
        "food":    np.array([[0.42, 0.45], [0.62, 0.10]], dtype=np.float32),
        "threat":  np.array([[0.05, 0.20], [0.15, -0.30], [-0.10, -0.05]],
                            dtype=np.float32),
        "shelter": np.array([[-0.64, 0.52], [-0.28, -0.58]], dtype=np.float32),
    }


def make_object_velocities(rng: np.random.Generator, objects: dict[str, np.ndarray],
                           speed: float) -> dict[str, np.ndarray]:
    """A small constant velocity (random direction) per object."""
    vels = {}
    for key in CATEGORIES:
        n = objects[key].shape[0]
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
        vels[key] = (speed * np.stack([np.cos(theta), np.sin(theta)], axis=1)
                     ).astype(np.float32)
    return vels


def update_dynamic_objects(objects: dict[str, np.ndarray],
                           velocities: dict[str, np.ndarray]) -> None:
    """Advance all objects and bounce them off the walls of [-1, 1]^2 (in place)."""
    for key in CATEGORIES:
        objects[key] += velocities[key]
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
    """Advance ONLY the threats and bounce them inside [-bounds, bounds]^2.

    A wandering threat leaves the food<->shelter corridor on its own, which
    dissolves the deadlock a parked threat would cause.  `shelter_keepout > 0`
    also keeps threats out of shelter regions so a resting agent is undisturbed.
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
                    t[i] = (s + n * shelter_keepout).astype(np.float32)
                    vn = float(np.dot(threat_vels[i], n))
                    if vn < 0.0:
                        threat_vels[i] = (threat_vels[i] - 2.0 * vn * n).astype(np.float32)
        np.clip(t, -1.0, 1.0, out=t)


# ============================================================
# Sensing and targets
# ============================================================

def nearest_relative_vector(position: np.ndarray, object_positions: np.ndarray,
                            available: np.ndarray = None) -> np.ndarray:
    """Raw relative vector to the nearest available object of one category.

    Objects marked unavailable (e.g. eaten food) are ignored; a zero vector is
    returned when the category is empty or fully unavailable.
    """
    if object_positions.shape[0] == 0:
        return np.zeros(2, dtype=np.float32)
    rel = object_positions - position[None, :]
    dist = np.linalg.norm(rel, axis=1)
    if available is not None:
        dist = np.where(np.asarray(available) > 0.0, dist, np.inf)
    j = int(np.argmin(dist))
    if not np.isfinite(dist[j]):
        return np.zeros(2, dtype=np.float32)
    return rel[j].astype(np.float32)


def encode_observation(position: np.ndarray, objects: dict[str, np.ndarray],
                       food_strengths: np.ndarray = None) -> np.ndarray:
    """6D nearest-object sensory input for one agent position.

    Perception is pure geometry: internal drives are NOT mixed in here, they only
    gate WHICH noise fields are recruited (see `neuromod_weights`).
    """
    return np.concatenate([
        nearest_relative_vector(position, objects["food"], available=food_strengths),
        nearest_relative_vector(position, objects["threat"]),
        nearest_relative_vector(position, objects["shelter"]),
    ]).astype(np.float32)


def encode_observations(positions: np.ndarray, objects: dict[str, np.ndarray],
                        food_strengths: np.ndarray = None) -> np.ndarray:
    """Batch encoder: [N, 2] positions -> [N, 6] nearest-object observations."""
    return np.stack([encode_observation(p, objects, food_strengths)
                     for p in positions], axis=0)


def make_mixed_behavior_targets(observations: np.ndarray, alpha: np.ndarray,
                                gamma: float = 2.0, eps: float = 1e-8) -> np.ndarray:
    """Bounded mixed-velocity targets from observations and drive weights.

    Approach food, flee threat (note the minus), approach shelter, all at once.
    """
    r_food, r_threat, r_shelter = (observations[:, 0:2], observations[:, 2:4],
                                   observations[:, 4:6])
    z = alpha[0] * r_food - alpha[1] * r_threat + alpha[2] * r_shelter
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    return (np.tanh(gamma * norm) * z / (norm + eps)).astype(np.float32)


def make_training_grid(grid_side: int) -> np.ndarray:
    """A regular grid of agent positions over [-1, 1]^2, [grid_side**2, 2]."""
    axis = np.linspace(-1.0, 1.0, grid_side, dtype=np.float32)
    gx, gy = np.meshgrid(axis, axis)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


# ============================================================
# Closed-loop drives and neuromodulatory arbitration
# ============================================================

def initialize_demo_state(objects: dict[str, np.ndarray], layout: str) -> dict:
    """Agent state plus the internal drives that gate the noise field."""
    start = SCRIPTED_START.copy() if layout == "scripted" \
        else np.zeros(2, dtype=np.float32)
    return {
        "pos": start.copy(),
        "start": start.copy(),
        "vel": np.zeros(2, dtype=np.float32),               # smoothed heading
        "speed": 0.0,                                       # last step length / dt
        "trail": [start.copy()],
        "objects": {k: objects[k].copy() for k in CATEGORIES},
        "food_strengths": np.ones(objects["food"].shape[0], dtype=np.float32),
        "hunger": 1.0,
        "shelter_need": 0.0,
        "goal": "food",
        "rest": 0,
        "w": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }


def apply_food_depletion(position: np.ndarray, objects: dict[str, np.ndarray],
                         food_strengths: np.ndarray, eat_radius: float,
                         respawn: bool = False) -> bool:
    """Deplete any food the agent reaches (in place); True if one was eaten."""
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
    """True if the agent is inside any shelter's arrival radius."""
    shelter = objects["shelter"]
    if shelter.shape[0] == 0:
        return False
    return float(np.linalg.norm(shelter - position[None, :], axis=1).min()) < shelter_radius


def neuromod_weights(position: np.ndarray, objects: dict[str, np.ndarray],
                     hunger: float, shelter_need: float, goal: str,
                     prev_w: np.ndarray, threat_gain: float, threat_range: float,
                     smoothing: float) -> tuple:
    """Continuous neuromodulatory weights [food, threat, shelter] and updated goal.

    Two levels, deliberately simple: a committed food/shelter GOAL (1 bit) that
    switches only at the satiation thresholds, and a distance-graded threat
    urgency that continuously cross-fades the threat field over that goal.  The
    whole vector is low-pass filtered, which prevents the chattering a hard rule
    produces and blends the recruited subnetworks like overlapping concentrations.
    """
    if goal == "food" and hunger <= HUNGER_LO:
        goal = "shelter"
    elif goal == "shelter" and shelter_need <= SHELTER_LO:
        goal = "food"
    base = np.array([1.0, 0.0, 0.0] if goal == "food" else [0.0, 0.0, 1.0],
                    dtype=np.float32)

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


def blend_alpha(weights: np.ndarray, alphas: dict[str, np.ndarray]) -> np.ndarray:
    """Interpolated drive weights matching a blended noise field."""
    return (weights[0] * alphas["food_biased"]
            + weights[1] * alphas["threat_biased"]
            + weights[2] * alphas["shelter_biased"])


def cyclic_weights(phase: float, beta: float = 3.0) -> np.ndarray:
    """Smooth cyclic softmax weights over the three states, order [F, T, S].

    Three phase-shifted cosines make each state dominate in turn; softmax gives a
    smooth normalized transition rather than a hard switch.
    """
    logits = beta * np.array([
        np.cos(phase),
        np.cos(phase - 2.0 * np.pi / 3.0),
        np.cos(phase - 4.0 * np.pi / 3.0),
    ])
    z = logits - np.max(logits)
    e = np.exp(z)
    return (e / e.sum()).astype(np.float32)


def step_agent(state: dict, v_pred: np.ndarray, dt: float, speed_gain: float,
               mode: str = "learned", smoothing: float = 0.2,
               resting: bool = False) -> float:
    """Advance the agent one frame along the network's predicted velocity.

    `mode` decides whether the network's output MAGNITUDE is used:

        "learned"  step = dt * speed_gain * v.  The learned tanh speed law
                   survives, so the agent decelerates on approach AND freezes
                   when the network falls silent.  Required for behaviour-level
                   stochastic resonance: normalising the magnitude away hides the
                   low-noise collapse, which is the left arm of the inverted U
                   (docs/idea_neuromod.md section 7.1).
        "cruise"   direction only at constant speed.  The original demo look;
                   keep for figures where a constant-speed trace is easier to read.

    Returns the smoothed speed |v|, which is worth plotting: it is the channel
    that carries the collapse.
    """
    if resting:
        state["vel"] = np.zeros(2, dtype=np.float32)
        state["speed"] = 0.0
        return 0.0

    state["vel"] = ((1.0 - smoothing) * state["vel"]
                    + smoothing * v_pred).astype(np.float32)
    v = state["vel"]
    speed = float(np.linalg.norm(v))
    state["speed"] = speed

    if mode == "learned":
        step = dt * speed_gain * v
    else:
        step = dt * speed_gain * v / speed if speed > 1e-3 else np.zeros(2, np.float32)

    state["pos"] = np.clip(state["pos"] + step, -1.0, 1.0).astype(np.float32)
    return speed


def advance_drives(state: dict, shelter_radius: float, hunger_rate: float,
                   need_rate: float, rest_frames: int) -> None:
    """One frame of hunger / shelter-need / rest bookkeeping (in place).

    Sheltering uses a time-based REST: on first arrival a countdown starts, and
    when it elapses the shelter need clears.  A time-based rest is robust to a
    wandering threat briefly nudging the agent off the shelter, unlike a
    "decay only while inside" rule which such a threat can stall forever.
    """
    inside = apply_shelter_satisfaction(state["pos"], state["objects"], shelter_radius)
    state["hunger"] = min(1.0, state["hunger"] + hunger_rate)

    if state["goal"] == "shelter":
        if inside and state["rest"] == 0:
            state["rest"] = rest_frames
        if state["rest"] > 0:
            state["rest"] -= 1
            state["shelter_need"] = 1.0
            if state["rest"] == 0:
                state["shelter_need"] = 0.0
        else:
            state["shelter_need"] = min(1.0, state["shelter_need"] + need_rate)
    else:
        state["shelter_need"] = min(1.0, state["shelter_need"] + need_rate)
        state["rest"] = 0
