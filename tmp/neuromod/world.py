"""The standard Foraging / Avoidance / Sheltering problem.

The three behaviours are not a scenario choice.  They are the behavioural image
of the three minimal internal axes any neuromodulator-dynamics model must carry
-- a circadian clock (rest/activity), hunger (feeding drive), and threat
urgency (defense) -- mapped onto one arena.  Fewer behaviours and the claims
(partial recruitment, controlled overlap, behaviour-level SR) become untestable;
more and nothing is gained.  The derivation is docs/idea_neuromod.md, C.0.

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
A circadian clock commits the slow goal (forage by day, shelter by night);
`neuromod_weights` cross-fades the threat field over that goal by proximity,
gated by hunger (courage) and vetoed at point-blank range (panic).  The result
is a continuous 3-vector that BLENDS the three noise fields, like an
overlapping mixture of neuromodulator concentrations rather than a hard
switch.  It selects the field and computes no movement; all approach and avoid
geometry comes from the network.
"""
from __future__ import annotations

import dataclasses

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

# The animal starts the demo AT a den, which is where a night ends anyway, so
# frame 0 is dawn rather than an arbitrary drop-in.  Keep this on a shelter of
# `make_scripted_objects` if you move the scene.
SCRIPTED_START = np.array([-0.40, 0.52], dtype=np.float32)

# Radius of the drawn circular shelter region.  The arrival-detection radius is
# kept clearly smaller, and threats are reflected at 1.5x this radius so they
# never enter the region.
SHELTER_REGION_RADIUS = 0.125
# Predators are held well clear of the dens.  1.5x let a wanderer sit 0.19 from a
# shelter -- inside `threat_range` (0.50), so the flee blend fought the arrival
# drive right at the door and the animal failed to get home on ~9% of nights.
# The refuge override exists for exactly that, but `panic_range` now vetoes it at
# point-blank range, so the geometry has to do the work instead.  Measured on the
# scripted scene: night-home rate 0.914 -> 0.992, wall dwell and threat contact
# both to zero, foraging up.
THREAT_KEEPOUT_RADIUS = 3.6 * SHELTER_REGION_RADIUS

# Wandering threats bounce inside this inner box so they stay central and do not
# push the agent onto the map edges.
THREAT_BOUNDS = 0.72

# Behavioural-artefact thresholds, used by `rollout` to score a run.  These name
# the three things that make a trajectory look wrong to a viewer:
#   - touching a threat at all (the avoidance behaviour has failed outright),
#   - sitting in the boundary band (the "glued to the wall" artefact),
#   - stopping while neither resting in a shelter nor decelerating onto food.
THREAT_CONTACT_RADIUS = 0.08
WALL_BAND = 0.04          # |pos| within this of 1.0 counts as at the wall
STALL_SPEED = 0.05        # |v| below this, outside a rest, counts as a stall

# ------------------------------------------------------------
# Circadian rhythm (default goal commitment)
# ------------------------------------------------------------
# Foraging and sheltering alternate on a free-running internal clock, not on
# satiation: the animal goes home because it is dusk, not because it is full.
# DAY_FRACTION is the share of the period spent foraging; the remainder is the
# night, and the night has to be long enough to cover the TRAVEL to a shelter as
# well as the sleep, or the animal never gets home before dawn.
CIRCADIAN_PERIOD = 420    # frames per full day+night cycle
DAY_FRACTION = 0.62       # share of the period spent foraging

# Hunger level at which foraging courage starts to build (risk_hunger gate in
# neuromod_weights).  Below this the threat urgency is untouched, so the fed
# agent's avoidance stays legible; full courage is reached at hunger 1.
RISK_HUNGER_ONSET = 0.6

# ------------------------------------------------------------
# Sensing configuration (Stage 1, docs/idea_neuromod.md)
# ------------------------------------------------------------
# "sector": K angular sectors x 4 channels (food, threat, shelter, wall) -- the
#           STANDARD.  All behaviours read the SAME angular substrate, which
#           makes the one-network multiplexing claim structural instead of
#           accidental, and walls become perceivable instead of a teacher-side
#           hack.  The 6D vector code also makes the teacher near-linear, which
#           is the sec4 triviality critique.
# "vector": the original 6D nearest-object relative vectors, kept ONLY so the
#           older experiment scripts reproduce their recorded runs; those
#           scripts pin it explicitly via set_sensing("vector").
SENSING = "sector"
N_SECTORS = 8
SECTOR_CENTERS = 2.0 * np.pi * np.arange(N_SECTORS) / N_SECTORS
SECTOR_DIRS = np.stack([np.cos(SECTOR_CENTERS), np.sin(SECTOR_CENTERS)],
                       axis=1).astype(np.float32)          # [K, 2]
SENSE_LAMBDA = 2.0      # object-channel distance constant  g(d) = exp(-d/lambda);
                        # 1.0 measured too short-sighted (far food ~0.14 leaves the
                        # return drive after a flee too weak; wall-band dwell rises)
WALL_LAMBDA = 0.25      # wall-channel constant on the ray distance to the box
SENSE_CHANNELS = ("food", "threat", "shelter", "wall")


def set_sensing(mode: str) -> None:
    if mode not in ("vector", "sector"):
        raise ValueError(f"unknown sensing mode {mode!r}")
    global SENSING
    SENSING = mode


def set_sectors(k: int) -> None:
    """Change the angular resolution K (rebuilds the derived sector tables)."""
    global N_SECTORS, SECTOR_CENTERS, SECTOR_DIRS
    N_SECTORS = int(k)
    SECTOR_CENTERS = 2.0 * np.pi * np.arange(N_SECTORS) / N_SECTORS
    SECTOR_DIRS = np.stack([np.cos(SECTOR_CENTERS), np.sin(SECTOR_CENTERS)],
                           axis=1).astype(np.float32)


def obs_dim() -> int:
    return 6 if SENSING == "vector" else N_SECTORS * len(SENSE_CHANNELS)


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

def make_scripted_objects() -> dict[str, np.ndarray]:
    """Fixed multi-object scene, laid out against three measured constraints.

    The earlier layout put all three threats in one central clump (minimum
    pairwise separation 0.29, centroid on the origin) and both foods on the
    right within 0.40 of each other, which let the animal farm one corner: it
    took up to 10 foods in a single day, averaging 4.5.

    This layout was chosen by rejection-sampling positions under separation
    constraints and then SCORING the candidates by running the closed loop
    (4 networks x 8 episodes x 1680 frames).  Spread was searched over ALL
    THREE categories at once, including the dens -- an earlier search pinned
    the dens on the left, which forced the food left too and produced the
    lopsided scene this one replaces.  Two limits emerged from the
    measurements and both are load-bearing:

        no object past |0.6|   objects nearer the rim tow the animal into the
                               boundary band -- a maximin layout over the full
                               box measured 3-7% wall dwell against 0.4% here
        dens at the periphery  a central den is swept by the wandering threats,
                               and the animal is repelled from its own refuge:
                               night-home rate 0.03-0.61 for centred dens

    Honest note on the trade-off: an evenly spread scene costs reliability.  This
    layout measures a 0.86 night-home rate against 1.00 for the older compact
    scene, because at dusk the animal is more often far from a den.  Over 30
    candidates across four search strategies, no spread layout beat 0.86.  The
    compact alternative is one edit away if the failed nights matter more than
    the look.
    """
    return {
        "food":    np.array([[0.52, -0.08], [-0.37, 0.08], [0.55, -0.55]],
                            dtype=np.float32),
        "threat":  np.array([[0.05, -0.07], [0.42, 0.52], [-0.51, -0.43]],
                            dtype=np.float32),
        "shelter": np.array([[-0.40, 0.52], [0.10, -0.56]], dtype=np.float32),
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


def _sector_tuning(theta: np.ndarray) -> np.ndarray:
    """Raised-cosine angular tuning per sector, [K] per angle -> [K, n].

    w_k(theta) = cos^2(K (theta - phi_k) / 4) on |theta - phi_k| <= 2pi/K,
    else 0.  Adjacent sectors overlap at half height and the K curves sum to 1
    (partition of unity), so the code is smooth as an object sweeps the sheet.
    """
    d = (theta[None, :] - SECTOR_CENTERS[:, None] + np.pi) % (2 * np.pi) - np.pi
    w = np.cos(N_SECTORS * d / 4.0) ** 2
    return np.where(np.abs(d) <= 2.0 * np.pi / N_SECTORS, w, 0.0)


def _sector_channel(position: np.ndarray, object_positions: np.ndarray,
                    available: np.ndarray = None) -> np.ndarray:
    """[K] sector code for one object category: s_k = max_j w_k(theta_j) g(d_j)."""
    if object_positions.shape[0] == 0:
        return np.zeros(N_SECTORS, dtype=np.float32)
    rel = object_positions - position[None, :]
    if available is not None:
        rel = rel[np.asarray(available) > 0.0]
        if rel.shape[0] == 0:
            return np.zeros(N_SECTORS, dtype=np.float32)
    d = np.linalg.norm(rel, axis=1)
    g = np.exp(-d / SENSE_LAMBDA)
    w = _sector_tuning(np.arctan2(rel[:, 1], rel[:, 0]))       # [K, n]
    return (w * g[None, :]).max(axis=1).astype(np.float32)


def _wall_channel(position: np.ndarray) -> np.ndarray:
    """[K] wall proximity: s_k = exp(-t_k / lambda_w), t_k = ray distance to
    the [-1, 1]^2 boundary along the sector centre direction."""
    out = np.zeros(N_SECTORS, dtype=np.float32)
    for k in range(N_SECTORS):
        u = SECTOR_DIRS[k]
        t = np.inf
        for a in range(2):
            if abs(u[a]) > 1e-9:
                t = min(t, (np.sign(u[a]) * 1.0 - position[a]) / u[a])
        out[k] = np.exp(-max(t, 0.0) / WALL_LAMBDA)
    return out


def encode_observation(position: np.ndarray, objects: dict[str, np.ndarray],
                       food_strengths: np.ndarray = None) -> np.ndarray:
    """Sensory input for one agent position, in the module's SENSING mode.

    Perception is pure geometry: internal drives are NOT mixed in here, they only
    gate WHICH noise fields are recruited (see `neuromod_weights`).

    "vector" (original): 6D raw relative vector to the nearest available object
    of each category.  "sector" (Stage 1): K angular sectors x (food, threat,
    shelter, wall) bounded proximities -- every behaviour reads the same
    angular substrate, and walls are perceivable.
    """
    if SENSING == "vector":
        return np.concatenate([
            nearest_relative_vector(position, objects["food"], available=food_strengths),
            nearest_relative_vector(position, objects["threat"]),
            nearest_relative_vector(position, objects["shelter"]),
        ]).astype(np.float32)
    return np.concatenate([
        _sector_channel(position, objects["food"], available=food_strengths),
        _sector_channel(position, objects["threat"]),
        _sector_channel(position, objects["shelter"]),
        _wall_channel(position),
    ]).astype(np.float32)


def _batch_nearest_rel(positions: np.ndarray, object_positions: np.ndarray,
                       available: np.ndarray = None) -> np.ndarray:
    """Vectorised nearest_relative_vector: [N, 2] x [M, 2] -> [N, 2]."""
    n = positions.shape[0]
    if object_positions.shape[0] == 0:
        return np.zeros((n, 2), dtype=np.float32)
    rel = object_positions[None, :, :] - positions[:, None, :]      # [N, M, 2]
    dist = np.linalg.norm(rel, axis=2)                              # [N, M]
    if available is not None:
        dist = np.where(np.asarray(available)[None, :] > 0.0, dist, np.inf)
    j = dist.argmin(axis=1)
    out = rel[np.arange(n), j]
    out[~np.isfinite(dist[np.arange(n), j])] = 0.0
    return out.astype(np.float32)


def _batch_sector_channel(positions: np.ndarray, object_positions: np.ndarray,
                          available: np.ndarray = None) -> np.ndarray:
    """Vectorised sector code: [N, 2] x [M, 2] -> [N, K]."""
    n = positions.shape[0]
    if object_positions.shape[0] == 0:
        return np.zeros((n, N_SECTORS), dtype=np.float32)
    obj = object_positions
    if available is not None:
        obj = obj[np.asarray(available) > 0.0]
        if obj.shape[0] == 0:
            return np.zeros((n, N_SECTORS), dtype=np.float32)
    rel = obj[None, :, :] - positions[:, None, :]                   # [N, M, 2]
    d = np.linalg.norm(rel, axis=2)                                 # [N, M]
    g = np.exp(-d / SENSE_LAMBDA)
    theta = np.arctan2(rel[..., 1], rel[..., 0])                    # [N, M]
    dth = (theta[:, :, None] - SECTOR_CENTERS[None, None, :]
           + np.pi) % (2 * np.pi) - np.pi                           # [N, M, K]
    w = np.where(np.abs(dth) <= 2.0 * np.pi / N_SECTORS,
                 np.cos(N_SECTORS * dth / 4.0) ** 2, 0.0)
    return (w * g[:, :, None]).max(axis=1).astype(np.float32)       # [N, K]


def _batch_wall_channel(positions: np.ndarray) -> np.ndarray:
    """Vectorised wall code: [N, 2] -> [N, K]."""
    n = positions.shape[0]
    t = np.full((n, N_SECTORS), np.inf)
    for a in range(2):
        u = SECTOR_DIRS[:, a][None, :]                              # [1, K]
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (np.sign(u) * 1.0 - positions[:, a][:, None]) / u  # [N, K]
        ta = np.where(np.abs(u) > 1e-9, ta, np.inf)
        t = np.minimum(t, ta)
    return np.exp(-np.maximum(t, 0.0) / WALL_LAMBDA).astype(np.float32)


def encode_observations(positions: np.ndarray, objects: dict[str, np.ndarray],
                        food_strengths: np.ndarray = None) -> np.ndarray:
    """Batch encoder: [N, 2] positions -> [N, obs_dim()] (vectorised)."""
    if SENSING == "vector":
        return np.concatenate([
            _batch_nearest_rel(positions, objects["food"], available=food_strengths),
            _batch_nearest_rel(positions, objects["threat"]),
            _batch_nearest_rel(positions, objects["shelter"]),
        ], axis=1).astype(np.float32)
    return np.concatenate([
        _batch_sector_channel(positions, objects["food"], available=food_strengths),
        _batch_sector_channel(positions, objects["threat"]),
        _batch_sector_channel(positions, objects["shelter"]),
        _batch_wall_channel(positions),
    ], axis=1).astype(np.float32)


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


def make_behavior_targets(positions: np.ndarray, objects: dict[str, np.ndarray],
                          alpha: np.ndarray, gamma: float = 2.0,
                          wall_margin: float = 0.15, wall_kappa: float = 1.0,
                          food_strengths: np.ndarray = None,
                          eps: float = 1e-8) -> np.ndarray:
    """Behaviour targets from GEOMETRY (positions + objects), sensing-agnostic.

    Same rule as `make_mixed_behavior_targets` -- z = a_f r_food - a_t r_threat
    + a_s r_shelter, v = tanh(gamma ||z||) z/||z|| -- but computed from the
    agent positions directly, so it defines the teacher for any SENSING mode,
    plus the wall-consistency term that used to live in the driver: within
    `wall_margin` of a wall the outward component of v is smoothstep-blended
    to -wall_kappa times itself (soft reflection).  Pure damping (kappa=0)
    measured WORSE than no treatment (viscous band); keep kappa near 1.
    """
    r = {c: _batch_nearest_rel(
             positions, objects[c],
             available=food_strengths if c == "food" else None)
         for c in CATEGORIES}
    z = alpha[0] * r["food"] - alpha[1] * r["threat"] + alpha[2] * r["shelter"]
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    v = (np.tanh(gamma * norm) * z / (norm + eps)).astype(np.float32)
    if wall_margin > 0.0:
        for ax in (0, 1):
            p, u = positions[:, ax], v[:, ax]
            d = np.clip((1.0 - np.abs(p)) / wall_margin, 0.0, 1.0)
            s = d * d * (3.0 - 2.0 * d)                     # smoothstep
            m = s - (1.0 - s) * wall_kappa                  # 1 inside -> -kappa at wall
            v[:, ax] = np.where(u * p > 0.0, u * m, u)
    return v


def make_training_grid(grid_side: int) -> np.ndarray:
    """A regular grid of agent positions over [-1, 1]^2, [grid_side**2, 2]."""
    axis = np.linspace(-1.0, 1.0, grid_side, dtype=np.float32)
    gx, gy = np.meshgrid(axis, axis)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


# ============================================================
# Closed-loop drives and neuromodulatory arbitration
# ============================================================

def initialize_demo_state(objects: dict[str, np.ndarray]) -> dict:
    """Agent state plus the internal drives that gate the noise field."""
    start = SCRIPTED_START.copy()
    return {
        "pos": start.copy(),
        "start": start.copy(),
        "vel": np.zeros(2, dtype=np.float32),               # smoothed heading
        "heading": np.array([1.0, 0.0], dtype=np.float32),  # unit facing
        "speed": 0.0,                                       # last step length / dt
        "trail": [start.copy()],
        "objects": {k: objects[k].copy() for k in CATEGORIES},
        "food_strengths": np.ones(objects["food"].shape[0], dtype=np.float32),
        # Frames left before each eaten food regrows; 0 means available.
        "food_timer": np.zeros(objects["food"].shape[0], dtype=np.int32),
        "hunger": 1.0,
        "goal": "food",
        "clock": 0,
        "rest": 0,
        "w": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }


def apply_food_depletion(position: np.ndarray, objects: dict[str, np.ndarray],
                         food_strengths: np.ndarray, eat_radius: float,
                         respawn: bool = False, timers: np.ndarray = None,
                         regrow_frames: int = 0) -> int:
    """Deplete any food the agent reaches (in place); index eaten, else -1.

    `regrow_frames` is a REGROWTH DELAY, and it is what makes the animal forage
    over the whole scene instead of one patch.  The original rule respawned a
    food the moment the agent was 2.5 eat-radii away -- about 0.25 -- so a single
    food could be harvested over and over by stepping back and forth across its
    own regrowth boundary, and the diet collapsed onto whichever food was
    nearest.  With a delay, an eaten patch stays empty long enough that the
    cheapest next meal is a DIFFERENT one.

    The distance term is kept as an additional guard: a food never pops back
    while the animal is standing on it, however long the timer.
    """
    eaten = -1
    food = objects["food"]
    for j in range(food.shape[0]):
        d = float(np.linalg.norm(food[j] - position))
        if d < eat_radius and food_strengths[j] > 0.0:
            food_strengths[j] = 0.0
            if timers is not None:
                timers[j] = int(regrow_frames)
            eaten = j
        elif respawn and food_strengths[j] <= 0.0:
            if timers is not None and timers[j] > 0:
                timers[j] -= 1
            elif d > 2.5 * eat_radius:
                food_strengths[j] = 1.0
    return eaten


def apply_shelter_satisfaction(position: np.ndarray, objects: dict[str, np.ndarray],
                               shelter_radius: float) -> bool:
    """True if the agent is inside any shelter's arrival radius."""
    shelter = objects["shelter"]
    if shelter.shape[0] == 0:
        return False
    return float(np.linalg.norm(shelter - position[None, :], axis=1).min()) < shelter_radius


def circadian_phase(clock: int, period: int = None) -> float:
    """Position in the day/night cycle as a fraction in [0, 1); 0 is dawn."""
    period = int(period or CIRCADIAN_PERIOD)
    return (int(clock) % period) / period


def circadian_goal(clock: int, period: int = None,
                   day_fraction: float = None) -> str:
    """'food' during the day, 'shelter' during the night -- clock only.

    Nothing about the animal's internal state enters here: that is the point of
    a circadian rhythm as opposed to a homeostatic one.  A starving animal still
    goes home at dusk, and a full one still leaves at dawn.
    """
    day = DAY_FRACTION if day_fraction is None else float(day_fraction)
    return "food" if circadian_phase(clock, period) < day else "shelter"


def neuromod_weights(position: np.ndarray, objects: dict[str, np.ndarray],
                     hunger: float, goal: str,
                     prev_w: np.ndarray, threat_gain: float, threat_range: float,
                     smoothing: float, refuge_range: float = 0.0,
                     risk_hunger: float = 0.0, panic_range: float = 0.0) -> tuple:
    """Continuous neuromodulatory weights [food, threat, shelter].

    Two levels, deliberately simple: a committed food/shelter GOAL (1 bit, set by
    the circadian clock BEFORE this function runs -- see `circadian_goal`), and a
    distance-graded threat urgency that continuously cross-fades the threat field
    over that goal.  The whole vector is low-pass filtered, which prevents the
    chattering a hard rule produces and blends the recruited subnetworks like
    overlapping concentrations.

    Hunger no longer decides where to go -- the clock does.  Its one remaining
    role is the courage gate below (`risk_hunger`), which is what makes it a
    clean physiological variable instead of a second goal system.

    `refuge_range > 0` adds a refuge override: while the goal is the shelter,
    threat urgency is damped by proximity to the nearest shelter, so a
    shelter-bound agent dashes the last stretch in instead of hovering at the
    doorstep whenever a wandering threat loiters near the entrance (the flee
    blend otherwise cancels the weak arrival-side approach drive for hundreds
    of frames).  Entering the refuge IS the anti-threat response there.  0
    disables it and preserves the original arbitration.

    `risk_hunger > 0` adds the starvation-predation risk trade-off: while
    foraging, threat urgency is damped by up to `risk_hunger`, but only once
    hunger exceeds RISK_HUNGER_ONSET (smoothstep to full courage at hunger 1),
    so a fed agent's avoidance is untouched and a starving one darts past the
    threat.  Without this, a threat loitering near the foods (the wander box
    contains both) blocks foraging indefinitely -- measured at ~80% of all
    food-less foraging time; a LINEAR hunger gate measured as overshoot (the
    agent brushes threats even when fed, min distance 0.02).  Because hunger
    keeps rising while blocked, the block is self-limiting.  0 disables it.
    """
    base = np.array([1.0, 0.0, 0.0] if goal == "food" else [0.0, 0.0, 1.0],
                    dtype=np.float32)

    threat = objects["threat"]
    if threat.shape[0] > 0:
        d_threat = float(np.linalg.norm(threat - position[None, :], axis=1).min())
        g_threat = float(np.exp(-(d_threat / threat_range) ** 2))   # 1 near, 0 far
    else:
        g_threat = 0.0
    a = float(np.clip(threat_gain * g_threat, 0.0, 1.0))

    # Both overrides below DAMP avoidance, and either can damp it to nothing at
    # point-blank range -- that is how the agent ends up touching a threat.
    # `panic` is the veto: within `panic_range` the damping is faded out, so
    # courage only ever changes whether the agent APPROACHES a distant threat,
    # never whether it recoils from an adjacent one.  Trading the two apart this
    # way is what removes contact without paying for it in foraging.
    panic = (float(np.exp(-(d_threat / panic_range) ** 2))
             if panic_range > 0.0 and threat.shape[0] > 0 else 0.0)
    keep = 1.0 - panic

    if refuge_range > 0.0 and goal == "shelter" and objects["shelter"].shape[0] > 0:
        d_shelter = float(np.linalg.norm(objects["shelter"] - position[None, :],
                                         axis=1).min())
        a *= 1.0 - keep * float(np.exp(-(d_shelter / refuge_range) ** 2))
    if risk_hunger > 0.0 and goal == "food":
        h = float(np.clip((hunger - RISK_HUNGER_ONSET)
                          / (1.0 - RISK_HUNGER_ONSET), 0.0, 1.0))
        a *= 1.0 - keep * risk_hunger * h * h * (3.0 - 2.0 * h)

    w_target = (1.0 - a) * base + a * np.array([0.0, 1.0, 0.0], dtype=np.float32)
    w = ((1.0 - smoothing) * prev_w + smoothing * w_target).astype(np.float32)
    return w, g_threat


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
               smoothing: float = 0.2,
               resting: bool = False, speed_ref: float = None,
               wall_restitution: float = 0.5,
               turn_rate: float = 0.65) -> float:
    """Advance the agent one frame along the network's predicted velocity.

    The network's output MAGNITUDE is always used: the agent decelerates on
    approach AND freezes when the network falls silent.  This is required for
    behaviour-level stochastic resonance -- normalising the magnitude away
    would hide the low-noise collapse, which is the left arm of the inverted U.

    Integration smooths the SPEED and the HEADING separately.  Low-pass
    filtering the velocity VECTOR cancels opposing commands: when the
    arbitration cross-fades and v_pred reverses, (1-s)*v + s*v_pred collapses
    to ~0 and the agent freezes in open space with a perfectly healthy network
    behind it (measured: |v_pred| 0.12 while |vel| 0.03, and a third of those
    frames were outright reversals) -- the "stops for no reason" artefact.  A
    scalar speed cannot cancel, and a rate-limited heading turns the animal
    around instead of stopping it.  The magnitude channel still carries the
    collapse, so |v_pred| -> 0 still means frozen.

    `speed_ref` is this animal's NORMAL output magnitude (a trained net's
    typical |v| is well below the 1.0 that `speed_gain` was scaled for); the
    step is `speed_gain * min(1, |v| / speed_ref)`, which keeps the collapse
    while pacing the animal correctly.  None disables the rescale.

    `wall_restitution` reflects the outward velocity component on boundary
    contact so the agent slides off instead of pressing in.  With the standard
    speeds the position clip never actually fires (0 frames in 12000); kept as
    a guard for faster configurations.

    Returns the smoothed |v| itself, unscaled: that is the channel to plot,
    because it is what actually collapses.
    """
    if resting:
        state["vel"] = np.zeros(2, dtype=np.float32)
        state["speed"] = 0.0
        return 0.0

    cmd = float(np.linalg.norm(v_pred))
    speed = (1.0 - smoothing) * float(state["speed"]) + smoothing * cmd
    head = state.get("heading")
    if head is None or not np.any(head):
        head = np.array([1.0, 0.0], dtype=np.float32)
    if cmd > 1e-6:
        want = (v_pred / cmd).astype(np.float32)
        cur = float(np.arctan2(head[1], head[0]))
        tgt = float(np.arctan2(want[1], want[0]))
        d = (tgt - cur + np.pi) % (2.0 * np.pi) - np.pi
        d = float(np.clip(d, -turn_rate, turn_rate))
        head = np.array([np.cos(cur + d), np.sin(cur + d)], dtype=np.float32)
    state["heading"] = head
    state["vel"] = (speed * head).astype(np.float32)
    v = state["vel"]
    state["speed"] = speed

    if speed <= 1e-3:
        step = np.zeros(2, dtype=np.float32)
    elif speed_ref:
        scale = min(1.0, speed / float(speed_ref))
        step = dt * speed_gain * scale * v / speed
    else:
        step = dt * speed_gain * v

    raw = state["pos"] + step
    state["pos"] = np.clip(raw, -1.0, 1.0).astype(np.float32)
    # Reflect the heading on contact so the agent does not press into the wall.
    hit = (raw > 1.0) | (raw < -1.0)
    if hit.any():
        state["vel"] = state["vel"].copy()
        state["vel"][hit] *= -float(wall_restitution)
    return speed


@dataclasses.dataclass
class LoopParams:
    """Every knob of the closed loop, in one place.

    The animation and the headless rollout must run the SAME dynamics, otherwise the
    behavioural SR curve would not describe the animal you are watching.  Both go
    through `advance_frame` with one of these.
    """
    eat_radius: float = 0.10
    shelter_radius: float = 0.12  # arrival radius, just inside the DRAWN circle
                                  # (0.125); at 0.08 the agent sat visually inside
                                  # the shelter for tens of frames uncounted
    food_respawn: bool = True
    speed_gain: float = 0.9
    speed_ref: float = None      # this animal's normal |v|; see step_agent
    wall_restitution: float = 0.5  # outward-velocity reflection on wall contact
    turn_rate: float = 0.65      # max heading change per frame, radians;
                                 # 0.65 measured wall dwell and stalls both
                                 # to exactly zero (0.20 leaves 1.5% wall)
    velocity_smoothing: float = 0.2
    dt: float = 0.04
    hunger_rate: float = 0.006
    eat_amount: float = 0.6
    food_regrow_frames: int = 260  # delay before an eaten food returns.  0 is the
                                   # old distance-only rule, under which the diet
                                   # collapsed onto the nearest patch (one of the
                                   # three foods was taken 1 time against 200 and
                                   # 183) and one den took 105 nights against 5.
                                   # 260 measures diet evenness 0.86 and den
                                   # evenness 0.94 with the night-home rate
                                   # unchanged, at ~12% less foraging.
    rest_frames: int = 50
    threat_gain: float = 1.7
    # threat_range / neuromod_smoothing raised from 0.40 / 0.12: at the old
    # values the urgency ramp plus the field cross-fade lag (~8 frames) let the
    # agent graze right past a threat before the flee field took over.
    threat_range: float = 0.50
    neuromod_smoothing: float = 0.20
    refuge_range: float = 0.30   # damp threat urgency near the shelter while
                                 # shelter-bound (see neuromod_weights)
    risk_hunger: float = 0.9     # hungrier agents tolerate more threat while
                                 # foraging (see neuromod_weights); vetoed at
                                 # point-blank range by panic_range
    circadian_period: int = CIRCADIAN_PERIOD
    day_fraction: float = DAY_FRACTION
    panic_range: float = 0.22    # >0: inside this radius the courage/refuge
                                 # damping is vetoed, so avoidance is never
                                 # switched off at point-blank range
    threat_motion: str = "moving"
    threat_speed: float = 0.01
    threat_seed: int = 0


def advance_frame(state: dict, predict, fields, params: LoopParams,
                  concentration: float = 1.0, demo_mode: str = "scripted",
                  frame: int = 0, n_frames: int = 360) -> dict:
    """One closed-loop frame: drives -> field -> network -> movement -> bookkeeping.

    `predict(obs[1, 6], field) -> [1, 2]` hides the model.  `concentration` scales
    the whole blended field, which is the neuromodulator-concentration axis: at low
    concentration the crossing falls below threshold and the network goes silent, at
    high concentration it saturates.  Because the network's output magnitude is
    used as the speed, that shows up in the agent as freezing and as aimless
    drift respectively.

    Returns a per-frame record; the caller decides whether to draw it or count it.
    """
    from . import fields as fields_mod

    objs = state["objects"]
    if demo_mode == "scripted":
        if params.threat_motion == "moving" and "threat_vels" in state:
            step_threats(objs, state["threat_vels"],
                         shelter_keepout=THREAT_KEEPOUT_RADIUS, bounds=THREAT_BOUNDS)
        state["clock"] = int(state.get("clock", 0)) + 1
        state["goal"] = circadian_goal(state["clock"], params.circadian_period,
                                       params.day_fraction)
        advance_drives(state, params.shelter_radius, params.hunger_rate,
                       params.rest_frames)
        state["w"], _ = neuromod_weights(
            state["pos"], objs, state["hunger"],
            state["goal"], state["w"], params.threat_gain, params.threat_range,
            params.neuromod_smoothing, refuge_range=params.refuge_range,
            risk_hunger=params.risk_hunger, panic_range=params.panic_range)
        weights = state["w"]
    else:
        weights = cyclic_weights(2.0 * np.pi * frame / max(1, n_frames))

    field = fields_mod.blend_fields(fields, weights, CATEGORIES) * float(concentration)

    obs = encode_observation(state["pos"], objs, food_strengths=state["food_strengths"])
    v_pred = np.asarray(predict(obs[None, :], field)).ravel()

    resting = (demo_mode == "scripted" and state["rest"] > 0)
    speed = step_agent(state, v_pred, params.dt, params.speed_gain,
                       smoothing=params.velocity_smoothing,
                       resting=resting, speed_ref=params.speed_ref,
                       wall_restitution=params.wall_restitution,
                       turn_rate=params.turn_rate)

    # NOTE: index, not a flag -- food 0 is a valid result, so test against -1.
    eaten = apply_food_depletion(
        state["pos"], objs, state["food_strengths"], params.eat_radius,
        respawn=(params.food_respawn or demo_mode == "scripted"),
        timers=state.get("food_timer"), regrow_frames=params.food_regrow_frames)
    ate = eaten >= 0
    if ate:
        state["hunger"] = max(0.0, state["hunger"] - params.eat_amount)

    state["trail"].append(state["pos"].copy())
    state["trail"] = state["trail"][-160:]

    threat = objs["threat"]
    d_threat = (float(np.linalg.norm(threat - state["pos"][None, :], axis=1).min())
                if threat.shape[0] else float("inf"))
    dom = int(np.argmax(weights))
    inside = apply_shelter_satisfaction(state["pos"], objs, params.shelter_radius)

    # A stall is a stop with no behavioural excuse: not the scripted rest, not
    # sheltering, and not the deceleration zone of the food it is arriving at.
    d_food = float("inf")
    avail = objs["food"][state["food_strengths"] > 0.0]
    if avail.shape[0]:
        d_food = float(np.linalg.norm(avail - state["pos"][None, :], axis=1).min())
    stalled = (speed < STALL_SPEED and not resting and not inside
               and d_food > params.eat_radius)

    return {
        "weights": weights, "field": field, "speed": speed, "ate": ate,
        "ate_index": eaten,        # which food, for diet-diversity scoring
        "resting": resting, "d_threat": d_threat,
        "at_wall": bool((np.abs(state["pos"]) > 1.0 - WALL_BAND).any()),
        "stalled": bool(stalled),
        "inside_shelter": inside,
        "state_name": STATES[dom], "field_name": STATE_TO_FIELD[STATES[dom]],
        "label": EPISODE_LABEL[STATES[dom]],
    }


def evenness(counts) -> float:
    """Normalised Shannon entropy of a count vector: 1 = all options used
    equally, 0 = a single option (or fewer than two options)."""
    c = np.asarray(counts, dtype=float)
    total = c.sum()
    if total <= 0 or c.size < 2:
        return 0.0
    p = c[c > 0] / total
    return float(-(p * np.log(p)).sum() / np.log(c.size))


def rollout(predict, fields, params: LoopParams, n_frames: int = 1200,
            concentration: float = 1.0, demo_mode: str = "scripted",
            objects: dict = None, seed: int = 0) -> dict:
    """Run one closed-loop episode headlessly and return behavioural measures.

    These are the quantities a behaviour-level stochastic-resonance claim is about:
    what the animal MANAGES TO DO, not how well a vector field is regressed.
    """
    objs = objects if objects is not None else make_scripted_objects()
    state = initialize_demo_state(objs)
    if params.threat_motion == "moving":
        state["threat_vels"] = make_threat_velocities(state["objects"],
                                                      params.threat_speed, seed)
    foods = 0
    speeds, sheltered, close_calls = [], 0, 0
    d_threat_min = float("inf")
    contacts, wall_frames, stalls = 0, 0, 0
    run, stall_run, wrun, wall_run = 0, 0, 0, 0
    start = state["pos"].copy()
    path = 0.0
    # Circadian bookkeeping: a "night" begins at each food->shelter goal flip;
    # it counts as HOMED on the first arrival inside a den that night.
    nights, nights_home, got_home = 0, 0, False
    prev_goal = state["goal"]
    food_hits = np.zeros(objs["food"].shape[0], dtype=int)
    den_hits = np.zeros(objs["shelter"].shape[0], dtype=int)

    for k in range(n_frames):
        prev = state["pos"].copy()
        rec = advance_frame(state, predict, fields, params,
                            concentration=concentration, demo_mode=demo_mode,
                            frame=k, n_frames=n_frames)
        foods += int(rec["ate"])
        if rec["ate_index"] >= 0:
            food_hits[rec["ate_index"]] += 1
        goal = state["goal"]
        if goal == "shelter" and prev_goal == "food":
            nights += 1
            got_home = False
        if goal == "shelter" and rec["inside_shelter"] and not got_home:
            got_home = True
            nights_home += 1
            den_hits[int(np.argmin(np.linalg.norm(
                objs["shelter"] - state["pos"][None, :], axis=1)))] += 1
        prev_goal = goal
        speeds.append(rec["speed"])
        sheltered += int(rec["inside_shelter"])
        d_threat_min = min(d_threat_min, rec["d_threat"])
        close_calls += int(rec["d_threat"] < 0.2)
        contacts += int(rec["d_threat"] < THREAT_CONTACT_RADIUS)
        wall_frames += int(rec["at_wall"])
        stalls += int(rec["stalled"])
        # Run lengths matter more than the fraction: a viewer forgives a single
        # frame at a zero crossing but not a multi-second freeze.
        run = run + 1 if rec["stalled"] else 0
        stall_run = max(stall_run, run)
        wrun = wrun + 1 if rec["at_wall"] else 0
        wall_run = max(wall_run, wrun)
        path += float(np.linalg.norm(state["pos"] - prev))

    return {
        "foods": foods,
        "foods_per_1k": 1000.0 * foods / n_frames,
        "mean_speed": float(np.mean(speeds)),
        "shelter_frac": sheltered / n_frames,
        "close_frac": close_calls / n_frames,
        "d_threat_min": d_threat_min,
        # The three behavioural artefacts, as fractions of the episode:
        "contact_frac": contacts / n_frames,   # touching a threat
        "wall_frac": wall_frames / n_frames,   # pinned to the arena boundary
        "stall_frac": stalls / n_frames,       # stopped for no legible reason
        "stall_run": stall_run,                # longest unbroken freeze (frames)
        "wall_run": wall_run,                  # longest unbroken wall contact
        # Circadian competence and diversity (see docs/idea_neuromod.md C.9):
        "nights": nights,
        "night_home_rate": nights_home / nights if nights else 0.0,
        "diet_evenness": evenness(food_hits),
        "den_evenness": evenness(den_hits),
        "food_hits": food_hits.tolist(),
        "den_hits": den_hits.tolist(),
        "path_len": path,
        "net_displacement": float(np.linalg.norm(state["pos"] - start)),
    }


def advance_drives(state: dict, shelter_radius: float, hunger_rate: float,
                   rest_frames: int) -> None:
    """One frame of hunger / rest bookkeeping (in place).

    Sheltering uses a time-based REST: on arrival at night a countdown starts and
    the animal sits still while it runs; the clock keeps re-arming it, so in
    practice the animal sleeps until dawn.  A time-based rest is robust to a
    wandering threat briefly nudging the agent off the shelter, unlike a
    "decay only while inside" rule which such a threat can stall forever.

    Hunger is the only other internal variable, and its only consumer is the
    courage gate in `neuromod_weights` -- the circadian clock owns the goal.
    """
    inside = apply_shelter_satisfaction(state["pos"], state["objects"], shelter_radius)
    state["hunger"] = min(1.0, state["hunger"] + hunger_rate)

    if state["goal"] == "shelter":
        if inside and state["rest"] == 0:
            state["rest"] = rest_frames
        if state["rest"] > 0:
            state["rest"] -= 1
    else:
        state["rest"] = 0
