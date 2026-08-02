"""tmp/rl.envs_swingup -- CartPole SWING-UP (gym does not ship one).

Standard cart-pole dynamics (same constants as gymnasium CartPole), but the pole starts
HANGING DOWN (theta = pi) and there is no angle termination: the agent must pump energy to
swing the pole up and then balance it near upright.  theta is measured from upright
(0 = up, pi = down).  Observation uses cos/sin(theta) to avoid the angle wrap discontinuity:

    obs = [x/x_thr, x_dot, cos(theta), sin(theta), theta_dot]

Reward = cos(theta) (up = +1, down = -1) minus a small cart-position penalty; the episode
ends (with a penalty) only if the cart runs off the track.  gym-compatible interface so the
existing trainer can drive it via `env_fn`.
"""
from __future__ import annotations

import math

import numpy as np


class CartPoleSwingUp:
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    length = 0.5                     # half pole length
    tau = 0.02

    def __init__(self, horizon=500, seed=0, random_start=False,
                 force_mag=20.0, x_threshold=4.0, continuous=False, energy_reward=True,
                 wall_mode="stop", wall_penalty=None, x_barrier=0.0, alive_bonus=0.0,
                 top_center=0.0):
        self.horizon = horizon
        self.random_start = random_start   # curriculum: start from any angle while training
        self.force_mag = force_mag
        self.x_threshold = x_threshold
        self.continuous = continuous       # if True, step() takes a float force in [-F, F]
        self.energy_reward = energy_reward  # True=swing-up shaping; False=pure cos (balance)
        # wall_mode "stop": the cart hits a physical stopper (clip x, zero x_dot) and the
        # episode continues -- the historical setting, which lets the policy EXPLOIT the
        # stopper (wall-assisted pumping).  wall_mode "end": touching either stopper
        # TERMINATES the episode with -wall_penalty, so a successful swing-up must never
        # contact the bounds.  x_barrier > 0 adds a soft quadratic penalty that turns on
        # beyond 70% of the track, giving a gradient away from the stoppers BEFORE the
        # catastrophic contact.
        self.wall_mode = wall_mode
        # default wall penalty: 0.6/step in "stop" mode (the historical anti-pinning
        # value -- weak enough that v4 still balanced while leaning on a stopper),
        # 3.0 terminal in "end" mode.  Pass wall_penalty explicitly to override (the
        # wall-free-policy runs use "stop" with a heavy per-step cost so that the
        # OPTIMAL policy never touches a stopper even though the physics allow it).
        self.wall_penalty = (wall_penalty if wall_penalty is not None
                             else (0.6 if wall_mode == "stop" else 3.0))
        self.x_barrier = x_barrier
        # alive_bonus shifts the per-step reward so that SURVIVING beats dying.  With the
        # raw shaped reward (hanging ~ -2/step) a terminal wall is a reward ESCAPE: ending
        # the episode is worth more than living, and the policy learns suicide-by-stopper
        # (observed: ep_len pinned at ~34 for 50 updates).  A shift of ~+2.2 makes the
        # worst persistent state slightly positive, so termination forfeits future reward
        # and the wall becomes genuinely repulsive.  Greedy eval metrics (cos-based) are
        # unaffected by the shift.
        self.alive_bonus = alive_bonus
        # top_center > 0 adds a CENTERING penalty active only near upright
        # (top_center * (x/x_thr)^2 * max(0, cos theta)).  Diagnosis of the no-stopper
        # runs: the policy catches at x ~ +3 and HOLDS (up to 181 consecutive steps) but
        # drifts ~0.005/step into the barrier region and falls there -- the default
        # 0.05 x^2 term is too weak (gradient ~0.02/step at x=3) to make drift-arrest
        # worth learning.  Gating by cos keeps the pumping phase (which needs cart
        # excursions) unpenalized.
        self.top_center = top_center
        self.rng = np.random.default_rng(seed)
        self.total_mass = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.length
        self.state = None
        self.t = 0
        # gym-like handles used by the renderer / trainer
        self.observation_space_dim = 5

    def reset(self, seed=None, start_theta=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if start_theta is not None:
            theta, theta_dot = float(start_theta), float(self.rng.uniform(-1.0, 1.0))
        elif self.random_start:
            theta = float(self.rng.uniform(-math.pi, math.pi))   # any angle (curriculum)
            theta_dot = float(self.rng.uniform(-1.0, 1.0))
        else:
            theta = math.pi + float(self.rng.uniform(-0.1, 0.1))  # hanging down (eval)
            theta_dot = 0.0
        self.state = np.array([
            float(self.rng.uniform(-0.05, 0.05)), 0.0, theta, theta_dot], dtype=np.float64)
        self.t = 0
        return self._obs(), {}

    def _obs(self):
        x, x_dot, theta, theta_dot = self.state
        return np.array([x / self.x_threshold, x_dot,
                         math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        if self.continuous:
            force = float(np.clip(action, -self.force_mag, self.force_mag))
        else:
            force = self.force_mag if action == 1 else -self.force_mag
        costheta, sintheta = math.cos(theta), math.sin(theta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x += self.tau * x_dot
        x_dot += self.tau * xacc
        theta += self.tau * theta_dot
        theta_dot += self.tau * thetaacc

        # See __init__: "stop" = physical stopper (episode continues), "end" = terminal.
        at_wall = abs(x) > self.x_threshold
        if at_wall and self.wall_mode == "stop":
            x = math.copysign(self.x_threshold, x)
            x_dot = 0.0
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self.t += 1

        # Energy-shaped reward: reward BOTH height (cos) and having the pole energy near the
        # exact amount needed to reach the top (E_hat=1).  The energy term rewards pumping
        # while the pole is still low (build energy) and sheds energy near the top (catch),
        # so swing-up becomes the explicit objective -- a constant push no longer wins.
        wall_cost = self.wall_penalty if self.wall_mode == "stop" else 0.6
        if self.energy_reward:
            e_hat = 0.5 * (self.length / self.gravity) * theta_dot ** 2 + math.cos(theta)
            reward = (math.cos(theta)
                      - 0.5 * min(abs(e_hat - 1.0), 2.0)   # clip so wild spin doesn't dominate
                      - 0.05 * (x / self.x_threshold) ** 2
                      - (wall_cost if at_wall else 0.0))
        else:                                             # pure balance reward (Phase 1)
            reward = (math.cos(theta) - 0.05 * theta_dot ** 2
                      - 0.05 * (x / self.x_threshold) ** 2 - (wall_cost if at_wall else 0.0))
        if self.x_barrier > 0.0:
            over = max(0.0, abs(x) / self.x_threshold - 0.7) / 0.3
            reward -= self.x_barrier * over ** 2
        if self.top_center > 0.0:
            reward -= (self.top_center * (x / self.x_threshold) ** 2
                       * max(0.0, math.cos(theta)))
        reward += self.alive_bonus
        terminated = False
        if at_wall and self.wall_mode == "end":
            reward -= self.wall_penalty
            terminated = True
        truncated = self.t >= self.horizon
        return self._obs(), reward, terminated, truncated, {"wall": at_wall}

    def close(self):
        pass

    # ---- helpers for analysis / rendering ----
    @property
    def upright(self):
        return math.cos(self.state[2]) > 0.9

    def cos_theta(self):
        return math.cos(self.state[2])
