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
                 force_mag=20.0, x_threshold=4.0, continuous=False, energy_reward=True):
        self.horizon = horizon
        self.random_start = random_start   # curriculum: start from any angle while training
        self.force_mag = force_mag
        self.x_threshold = x_threshold
        self.continuous = continuous       # if True, step() takes a float force in [-F, F]
        self.energy_reward = energy_reward  # True=swing-up shaping; False=pure cos (balance)
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

        # The cart hits a WALL rather than ending the episode: clip position and stop it.
        # Full-horizon episodes give the agent room to discover the swing-up.
        at_wall = abs(x) > self.x_threshold
        if at_wall:
            x = math.copysign(self.x_threshold, x)
            x_dot = 0.0
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self.t += 1

        # Energy-shaped reward: reward BOTH height (cos) and having the pole energy near the
        # exact amount needed to reach the top (E_hat=1).  The energy term rewards pumping
        # while the pole is still low (build energy) and sheds energy near the top (catch),
        # so swing-up becomes the explicit objective -- a constant push no longer wins.
        if self.energy_reward:
            e_hat = 0.5 * (self.length / self.gravity) * theta_dot ** 2 + math.cos(theta)
            reward = (math.cos(theta)
                      - 0.5 * min(abs(e_hat - 1.0), 2.0)   # clip so wild spin doesn't dominate
                      - 0.05 * (x / self.x_threshold) ** 2
                      - (0.6 if at_wall else 0.0))         # kill the wall-pinning local optimum
        else:                                             # pure balance reward (Phase 1)
            reward = (math.cos(theta) - 0.05 * theta_dot ** 2
                      - 0.05 * (x / self.x_threshold) ** 2 - (0.6 if at_wall else 0.0))
        terminated = False
        truncated = self.t >= self.horizon
        return self._obs(), reward, terminated, truncated, {}

    def close(self):
        pass

    # ---- helpers for analysis / rendering ----
    @property
    def upright(self):
        return math.cos(self.state[2]) > 0.9

    def cos_theta(self):
        return math.cos(self.state[2])
