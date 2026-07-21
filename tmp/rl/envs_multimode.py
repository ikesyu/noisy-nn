"""tmp/rl.envs_multimode -- minimal multi-mode environment for the noise-field option test.

Two-target reach on a 1-D line.  Each episode has a regime r in {0, 1} whose target is at
targets[r] (default -1 / +1).  The reward is the negative distance to that target, so the
agent must drive x to targets[r] and stay.  Crucially the regime is NOT in the observation
(the observation is only the position x): the ONLY thing that tells the agent which target
to seek is its noise field.  So the field must act as the behavioral mode (§7.2, §14.2) --
fixing the field should commit the behavior; switching it should switch the target with the
SAME shared weights (the RL version of front_comp's L1 addressing result).
"""
from __future__ import annotations

import numpy as np


class MultiModeReach:
    def __init__(self, targets=(-1.0, 1.0), horizon=40, step_size=0.15, x_range=2.0,
                 seed=0):
        self.targets = targets
        self.horizon = horizon
        self.step_size = step_size
        self.x_range = x_range
        self.rng = np.random.default_rng(seed)

    @property
    def obs_dim(self):
        return 1

    def reset(self, regime: int):
        self.regime = regime
        self.target = self.targets[regime]
        self.x = float(self.rng.uniform(-0.3, 0.3))
        self.t = 0
        return self._obs()

    def _obs(self):
        # position only, scaled to ~[-1, 1]; the regime is deliberately hidden.
        return np.array([self.x / self.x_range], dtype=np.float32)

    def step(self, action: int):
        self.x += self.step_size * (1.0 if action == 1 else -1.0)
        self.x = float(np.clip(self.x, -self.x_range, self.x_range))
        self.t += 1
        reward = -abs(self.x - self.target)
        done = self.t >= self.horizon
        return self._obs(), reward, done
