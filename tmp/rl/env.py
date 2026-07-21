"""tmp/rl.env -- CartPole helpers: state collection (Step A) and a stepping wrapper.

`collect_states` gathers a set of realistic observations by rolling out a random
policy, so the Step-A credit-quality measurement is done on states the agent will
actually visit rather than on arbitrary points.
"""
from __future__ import annotations

import numpy as np
import torch

from . import constants  # noqa: F401


def make_env(reward_delay: int = 0):
    import gymnasium as gym
    return gym.make("CartPole-v1")


def collect_states(n_states: int = 256, seed: int = 0) -> torch.Tensor:
    """Roll out a random policy and return up to `n_states` observations [n, 4]."""
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    obs_list = []
    obs, _ = env.reset(seed=seed)
    while len(obs_list) < n_states:
        obs_list.append(np.asarray(obs, dtype=np.float32))
        action = int(rng.integers(0, env.action_space.n))
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return torch.tensor(np.stack(obs_list[:n_states]), dtype=torch.float32)
