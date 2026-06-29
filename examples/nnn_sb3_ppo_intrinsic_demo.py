"""
NNN sample action + Gaussian moment log_prob with Stable-Baselines3 PPO.

Core idea
---------
1. The policy generates T raw action samples from a sample-level NNN.
2. The actual action is selected from those NNN samples, not from an external Normal.
3. For PPO, log_prob(action | obs) is approximated by a Gaussian fitted to the
   NNN raw action samples by their empirical mean and variance.
4. The action is tanh-squashed and affine-scaled to the Gymnasium Box action space.

Install example
---------------
pip install "stable-baselines3[extra]" "gymnasium[classic-control]" matplotlib

Run
---
python nnn_sb3_ppo_intrinsic_demo.py --env Pendulum-v1 --timesteps 50000

Notes
-----
This is an experimental policy class. It intentionally avoids sampling actions
from torch.distributions.Normal. The Normal distribution is used only as a
moment-matched approximation for PPO log_prob and entropy.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Sequence, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.type_aliases import Schedule


# =============================================================================
# Crossing activation, sample-level version
# =============================================================================

class CrossingSample(th.autograd.Function):
    """Sample-level Crossing activation for tensors shaped [B, T, D].

    Forward output is ternary-valued in {0, 0.5, 1} because it averages the two
    crossing indicators around thresholds +h and -h.
    Backward uses the same finite-difference/KDE-style surrogate as in the
    provided implementation.
    """

    @staticmethod
    def forward(ctx, input: th.Tensor, h: float = 0.05) -> th.Tensor:
        h = abs(float(h))
        bin1 = (input > h).float()
        bin2 = (input > -h).float()

        xor1 = (bin1[:, 1:, :] - bin1[:, :-1, :]).abs()
        xor1_last = (bin1[:, -1, :] - bin1[:, 0, :]).abs()
        xor1 = th.cat([xor1, xor1_last.unsqueeze(1)], dim=1)

        xor2 = (bin2[:, 1:, :] - bin2[:, :-1, :]).abs()
        xor2_last = (bin2[:, -1, :] - bin2[:, 0, :]).abs()
        xor2 = th.cat([xor2, xor2_last.unsqueeze(1)], dim=1)

        ctx.save_for_backward(xor1, xor2)
        ctx.h = h
        return 0.5 * (xor1 + xor2)

    @staticmethod
    def backward(ctx, grad_output: th.Tensor) -> Tuple[th.Tensor, None]:
        xor1, xor2 = ctx.saved_tensors
        h = ctx.h
        T = xor1.size(1)
        coeff = (xor2 - xor1).mean(dim=1) / (2.0 * h)
        coeff = coeff.unsqueeze(1).repeat(1, T, 1)
        return coeff * grad_output, None


class GaussianCrossingSampleLayer(nn.Module):
    def __init__(self, std: float = 0.6, h: float = 0.15):
        super().__init__()
        self.std = float(std)
        self.h = float(h)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x_noisy = x + th.randn_like(x) * self.std
        return CrossingSample.apply(x_noisy, self.h)


class SampleLayer(nn.Module):
    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = int(num_samples)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x.unsqueeze(1).repeat(1, self.num_samples, 1)


class SimpleNNNBase(nn.Module):
    """Sample-level NNN hidden base without final action/readout layer.

    Input:  [B, obs_dim]
    Output: [B, T, hidden[-1]]
    """

    def __init__(
        self,
        input_dim: int,
        hidden: Sequence[int] = (64, 64),
        n_samples: int = 64,
        crossing_std: float = 0.6,
        crossing_h: float = 0.15,
    ):
        super().__init__()
        dims = [int(input_dim), *map(int, hidden)]
        if len(dims) < 2:
            raise ValueError("SimpleNNNBase requires at least one hidden layer.")

        self.fcs = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.crossings = nn.ModuleList([
            GaussianCrossingSampleLayer(std=crossing_std, h=crossing_h)
            for _ in self.fcs
        ])
        self.sample_layer = SampleLayer(n_samples)
        self.output_dim = dims[-1]

    def forward(self, x: th.Tensor) -> th.Tensor:
        for i, (fc, crossing) in enumerate(zip(self.fcs, self.crossings)):
            x = fc(x)
            if i == 0:
                x = self.sample_layer(x)
            x = crossing(x)
        return x


class NNNIntrinsicActorCore(nn.Module):
    """Actor core that produces raw action samples from NNN intrinsic noise.

    The actual raw action samples are
        raw_y_t = mu + sigma * eps_t
    where eps_t is obtained by a trainable residual projection of NNN samples
    and then standardized over the sample axis T.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden: Sequence[int] = (64, 64),
        n_samples: int = 64,
        crossing_std: float = 0.6,
        crossing_h: float = 0.15,
        log_sigma_bounds: Tuple[float, float] = (-4.0, 1.0),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.base = SimpleNNNBase(
            input_dim=feature_dim,
            hidden=hidden,
            n_samples=n_samples,
            crossing_std=crossing_std,
            crossing_h=crossing_h,
        )
        hdim = self.base.output_dim
        self.mean_head = nn.Linear(hdim, action_dim)
        self.log_sigma_head = nn.Linear(hdim, action_dim)
        self.residual_head = nn.Linear(hdim, action_dim, bias=False)
        self.log_sigma_min, self.log_sigma_max = log_sigma_bounds
        self.eps = float(eps)

    def forward(self, features: th.Tensor) -> th.Tensor:
        z_samples = self.base(features)          # [B, T, H]
        z_mean = z_samples.mean(dim=1)           # [B, H]

        mu = self.mean_head(z_mean)              # [B, A]
        log_sigma = self.log_sigma_head(z_mean)
        log_sigma = th.clamp(log_sigma, self.log_sigma_min, self.log_sigma_max)
        sigma = th.exp(log_sigma)                # [B, A]

        r = self.residual_head(z_samples)        # [B, T, A]
        r = r - r.mean(dim=1, keepdim=True)
        r_std = th.sqrt(r.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        eps_samples = r / r_std                  # [B, T, A], approximately zero mean/unit variance

        raw_samples = mu.unsqueeze(1) + sigma.unsqueeze(1) * eps_samples
        return raw_samples


class MLPValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int] = (64, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, int(h)), nn.Tanh()]
            prev = int(h)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


# =============================================================================
# Stable-Baselines3 PPO policy
# =============================================================================

class NNNMomentGaussianPolicy(ActorCriticPolicy):
    """SB3 ActorCriticPolicy with NNN sample actions and moment-Gaussian log_prob.

    Continuous Box action spaces only.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        nnn_hidden: Sequence[int] = (64, 64),
        value_hidden: Sequence[int] = (64, 64),
        n_samples: int = 64,
        crossing_std: float = 0.6,
        crossing_h: float = 0.15,
        log_sigma_bounds: Tuple[float, float] = (-4.0, 1.0),
        moment_eps: float = 1e-6,
        **kwargs,
    ):
        if not isinstance(action_space, spaces.Box):
            raise ValueError("NNNMomentGaussianPolicy currently supports only continuous Box action spaces.")

        self.nnn_hidden = tuple(nnn_hidden)
        self.value_hidden = tuple(value_hidden)
        self.n_samples = int(n_samples)
        self.crossing_std = float(crossing_std)
        self.crossing_h = float(crossing_h)
        self.log_sigma_bounds = log_sigma_bounds
        self.moment_eps = float(moment_eps)

        # We override _build(), forward(), evaluate_actions(), and _predict().
        # net_arch is unused but accepted by SB3's constructor.
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            ortho_init=False,
            **kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        action_dim = get_action_dim(self.action_space)

        self.actor = NNNIntrinsicActorCore(
            feature_dim=self.features_dim,
            action_dim=action_dim,
            hidden=self.nnn_hidden,
            n_samples=self.n_samples,
            crossing_std=self.crossing_std,
            crossing_h=self.crossing_h,
            log_sigma_bounds=self.log_sigma_bounds,
            eps=self.moment_eps,
        )
        self.value_net = MLPValueNet(self.features_dim, hidden=self.value_hidden)

        low = th.as_tensor(self.action_space.low, dtype=th.float32)
        high = th.as_tensor(self.action_space.high, dtype=th.float32)
        self.register_buffer("action_bias", 0.5 * (high + low))
        self.register_buffer("action_scale", 0.5 * (high - low))

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _squash_action(self, raw_action: th.Tensor) -> th.Tensor:
        return self.action_bias + self.action_scale * th.tanh(raw_action)

    def _inverse_squash_action(self, action: th.Tensor) -> th.Tensor:
        # Map action in [low, high] to u in (-1, 1), then atanh(u).
        u = (action - self.action_bias) / (self.action_scale + self.moment_eps)
        u = th.clamp(u, -1.0 + 1e-6, 1.0 - 1e-6)
        return 0.5 * (th.log1p(u) - th.log1p(-u))

    def _sample_raw_action(self, raw_samples: th.Tensor, deterministic: bool) -> th.Tensor:
        # raw_samples: [B, T, A]
        if deterministic:
            return raw_samples.mean(dim=1)
        batch_size, n_samples, _ = raw_samples.shape
        idx = th.randint(n_samples, size=(batch_size,), device=raw_samples.device)
        return raw_samples[th.arange(batch_size, device=raw_samples.device), idx, :]

    def _raw_moment_stats(self, raw_samples: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean = raw_samples.mean(dim=1)
        var = raw_samples.var(dim=1, unbiased=False).clamp_min(self.moment_eps)
        return mean, var

    def _squashed_moment_log_prob(self, action: th.Tensor, raw_samples: th.Tensor) -> th.Tensor:
        # Approximate p(raw_action | obs) by moment-matched diagonal Gaussian,
        # then apply tanh-affine change-of-variables correction.
        raw_action = self._inverse_squash_action(action)
        mean, var = self._raw_moment_stats(raw_samples)
        std = th.sqrt(var)
        dist = th.distributions.Normal(mean, std)
        log_prob_raw = dist.log_prob(raw_action).sum(dim=-1)

        tanh_raw = th.tanh(raw_action)
        log_abs_det = th.log(self.action_scale * (1.0 - tanh_raw.pow(2)) + self.moment_eps).sum(dim=-1)
        return log_prob_raw - log_abs_det

    def _raw_moment_entropy(self, raw_samples: th.Tensor) -> th.Tensor:
        # Approximate raw entropy. This is not the exact entropy after tanh squashing,
        # but it is sufficient as a simple PPO entropy bonus.
        _, var = self._raw_moment_stats(raw_samples)
        return 0.5 * th.sum(th.log(2.0 * th.pi * th.e * var), dim=-1)

    def _actor_raw_samples_and_value(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        raw_samples = self.actor(features)
        values = self.value_net(features)
        return raw_samples, values

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        raw_samples, values = self._actor_raw_samples_and_value(obs)
        raw_action = self._sample_raw_action(raw_samples, deterministic=deterministic)
        action = self._squash_action(raw_action)
        log_prob = self._squashed_moment_log_prob(action, raw_samples)
        return action, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        raw_samples, values = self._actor_raw_samples_and_value(obs)
        log_prob = self._squashed_moment_log_prob(actions, raw_samples)
        entropy = self._raw_moment_entropy(raw_samples)
        return values, log_prob, entropy

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        return self.value_net(features)


# =============================================================================
# Training utilities
# =============================================================================

class RewardCallback(BaseCallback):
    """Small callback that stores training episode rewards from Monitor info."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


def make_env(env_id: str, log_dir: str, seed: int):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        env.reset(seed=seed)
        return env
    return _init


def plot_training_curve(log_dir: str, reward_callback: RewardCallback, out_file: str) -> None:
    plt.figure(figsize=(8, 4))

    plotted = False
    try:
        x, y = ts2xy(load_results(log_dir), "timesteps")
        if len(x) > 0:
            plt.plot(x, y, label="episode reward")
            plotted = True
    except Exception:
        pass

    if not plotted and len(reward_callback.episode_rewards) > 0:
        y = np.asarray(reward_callback.episode_rewards)
        x = np.arange(len(y))
        plt.plot(x, y, label="episode reward")
        plt.xlabel("episode")
    else:
        plt.xlabel("timesteps")

    plt.ylabel("reward")
    plt.title("PPO with NNN sample action + Gaussian moment log_prob")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    print(f"saved plot: {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="./nnn_ppo_logs")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    vec_env = make_vec_env(
        make_env(args.env, args.log_dir, args.seed),
        n_envs=args.n_envs,
        seed=args.seed,
    )

    policy_kwargs = dict(
        nnn_hidden=(64, 64),
        value_hidden=(64, 64),
        n_samples=64,
        crossing_std=0.6,
        crossing_h=0.15,
        log_sigma_bounds=(-4.0, 1.0),
        moment_eps=1e-6,
    )

    model = PPO(
        NNNMomentGaussianPolicy,
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    callback = RewardCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback, log_interval=1)

    model_path = os.path.join(args.log_dir, "ppo_nnn_intrinsic_policy")
    model.save(model_path)
    print(f"saved model: {model_path}.zip")

    eval_env = gym.make(args.env)
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True,
        warn=False,
    )
    print(f"deterministic eval reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    plot_path = os.path.join(args.log_dir, "training_curve.png")
    plot_training_curve(args.log_dir, callback, plot_path)


if __name__ == "__main__":
    main()
