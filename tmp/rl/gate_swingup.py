"""tmp/rl.gate_swingup -- 方向3b: LEARNED noise-field gate (modulatory core, §9/§21.3/§19).

The fixed cos(theta) gate of 3a is replaced by a small NNN "modulatory core" that maps the
state to the pump<->balance gate g; the field is blend((1-g)P_pump, g P_balance).  The gate
is a latent action (§19): sampled per step, and the core is trained by the SAME GAE
advantage via cov_jac (no transposed weights) -- so reward must organize the pump/balance
modes autonomously.  Actor (force body + gate core) is fully NNN; only the critic is
external backprop.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import constants  # noqa: F401
from .policy_cont import ContinuousNNNPolicy
from .critic import ValueMLP
from . import credit as C
from .envs_swingup import CartPoleSwingUp
from .train import ManualOpt, RunningNorm
from .multimode import _p
from .a2c_swingup import _norm


def _blend(fields, g):
    return [(1.0 - g) * fields[0][l] + g * fields[1][l] for l in range(len(fields[0]))]


def train_gate(seed=0, H=128, Hg=48, sigma=0.6, updates=400, episodes_per_update=3,
               horizon=400, gamma=0.99, lam=0.95, lr_actor=0.01, lr_gate=0.01,
               lr_critic=1e-3, critic_epochs=8, bottom_frac=0.5, force_mag=20.0,
               x_threshold=4.0, sigma_explore=0.4, sigma_explore_end=0.1,
               gate_force=5.0, gate_sigma=0.4, gate_sigma_end=0.15, fields=None,
               checkpoint_every=25, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = CartPoleSwingUp(horizon=horizon, seed=seed, force_mag=force_mag,
                          x_threshold=x_threshold, continuous=True)
    body = ContinuousNNNPolicy(obs_dim=5, hidden=H, std=sigma, t=64, force_max=force_mag,
                               sigma_explore=sigma_explore)
    gate = ContinuousNNNPolicy(obs_dim=5, hidden=Hg, std=sigma, t=64, force_max=gate_force,
                               sigma_explore=gate_sigma)
    critic = ValueMLP(5)
    bkeys, gkeys = C.param_keys(body), C.param_keys(gate)
    body_opt, gate_opt = ManualOpt("adam"), ManualOpt("adam")
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    norm = RunningNorm(5)

    hist, checkpoints = [], []
    for upd in range(updates):
        frac = upd / max(1, updates - 1)
        body.sigma_explore = sigma_explore + (sigma_explore_end - sigma_explore) * frac
        gate.sigma_explore = gate_sigma + (gate_sigma_end - gate_sigma) * frac
        b_psi, b_gpsi, b_adv, b_ret, b_obs = [], [], [], [], []
        ep_returns = []
        for _ in range(episodes_per_update):
            start = (math.pi + float(rng.uniform(-0.5, 0.5)) if rng.random() < bottom_frac
                     else float(rng.uniform(-math.pi, math.pi)))
            obs, _ = env.reset(seed=None, start_theta=start)
            on = _norm(norm, obs, True)
            psis, gpsis, rews, vals, obses = [], [], [], [], []
            for _ in range(horizon):
                gstep = gate.rollout_step(on.unsqueeze(0))       # gate latent action
                g = 1.0 / (1.0 + math.exp(-float(gstep.action.item())))
                body.field = _blend(fields, g)
                bstep = body.rollout_step(on.unsqueeze(0))
                psis.append(C.cov_jac_grad(body, bstep))
                gpsis.append(C.cov_jac_grad(gate, gstep))
                with torch.no_grad():
                    vals.append(float(critic(on.unsqueeze(0))))
                obses.append(on)
                obs, r, te, tr, _ = env.step(float(bstep.action.item()))
                rews.append(r)
                on = _norm(norm, obs, True)
                if te or tr:
                    break
            with torch.no_grad():
                v_last = 0.0 if te else float(critic(on.unsqueeze(0)))
            n = len(rews)
            adv = np.zeros(n)
            gae = 0.0
            for t in range(n - 1, -1, -1):
                v_next = vals[t + 1] if t + 1 < n else v_last
                gae = (rews[t] + gamma * v_next - vals[t]) + gamma * lam * gae
                adv[t] = gae
            b_psi.extend(psis); b_gpsi.extend(gpsis)
            b_adv.extend(adv.tolist()); b_ret.extend((adv + np.array(vals)).tolist())
            b_obs.extend(obses)
            ep_returns.append(float(sum(rews)))
        hist.append(np.mean(ep_returns))

        A = np.array(b_adv)
        A = (A - A.mean()) / (A.std() + 1e-6)
        N = len(b_psi)
        gb = {k: torch.zeros_like(_p(body, k)) for k in bkeys}
        gg = {k: torch.zeros_like(_p(gate, k)) for k in gkeys}
        for t in range(N):
            at = float(A[t])
            for k in bkeys:
                gb[k] += at * b_psi[t][k]
            for k in gkeys:
                gg[k] += at * b_gpsi[t][k]
        for k in bkeys:
            body_opt.update(str(k), _p(body, k), -gb[k] / N, lr_actor)
        for k in gkeys:
            gate_opt.update(str(k), _p(gate, k), -gg[k] / N, lr_gate)

        obs_b = torch.stack(b_obs)
        ret_b = torch.tensor(b_ret, dtype=torch.float32)
        for _ in range(critic_epochs):
            critic_opt.zero_grad()
            (((critic(obs_b) - ret_b) ** 2).mean()).backward()
            critic_opt.step()

        if checkpoint_every and (upd + 1) % checkpoint_every == 0:
            checkpoints.append((upd + 1, _snap(body, gate, norm, H, Hg, force_mag,
                                               gate_force, fields)))
        if verbose and (upd + 1) % 10 == 0:
            print(f"  [gate seed{seed}] upd {upd+1:4d}  ep_return/step "
                  f"{np.mean(hist[-10:]) / horizon:+.3f}")
    return body, gate, critic, norm, checkpoints, hist


def _snap(body, gate, norm, H, Hg, force_mag, gate_force, fields):
    std = torch.sqrt(norm.M2 / norm.count + norm.eps)
    return {"body": {k: v.detach().clone() for k, v in body.net.state_dict().items()},
            "gate": {k: v.detach().clone() for k, v in gate.net.state_dict().items()},
            "norm_mean": norm.mean.detach().clone(), "norm_std": std.detach().clone(),
            "H": H, "Hg": Hg, "t": body.t, "force_mag": force_mag,
            "gate_force": gate_force, "fields": fields}


def build_gate_policy(state):
    body = ContinuousNNNPolicy(obs_dim=5, hidden=state["H"], t=state["t"],
                               force_max=state["force_mag"], sigma_explore=0.1)
    gate = ContinuousNNNPolicy(obs_dim=5, hidden=state["Hg"], t=state["t"],
                               force_max=state["gate_force"], sigma_explore=0.1)
    body.net.load_state_dict(state["body"]); body.eval()
    gate.net.load_state_dict(state["gate"]); gate.eval()
    return body, gate, state["norm_mean"], state["norm_std"], state["fields"]


def eval_gate(state, horizon=500, seeds=(0, 1, 2), return_gate=False):
    body, gate, mean, std, fields = build_gate_policy(state)
    fm = state["force_mag"]
    out, g_of_cos = [], []
    for s in seeds:
        env = CartPoleSwingUp(horizon=horizon, random_start=False, seed=s,
                              force_mag=fm, x_threshold=4.0, continuous=True)
        obs, _ = env.reset(seed=s)
        cs = []
        for _ in range(horizon):
            on = torch.clamp((torch.tensor(obs, dtype=torch.float32) - mean) / std, -5, 5)
            g = 1.0 / (1.0 + math.exp(-float(gate.rollout_step(on.unsqueeze(0), greedy=True).action.item())))
            body.field = _blend(fields, g)
            step = body.rollout_step(on.unsqueeze(0), greedy=True)
            if return_gate:
                g_of_cos.append((float(obs[2]), g))
            obs, r, te, tr, _ = env.step(float(step.action.item()))
            cs.append(env.cos_theta())
        cs = np.array(cs)
        out.append((cs.mean(), (cs > 0.9).mean(), (cs[-100:] > 0.9).mean()))
    res = (float(np.mean([o[0] for o in out])), float(np.mean([o[1] for o in out])),
           float(np.mean([o[2] for o in out])))
    return (res, g_of_cos) if return_gate else res
