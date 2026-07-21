"""rl_sigma_credit -- validate the per-unit noise-field eligibility psi_sigma (§10).

Step-A for sigma: does the forward-only psi_sigma = delta * (-d/sigma) reproduce the
autograd d log pi / d sigma?  psi_sigma reuses the SAME forward credit delta = g*phi' as the
weight eligibility; only the local coordinate differs (z_prev -> -d/sigma, §10.2).  We make
the per-unit field sigma a differentiable leaf, take the autograd gold, and compare.

    cos(true-weight psi_sigma, gold) -- isolates the -d/sigma identity (mirror error = 0)
    cos(mirror     psi_sigma, gold) -- the full forward-only estimate

The -d/sigma factor is exact only in the coupled crossing-width regime (h ~ sigma); this
crossing uses a fixed h, so the cosine also measures how good that approximation is.

    .venv/bin/python tmp/rl_sigma_credit.py --H 16 32 64 128 --n_layers 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from rl.policy import CartPolePolicy
from rl import credit as C
from rl.env import collect_states


def _cos(a, b):
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 1e-12 else float("nan")


def measure(H, n_layers, sigma, states, seed):
    torch.manual_seed(seed)
    policy = CartPolePolicy(obs_dim=states.shape[1], hidden=H, std=sigma, t=64,
                            n_hidden_layers=n_layers)
    field = [torch.full((H,), sigma).requires_grad_(True) for _ in range(n_layers)]
    policy.field = field
    cm, ct = [], []
    for i in range(states.shape[0]):
        step = policy.rollout_step(states[i:i + 1])
        for f in field:
            f.grad = None
        step.logp.backward()
        gold = torch.cat([field[l].grad.detach().reshape(-1) for l in range(n_layers)])

        Wo_t, Wh_t = C.true_weights(policy, step)
        p_id = C.sigma_grad(policy, step, Wo_t, Wh_t)          # -d/sigma identity (true W)
        p_fw = C.sigma_grad_forward(policy, step, Wo_t, Wh_t)  # forward dz/dsigma (true W)
        Wo, Wh = C.mirror_weights(policy, step)
        p_fm = C.sigma_grad_forward(policy, step, Wo, Wh)      # forward dz/dsigma (mirror W)
        v_id = torch.cat([p_id[l].detach().reshape(-1) for l in range(n_layers)])
        v_fw = torch.cat([p_fw[l].detach().reshape(-1) for l in range(n_layers)])
        v_fm = torch.cat([p_fm[l].detach().reshape(-1) for l in range(n_layers)])
        cm.append((_cos(v_id, gold), _cos(v_fw, gold), _cos(v_fm, gold)))
    arr = np.array(cm)
    return tuple(np.nanmedian(arr, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--n_layers", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--states", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    states = collect_states(n_states=args.states, seed=args.seed)
    states = (states - states.mean(0)) / (states.std(0) + 1e-6)
    print(f"# psi_sigma validation -- median cosine to autograd d log pi/d sigma "
          f"(n_layers={args.n_layers}, sigma={args.sigma}, {states.shape[0]} states)")
    print(f"{'H':>5} | {'-d/sig(true)':>13} {'fwd(true)':>10} {'fwd(mirror)':>12}")
    print("-" * 48)
    for H in args.H:
        c_id, c_fw, c_fm = measure(H, args.n_layers, args.sigma, states, args.seed)
        print(f"{H:>5} | {c_id:>13.3f} {c_fw:>10.3f} {c_fm:>12.3f}")


if __name__ == "__main__":
    main()
