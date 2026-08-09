"""E8: close the tone-DECREASE side with intrinsic excitability, teacher-free.

E7 showed the nu-set-point rule restores function after tone increases but not
after decreases, using multiplicative gains alone.  The layer-1 algebra says
why: with the crossing thresholds +-h fixed, no gain (nor gain+bias) maps the
shrunken operating point back onto the old one.  What does map it back exactly
is the pair

    g  (pre-noise gain; synaptic scaling)      u = g*a + noise(alpha*sigma)
    s  (post-noise scale; since crossing(u/s, h) == crossing(u, h*s), s IS a
        per-unit threshold scale = intrinsic excitability)

with g = s = alpha: then (alpha*a + noise(alpha*sigma)) vs +-(alpha*h) is the
original computation rescaled.  This is the two-knob picture of section 5.2
(neuromodulators move excitability and fluctuation together; what matters is
the ratio).  Biology's counterpart of s is homeostatic intrinsic plasticity --
the very mechanism behind STG's recovery after decentralisation.

Variants (all teacher-free, nu-set-point objective as in E7, weights frozen
except where stated):

    bias        per-unit additive offsets d1, d2 before the noise (excitability
                as offset).  Exact compensation impossible (threshold pair +-h
                is symmetric; an offset cannot rescale it) -- measured anyway.
    gainthresh  per-unit g1, g2 (pre-noise) + s1, s2 (threshold scales).  The
                exact solution g = s = alpha exists in this space.
    freeW       ALL 25920 weights free under the same nu objective -- the
                control for "why not just update the weight matrix?".  Also
                reports weight drift and the task error back at the BASE tone
                (memory-erasure probe).

Run from the repository root:
    .venv/bin/python tmp/neuromod_homeostat2.py
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from neuromod import fields as F
from neuromod import world
from nnn import activation
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "neuromod_tolerance", Path(__file__).parent / "neuromod_tolerance.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

N_HIDDEN = 2
BASE = 0.8
ALPHAS = [0.25, 0.5, 4.0]
SEEDS = [7, 11, 23]


class HomeoAdapter(nn.Module):
    """Frozen net + teacher-free homeostatic knobs.

    mode "bias":       u_l = fc_l(.) + d_l          (offset excitability)
    mode "gainthresh": u_l = g_l * fc_l(.),  crossing input scaled by 1/s_l
                       (equivalently per-unit thresholds h * s_l)
    """

    def __init__(self, net, mode: str, hidden: int):
        super().__init__()
        self.net = net
        self.mode = mode
        for p in self.net.parameters():
            p.requires_grad_(False)
        if mode == "bias":
            self.d1 = nn.Parameter(torch.zeros(hidden))
            self.d2 = nn.Parameter(torch.zeros(hidden))
        elif mode == "gainthresh":
            # log-parameterised so gains and threshold scales stay positive
            # (a raw parameter clamped in the forward gets stuck at the clamp)
            self.lg1 = nn.Parameter(torch.zeros(hidden))
            self.lg2 = nn.Parameter(torch.zeros(hidden))
            self.ls1 = nn.Parameter(torch.zeros(hidden))
            self.ls2 = nn.Parameter(torch.zeros(hidden))
        else:
            raise ValueError(mode)

    def g(self, i):
        return torch.exp(self.lg1 if i == 0 else self.lg2)

    def sscale(self, i):
        return torch.exp(self.ls1 if i == 0 else self.ls2)

    def _layer(self, x, i, field):
        n = self.net
        a = n.fcs[i](x)
        if self.mode == "bias":
            a = a + (self.d1 if i == 0 else self.d2)
        else:
            a = self.g(i) * a
        if i == 0:
            a = n.sampled_layer(a)
        cross = n.gaussian_crossing[i]
        u = cross.noise_layer(a, std=field)
        if self.mode == "gainthresh":
            u = u / self.sscale(i)
        return activation.CrossingSample.apply(u, cross.h)

    def forward(self, x: torch.Tensor, stds: list = None) -> torch.Tensor:
        z1 = self._layer(x, 0, stds[0])
        z2 = self._layer(z1, 1, stds[1])
        return self.net.ensemble_layer(self.net.fcs[2](z2))

    def rates(self, x, field):
        z1 = self._layer(x, 0, field)
        z2 = self._layer(z1, 1, field)
        return z1.mean(dim=(0, 1)), z2.mean(dim=(0, 1))


def plain_rates(net, x, field):
    a1 = net.sampled_layer(net.fcs[0](x))
    z1 = net.gaussian_crossing[0](a1, field)
    z2 = net.gaussian_crossing[1](net.fcs[1](z1), field)
    return z1.mean(dim=(0, 1)), z2.mean(dim=(0, 1))


def run_homeostat(rate_fn, params, nf_new, setpoints, objects, seed, steps, lr,
                  device):
    nu1_star, nu2_star = setpoints
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(seed + 900)
    field = F.blend_fields(nf_new, np.float32([1, 1, 1]) / 3, world.CATEGORIES)
    for _ in range(steps):
        pos = rng.uniform(-1, 1, (256, 2)).astype(np.float32)
        obs = torch.tensor(world.encode_observations(pos, objects),
                          dtype=torch.float32, device=device)
        nu1, nu2 = rate_fn(obs, field)
        loss = ((nu1 - nu1_star) ** 2).sum() + ((nu2 - nu2_star) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-steps", type=int, default=40000)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--freew-lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=144)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--frames", type=int, default=1260)
    p.add_argument("--grid-side", type=int, default=61)
    p.add_argument("--out", default="tmp/out/sr_standard/homeostat2.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects = world.make_scripted_objects()
    alphas_states = world.alpha_states(None)
    positions = world.make_training_grid(args.grid_side)
    obs_grid = torch.tensor(world.encode_observations(positions, objects),
                            dtype=torch.float32, device=device)
    targets = {s: torch.tensor(world.make_behavior_targets(
                   positions, objects, alphas_states[s]),
                   dtype=torch.float32, device=device) for s in world.STATES}
    nf_base = {k: v.to(device) for k, v in F.build_fields(
        world.CATEGORIES, args.hidden_dim, BASE, 0.22, 0.15).items()}

    rows = []
    for seed in SEEDS:
        t0 = time.time()
        net = T.train_base(seed, args, objects, nf_base, device)
        print(f"[seed {seed}] base trained ({time.time()-t0:.0f}s)")
        probe = torch.tensor(world.encode_observations(
            np.random.default_rng(0).uniform(-1, 1, (2048, 2)).astype(np.float32),
            objects), dtype=torch.float32, device=device)
        field0 = F.blend_fields(nf_base, np.float32([1, 1, 1]) / 3,
                                world.CATEGORIES)
        with torch.no_grad():
            setpoints = plain_rates(net, probe, field0)

        for alpha in ALPHAS:
            nf_new = {k: v * alpha for k, v in nf_base.items()}
            for mode in ("bias", "gainthresh", "freeW"):
                t1 = time.time()
                if mode == "freeW":
                    model = copy.deepcopy(net)
                    for q in model.parameters():
                        q.requires_grad_(True)
                    run_homeostat(lambda x, f: plain_rates(model, x, f),
                                  list(model.parameters()), nf_new, setpoints,
                                  objects, seed, args.steps, args.freew_lr,
                                  device)
                    drift = float(sum(
                        (a - b).norm() ** 2 for a, b in
                        zip(model.parameters(), net.parameters())).sqrt() /
                        sum(b.norm() ** 2 for b in net.parameters()).sqrt())
                    err_base = T.task_err(model, obs_grid, targets, nf_base)
                else:
                    model = HomeoAdapter(net, mode, args.hidden_dim).to(device)
                    run_homeostat(model.rates,
                                  [q for q in model.parameters()
                                   if q.requires_grad], nf_new, setpoints,
                                  objects, seed, args.steps, args.lr, device)
                    drift, err_base = 0.0, float("nan")
                model.eval()
                pr = T.make_predict(model, device)
                sr = T.speed_ref_of(pr, objects, nf_new)
                fl, hl = T.behaviour(pr, nf_new, objects, sr, seed, args)
                el = T.task_err(model, obs_grid, targets, nf_new)
                extra = ""
                if mode == "gainthresh":
                    extra = (f" g1~{float(model.g(0).detach().mean()):.2f}"
                             f" s1~{float(model.sscale(0).detach().mean()):.2f}"
                             f" g2~{float(model.g(1).detach().mean()):.2f}"
                             f" s2~{float(model.sscale(1).detach().mean()):.2f}")
                if mode == "freeW":
                    extra = f" drift={drift:.3f} err@base={err_base:.3f}"
                print(f"  alpha={alpha:4.2f} {mode:10s} ({time.time()-t1:.0f}s) "
                      f"foods={fl:5.2f} home={hl:.2f} err={el:.3f}{extra}")
                rows.append((seed, alpha, mode, fl, hl, el, drift, err_base))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    codes = {"bias": 0, "gainthresh": 1, "freeW": 2}
    with open(out, "w") as f:
        f.write("# E8 intrinsic-excitability homeostat + freeW control\n")
        f.write(f"# base=0.8 steps={args.steps} lr={args.lr} "
                f"freew_lr={args.freew_lr}\n")
        f.write("# variant codes: bias=0 gainthresh=1 freeW=2\n")
        f.write("# columns: seed,alpha,variant,foods_per_1k,night_home_rate,"
                "task_err,weight_drift,task_err_at_base\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{codes[r[2]]},{r[3]:.4f},{r[4]:.4f},"
                    f"{r[5]:.5f},{r[6]:.4f},{r[7]:.5f}\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
