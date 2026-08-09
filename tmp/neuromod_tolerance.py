"""E6: does tolerance need full retraining, or do multiplicative gains suffice?

Protocol (docs/idea_neuromod.md section 8, E6; results feed section 7.13):

  1. Train a standard net at field strength 0.8 (sector + sample + blended 40k).
  2. Chronically shift the field to alpha x 0.8 for alpha in {0.25, 0.5, 2, 4}.
  3. Compare four adaptation variants at the shifted level:
       none       no adaptation at all (the acute condition)
       rule       a task-free local rule: scale layer-1 synapses by alpha
                  (g1 = alpha), nothing else -- the naive "synaptic scaling
                  follows the tone" homeostat.  Zero learned parameters.
       scalar3    learn 3 per-layer gains (g1, g2, g3) on the blended objective,
                  all 25920 weights frozen.
       perunit    learn per-unit gains on both hidden layers + readout gain
                  (144 + 144 + 1 = 289 params), weights frozen.
  4. Reference: full retraining at the shifted level (= experiment 2/3 plateau,
     foods ~ 7.0, home ~ 1.0).

A gain g_l multiplies the OUTPUT of linear layer l (weights and bias together),
which is the model counterpart of multiplicative synaptic scaling (Turrigiano).
The crossing thresholds h and the noise field are untouched.

What this does and does not answer: E6 is a CAPACITY question -- does the
low-dimensional scaling manifold contain a working solution at the new tone?
The learned variants use the task objective to search that manifold; only the
`rule` variant is also a statement about a biologically plausible LEARNING RULE.

Run from the repository root:
    .venv/bin/python tmp/neuromod_tolerance.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from neuromod import fields as F
from neuromod import protocol as P
from neuromod import world

N_HIDDEN = 2
BASE = 0.8
ALPHAS = [0.25, 0.5, 2.0, 4.0]
SEEDS = [7, 11, 23]


class GainAdapter(nn.Module):
    """Frozen SimpleNNNSample plus multiplicative per-layer (or per-unit) gains."""

    def __init__(self, net, per_unit: bool, hidden: int):
        super().__init__()
        self.net = net
        for p in self.net.parameters():
            p.requires_grad_(False)
        shape = (hidden,) if per_unit else ()
        self.g1 = nn.Parameter(torch.ones(shape))
        self.g2 = nn.Parameter(torch.ones(shape))
        self.g3 = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor, stds: list = None) -> torch.Tensor:
        n = self.net
        x = self.g1 * n.fcs[0](x)
        x = n.sampled_layer(x)
        x = n.gaussian_crossing[0](x, stds[0])
        x = self.g2 * n.fcs[1](x)
        x = n.gaussian_crossing[1](x, stds[1])
        x = n.fcs[2](x)
        x = n.ensemble_layer(x)
        return self.g3 * x


def train_base(seed, args, objects, nf, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = P.build_network(args.hidden_dim, n_hidden=N_HIDDEN, base_std=BASE,
                          kind="sample", t=args.samples, crossing_h=args.crossing_h,
                          in_dim=world.obs_dim()).to(device)
    pool = np.random.default_rng(seed).uniform(-1, 1, (16384, 2)).astype(np.float32)
    with contextlib.redirect_stdout(io.StringIO()):
        P.train_blended(net, pool, objects, world.alpha_states(None), nf,
                        N_HIDDEN, args.train_steps, bs=512, lr=3e-4, seed=seed,
                        verbose=False)
    net.eval()
    return net


def adapt_gains(base_net, per_unit, nf_new, objects, seed, args, device):
    """Fit only the gains on the standard blended objective at the NEW field."""
    wrapper = GainAdapter(base_net, per_unit, args.hidden_dim).to(device)
    pool = np.random.default_rng(seed + 500).uniform(
        -1, 1, (16384, 2)).astype(np.float32)
    with contextlib.redirect_stdout(io.StringIO()):
        P.train_blended(wrapper, pool, objects, world.alpha_states(None), nf_new,
                        N_HIDDEN, args.adapt_steps, bs=512, lr=args.adapt_lr,
                        seed=seed + 500, verbose=False)
    wrapper.eval()
    return wrapper


def make_predict(model, device):
    def predict(o, f):
        with torch.no_grad():
            return P.evaluate_vector_field(
                model, torch.as_tensor(o, dtype=torch.float32, device=device),
                f, N_HIDDEN).cpu().numpy()
    return predict


def speed_ref_of(predict, objects, nf):
    pos = np.random.default_rng(0).uniform(-1, 1, (4096, 2)).astype(np.float32)
    obs = world.encode_observations(pos, objects)
    mags = [np.linalg.norm(predict(obs, F.blend_fields(nf, w, world.CATEGORIES)),
                           axis=1)
            for w in (np.float32([1, 0, 0]), np.float32([0, 1, 0]),
                      np.float32([0, 0, 1]), np.float32([.5, .5, 0]),
                      np.float32([0, .5, .5]))]
    return float(np.percentile(np.concatenate(mags), 30))


def behaviour(predict, nf, objects, sref, seed, args):
    rows = []
    for e in range(args.episodes):
        torch.manual_seed(1000 + e)
        params = world.LoopParams(speed_ref=sref, threat_seed=seed + e)
        rows.append(world.rollout(predict, nf, params, n_frames=args.frames,
                                  objects=objects, seed=seed + e))
    return (float(np.mean([r["foods_per_1k"] for r in rows])),
            float(np.mean([r["night_home_rate"] for r in rows])))


def task_err(model, obs, targets, nf):
    _, _, err = P.capability(model, obs, targets, nf, world.CATEGORIES,
                             world.STATES, world.STATE_TO_FIELD, N_HIDDEN)
    return err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-steps", type=int, default=40000)
    p.add_argument("--adapt-steps", type=int, default=3000)
    p.add_argument("--adapt-lr", type=float, default=0.01)
    p.add_argument("--hidden-dim", type=int, default=144)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--frames", type=int, default=1260)
    p.add_argument("--grid-side", type=int, default=61)
    p.add_argument("--out", default="tmp/out/sr_standard/tolerance.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects = world.make_scripted_objects()
    alphas_states = world.alpha_states(None)
    positions = world.make_training_grid(args.grid_side)
    obs = torch.tensor(world.encode_observations(positions, objects),
                       dtype=torch.float32, device=device)
    targets = {s: torch.tensor(world.make_behavior_targets(
                   positions, objects, alphas_states[s]),
                   dtype=torch.float32, device=device) for s in world.STATES}
    nf_base = {k: v.to(device) for k, v in F.build_fields(
        world.CATEGORIES, args.hidden_dim, BASE, 0.22, 0.15).items()}

    rows = []
    for seed in SEEDS:
        t0 = time.time()
        net = train_base(seed, args, objects, nf_base, device)
        pr = make_predict(net, device)
        sref0 = speed_ref_of(pr, objects, nf_base)
        f0, h0 = behaviour(pr, nf_base, objects, sref0, seed, args)
        e0 = task_err(net, obs, targets, nf_base)
        print(f"[seed {seed}] base trained ({time.time()-t0:.0f}s)  "
              f"foods={f0:.2f} home={h0:.2f} err={e0:.3f}")
        rows.append((seed, 1.0, "adapted-at", 0, f0, h0, e0))

        for alpha in ALPHAS:
            nf_new = {k: v * alpha for k, v in nf_base.items()}

            # none: acute, no adaptation (speed_ref stays the old one)
            fa, ha = behaviour(pr, nf_new, objects, sref0, seed, args)
            ea = task_err(net, obs, targets, nf_new)
            rows.append((seed, alpha, "none", 0, fa, ha, ea))

            # rule: g1 = alpha, task-free
            ruled = GainAdapter(net, False, args.hidden_dim).to(device)
            with torch.no_grad():
                ruled.g1.fill_(alpha)
            ruled.eval()
            prr = make_predict(ruled, device)
            srr = speed_ref_of(prr, objects, nf_new)
            fr, hr = behaviour(prr, nf_new, objects, srr, seed, args)
            er = task_err(ruled, obs, targets, nf_new)
            rows.append((seed, alpha, "rule", 0, fr, hr, er))

            # learned gain variants
            for name, per_unit, npar in (("scalar3", False, 3),
                                         ("perunit", True, 289)):
                t1 = time.time()
                ad = adapt_gains(net, per_unit, nf_new, objects, seed, args, device)
                pra = make_predict(ad, device)
                sra = speed_ref_of(pra, objects, nf_new)
                fl, hl = behaviour(pra, nf_new, objects, sra, seed, args)
                el = task_err(ad, obs, targets, nf_new)
                rows.append((seed, alpha, name, npar, fl, hl, el))
                g = ad
                print(f"  alpha={alpha:4.2f} {name:8s} ({time.time()-t1:.0f}s) "
                      f"foods={fl:5.2f} home={hl:.2f} err={el:.3f}  "
                      f"g1~{float(g.g1.mean()):.2f} g2~{float(g.g2.mean()):.2f} "
                      f"g3~{float(g.g3):.2f}")
            print(f"  alpha={alpha:4.2f} none     foods={fa:5.2f} home={ha:.2f} "
                  f"err={ea:.3f} | rule(g1=a) foods={fr:5.2f} home={hr:.2f} "
                  f"err={er:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    variants = {"adapted-at": 0, "none": 1, "rule": 2, "scalar3": 3, "perunit": 4}
    with open(out, "w") as f:
        f.write("# E6 tolerance-by-scaling (docs/idea_neuromod.md 7.13/E6)\n")
        f.write(f"# base=0.8 adapt_steps={args.adapt_steps} lr={args.adapt_lr} "
                f"episodes={args.episodes} frames={args.frames}\n")
        f.write("# variant codes: " + " ".join(f"{k}={v}" for k, v in
                                               variants.items()) + "\n")
        f.write("# columns: seed,alpha,variant_code,n_params,foods_per_1k,"
                "night_home_rate,task_err\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{variants[r[2]]},{r[3]},"
                    f"{r[4]:.4f},{r[5]:.4f},{r[6]:.5f}\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
