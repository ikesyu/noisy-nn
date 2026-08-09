"""E3: multiplexing lesion test on the STANDARD benchmark, with mask-defined sets.

The legacy lesion experiment (neuromod_lesion.py, vector sensing) needed
alpha_mix > 0 for the multiplexing verdict and defined recruited sets through
the crossing rate.  On the standard benchmark with BINARY mask fields
(experiment 8) the sets are defined BY CONSTRUCTION: the three binary supports
overlap pairwise in 23-24 units, and pure-corner training forces those shared
units to serve both behaviours -- if the behaviours are truly multiplexed
there rather than partitioned.

Protocol (per seed x alpha setting):
    train    pure-corner, binary fields (as experiment 8)
    lesion   kill triple (sigma=0, h->inf, downstream column zeroed) at layer 1
    groups   per pair (a,b), all size-matched to |shared(a,b)|:
                 shared     the mask intersection itself
                 a-only     draws from a's private support
                 b-only     draws from b's private support
                 random     draws from the union
    measure  per-behaviour task error on the 61x61 grid before/after
    verdict  multiplexing = the shared lesion degrades BOTH behaviours,
             on the scale set by the private lesions

alpha settings: one-hot (the demo default) and mix=0.3 (the legacy requirement).
Whether the verdict needs the mix is exactly the E3 question.

Run from the repository root:
    .venv/bin/python tmp/neuromod_lesion_standard.py
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import itertools
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from neuromod import fields as F
from neuromod import protocol as P
from neuromod import world

N_HIDDEN = 2
SIGMA0 = 0.8
SEEDS = [7, 11, 23]
LAYER = 0
N_DRAWS = 3


def binary_fields(hidden, device):
    graded = F.build_fields(world.CATEGORIES, hidden, SIGMA0, 0.22, 0.15)
    return {k: (SIGMA0 * (v > 0).float()).to(device) for k, v in graded.items()}


def per_behaviour_error(net, nf, obs, targets):
    out = {}
    with torch.no_grad():
        for s, f in world.STATE_TO_FIELD.items():
            y = P.evaluate_vector_field(net, obs, nf[f], N_HIDDEN)
            out[f] = float(((y - targets[s]) ** 2).mean())
    return out


def draws(pool, k, n, rng, size):
    ms = []
    for _ in range(n):
        m = np.zeros(size, dtype=bool)
        m[rng.choice(pool, size=k, replace=False)] = True
        ms.append(m)
    return ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden-dim", type=int, default=144)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--grid-side", type=int, default=61)
    p.add_argument("--out", default="tmp/out/sr_standard/lesion_standard.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects = world.make_scripted_objects()
    positions = world.make_training_grid(args.grid_side)
    obs = torch.tensor(world.encode_observations(positions, objects),
                       dtype=torch.float32, device=device)
    nf = binary_fields(args.hidden_dim, device)
    masks = {c: (nf[c] > 0).cpu().numpy() for c in world.CATEGORIES}
    rows = []

    for mix in (None, 0.3):
        alphas = world.alpha_states(mix)
        targets = {s: torch.tensor(world.make_behavior_targets(
                       positions, objects, alphas[s]),
                       dtype=torch.float32, device=device)
                   for s in world.STATES}
        state_fields = {s: nf[world.STATE_TO_FIELD[s]] for s in world.STATES}
        for seed in SEEDS:
            t0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)
            rng = np.random.default_rng(seed + 100)
            net = P.build_network(args.hidden_dim, n_hidden=N_HIDDEN,
                                  base_std=SIGMA0, kind="sample",
                                  t=args.samples, crossing_h=args.crossing_h,
                                  in_dim=world.obs_dim()).to(device)
            with contextlib.redirect_stdout(io.StringIO()):
                P.train(net, obs, targets, state_fields, world.STATES,
                        N_HIDDEN, args.epochs, args.lr, chunk=1024)
            net.eval()
            base = per_behaviour_error(net, nf, obs, targets)
            print(f"[mix={mix} seed {seed}] trained ({time.time()-t0:.0f}s)  "
                  + " ".join(f"{c}={v:.4f}" for c, v in base.items()))

            for a, b in itertools.combinations(world.CATEGORIES, 2):
                shared = masks[a] & masks[b]
                k = int(shared.sum())
                groups = {
                    "shared": [shared],
                    "a-only": draws(np.flatnonzero(masks[a] & ~masks[b]), k,
                                    N_DRAWS, rng, shared.size),
                    "b-only": draws(np.flatnonzero(masks[b] & ~masks[a]), k,
                                    N_DRAWS, rng, shared.size),
                    "random": draws(np.flatnonzero(masks[a] | masks[b]), k,
                                    N_DRAWS, rng, shared.size),
                }
                for gname, group_masks in groups.items():
                    dam = []
                    for m in group_masks:
                        net_l = copy.deepcopy(net)
                        nf_l = {c: v.clone() for c, v in nf.items()}
                        F.kill_units(net_l, m, nf_l, layer=LAYER)
                        dam.append(per_behaviour_error(net_l, nf_l, obs, targets))
                    da = float(np.mean([d[a] for d in dam]) - base[a])
                    db = float(np.mean([d[b] for d in dam]) - base[b])
                    rows.append([0.0 if mix is None else mix, seed,
                                 world.CATEGORIES.index(a),
                                 world.CATEGORIES.index(b),
                                 ["shared", "a-only", "b-only",
                                  "random"].index(gname), k, da, db])
                sh = [r for r in rows[-4:] if r[4] == 0][0]
                print(f"    {a}|{b} (k={k}): shared d_{a}=+{sh[6]:.4f} "
                      f"d_{b}=+{sh[7]:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("# E3 lesion test on the standard benchmark "
                "(binary-mask pure training; kill triple at layer 1)\n")
        f.write("# group codes: shared=0 a-only=1 b-only=2 random=3; "
                "categories: food=0 threat=1 shelter=2\n")
        f.write("# columns: alpha_mix,seed,cat_a,cat_b,group,n_units,"
                "delta_err_a,delta_err_b\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" for v in r) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
