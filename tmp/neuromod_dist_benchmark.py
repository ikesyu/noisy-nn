"""The 3-behaviour benchmark with DISTRIBUTION addressing instead of fields.

Reproduces the standard-benchmark pipeline (appendix C) with one substitution:
the spatial noise fields are removed entirely.  Every unit receives the same
intensity sigma0; what differs between behavioural states is the SHAPE of the
noise distribution (all variance-matched, 9.4):

    default (--assign food=burst,threat=skewmix,shelter=bimodal — the working
    triplet, idea_neuromod.md 9.4 measurement 8):
      food     burst    scale mixture q N(0,sb^2) + (1-q) N(0,ss^2)
      threat   skewmix  asymmetric mixture 3/4 N(-m,s^2) + 1/4 N(+3m,s^2)
      shelter  bimodal  0.5 N(-mu,s^2) + 0.5 N(mu,s^2)
    Legacy triplets via --shelter-dist keep food=gauss threat=bimodal.

Triplet rationale (measurement 7): pairwise channel width is NOT sufficient —
the mixture's effective activation is a convex combination of the members', so
a distribution inside the mixture hull of the other two is unidentifiable
(Gaussian ~ 83:17 bimodal/burst matches both variance and kurtosis, which is
why every gauss-assigned state collapsed).  The working triplet spans a
non-degenerate triangle in (skew, kurtosis): (0,11.4), (1.0,2.45), (0,1.34) —
skew is unreachable by symmetric mixtures.

Mixing: the arbitration weight w(t) picks, per noise element, WHICH
distribution the sample is drawn from (component mixture; variance is sigma0^2
for every w).  Concentration c scales every distribution linearly, so the
intensity axis of experiment 1 survives unchanged.

Everything else is the standard process: blended training on the runtime
manifold (protocol.train_blended, reused as-is), capability metrics on the
61x61 grid, closed-loop rollouts with the standard LoopParams, and a
concentration sweep.  The interface trick: the "fields" dict maps each
category to a 3-dim BASIS vector, so blend_fields(w) * c hands the model the
vector c*w, from which it decodes concentration and mixture weights.

Run from the repository root:
    .venv/bin/python tmp/neuromod_dist_benchmark.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from nnn import activation
from neuromod import fields as F
from neuromod import protocol as P
from neuromod import world
import importlib.util

_s1 = importlib.util.spec_from_file_location(
    "neuromod_tolerance", Path(__file__).parent / "neuromod_tolerance.py")
T = importlib.util.module_from_spec(_s1)
_s1.loader.exec_module(T)

N_HIDDEN = 2
HID = 144
SIGMA0 = 0.8
S_MODE = 0.3 * SIGMA0
MU = math.sqrt(SIGMA0 ** 2 - S_MODE ** 2)
SQRT3 = math.sqrt(3.0)
Q_BURST, SS_BURST = 0.25, 0.15
SB_BURST = math.sqrt((SIGMA0 ** 2 - (1 - Q_BURST) * SS_BURST ** 2) / Q_BURST)
SHELTER_DIST = "burst"          # set from --shelter-dist in main()
# behaviour -> distribution kind; SHELTER_DIST only overrides ASSIGN["shelter"]
ASSIGN = {"food": "gauss", "threat": "bimodal", "shelter": "burst"}
SEEDS = [7, 11, 23]
CONCENTRATIONS = [0.05, 0.1, 0.2, 0.35, 0.7, 1.0, 1.5, 2.5, 4.0]

# categories order = ("food", "threat", "shelter")
PSEUDO = {c: torch.zeros(3) for c in world.CATEGORIES}
for i, c in enumerate(world.CATEGORIES):
    PSEUDO[c][i] = 1.0


def _draw(kind, shape, device):
    if kind == "gauss":
        return SIGMA0 * torch.randn(shape, device=device)
    if kind == "bimodal":
        sign = torch.where(torch.rand(shape, device=device) < 0.5, -1.0, 1.0)
        return sign * MU + S_MODE * torch.randn(shape, device=device)
    if kind == "uniform":
        return (torch.rand(shape, device=device) * 2.0 - 1.0) * SQRT3 * SIGMA0
    if kind == "laplace":                       # var 2b^2 = sigma0^2
        b = SIGMA0 / math.sqrt(2.0)
        u = torch.rand(shape, device=device) - 0.5
        return -b * torch.sign(u) * torch.log1p(-2.0 * u.abs().clamp(max=0.4999999))
    if kind == "burst":                         # scale mixture
        wide = (torch.rand(shape, device=device) < Q_BURST).float()
        return ((wide * SB_BURST + (1 - wide) * SS_BURST)
                * torch.randn(shape, device=device))
    if kind == "exp":                           # shot-noise-like, skew 2
        u = torch.rand(shape, device=device).clamp(min=1e-12)
        return SIGMA0 * (-torch.log(u) - 1.0)
    if kind == "skewmix":                       # asymmetric location mixture:
        # 3/4 N(-m, s^2) + 1/4 N(+3m, s^2); mean 0, var 3m^2 + s^2 = sigma0^2
        s = S_MODE
        m = math.sqrt((SIGMA0 ** 2 - s * s) / 3.0)
        neg = (torch.rand(shape, device=device) < 0.75).float()
        mu = neg * (-m) + (1 - neg) * (3.0 * m)
        return mu + s * torch.randn(shape, device=device)
    raise ValueError(kind)


class DistNNN(nn.Module):
    """[32,144,144,2] sample NNN; `stds` entries are 3-vectors c*w."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(world.obs_dim(), HID)
        self.fc2 = nn.Linear(HID, HID)
        self.fc3 = nn.Linear(HID, 2, bias=False)
        self.t = 64
        self.h = 0.2

    def _noise(self, shape, v, device):
        tot = float(v.sum())
        if tot <= 1e-8:
            return torch.zeros(shape, device=device)
        w = (v / tot).to(device)
        r = torch.rand(shape, device=device)
        pick0 = (r < w[0]).float()
        pick1 = ((r >= w[0]) & (r < w[0] + w[1])).float()
        pick2 = 1.0 - pick0 - pick1
        picks = (pick0, pick1, pick2)
        out = torch.zeros(shape, device=device)
        for i, cat in enumerate(world.CATEGORIES):
            out = out + picks[i] * _draw(ASSIGN[cat], shape, device)
        return tot * out

    def forward(self, x, stds=None):
        dev = x.device
        a1 = self.fc1(x).unsqueeze(1).repeat(1, self.t, 1)
        z1 = activation.CrossingSample.apply(
            a1 + self._noise(a1.shape, stds[0], dev), self.h)
        a2 = self.fc2(z1)
        z2 = activation.CrossingSample.apply(
            a2 + self._noise(a2.shape, stds[1], dev), self.h)
        return self.fc3(z2).mean(dim=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-steps", type=int, default=40000)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--frames", type=int, default=1260)
    p.add_argument("--grid-side", type=int, default=61)
    p.add_argument("--shelter-dist", choices=("uniform", "burst", "laplace"),
                   default="burst")
    p.add_argument("--assign", default=None,
                   help="e.g. food=burst,threat=skewmix,shelter=bimodal "
                        "(overrides --shelter-dist)")
    p.add_argument("--burst-q", type=float, default=None)
    p.add_argument("--burst-ss", type=float, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    global SHELTER_DIST, Q_BURST, SS_BURST, SB_BURST
    SHELTER_DIST = args.shelter_dist
    ASSIGN["shelter"] = {"uniform": "uniform", "burst": "burst",
                         "laplace": "laplace"}[args.shelter_dist]
    if args.assign:
        for part in args.assign.split(","):
            cat, kind = part.split("=")
            assert cat in ASSIGN, cat
            ASSIGN[cat] = kind
    tag = "-".join(ASSIGN[c] for c in world.CATEGORIES)
    if args.burst_q is not None:
        Q_BURST = args.burst_q
    if args.burst_ss is not None:
        SS_BURST = args.burst_ss
    SB_BURST = math.sqrt((SIGMA0 ** 2 - (1 - Q_BURST) * SS_BURST ** 2) / Q_BURST)
    if args.out is None:
        args.out = (f"tmp/out/sr_standard/dist_benchmark_{tag}.csv"
                    if args.assign else "tmp/out/sr_standard/dist_benchmark"
                    + {"uniform": "", "burst": "_burst", "laplace": "_laplace"}
                      [args.shelter_dist] + ".csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects = world.make_scripted_objects()
    alphas = world.alpha_states(None)
    positions = world.make_training_grid(args.grid_side)
    obs = torch.tensor(world.encode_observations(positions, objects),
                       dtype=torch.float32, device=device)
    targets = {s: torch.tensor(world.make_behavior_targets(
                   positions, objects, alphas[s]),
                   dtype=torch.float32, device=device) for s in world.STATES}
    nf = {k: v.to(device) for k, v in PSEUDO.items()}
    rows = []

    for seed in SEEDS:
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = DistNNN().to(device)
        pool = np.random.default_rng(seed).uniform(
            -1, 1, (16384, 2)).astype(np.float32)
        with contextlib.redirect_stdout(io.StringIO()):
            P.train_blended(net, pool, objects, alphas, nf, N_HIDDEN,
                            args.train_steps, bs=512, lr=3e-4, seed=seed,
                            verbose=False)
        net.eval()
        predict = T.make_predict(net, device)
        sref = T.speed_ref_of(predict, objects, nf)
        print(f"[seed {seed}] trained ({time.time()-t0:.0f}s) "
              f"speed_ref={sref:.3f}")

        for c in CONCENTRATIONS:
            scaled = {k: v * c for k, v in nf.items()}
            sep, sig, err = P.capability(net, obs, targets, scaled,
                                         world.CATEGORIES, world.STATES,
                                         world.STATE_TO_FIELD, N_HIDDEN)
            beh = []
            for e in range(args.episodes):
                torch.manual_seed(1000 + e)
                params = world.LoopParams(speed_ref=sref, threat_seed=seed + e)
                beh.append(world.rollout(predict, nf, params,
                                         n_frames=args.frames,
                                         concentration=c, objects=objects,
                                         seed=seed + e))
            foods = float(np.mean([r["foods_per_1k"] for r in beh]))
            home = float(np.mean([r["night_home_rate"] for r in beh]))
            stall = float(np.mean([r["stall_frac"] for r in beh]))
            wallf = float(np.mean([r["wall_frac"] for r in beh]))
            contact = float(np.mean([r["contact_frac"] for r in beh]))
            diet = float(np.mean([r["diet_evenness"] for r in beh]))
            den = float(np.mean([r["den_evenness"] for r in beh]))
            rows.append([seed, c, sep, sig, err, foods, home, stall, wallf,
                         contact, diet, den])
            print(f"  c={c:5.2f}  sep={sep:.3f} signal={sig:.3f} err={err:.3f}"
                  f"  foods={foods:5.2f} home={home:.2f} stall={stall:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("# 3-behaviour benchmark with DISTRIBUTION addressing "
                f"(no spatial fields; assignment "
                + " ".join(f"{c}={ASSIGN[c]}" for c in world.CATEGORIES) + ")\n")
        f.write(f"# sigma0={SIGMA0} bimodal mu={MU:.3f} s={S_MODE:.3f}; "
                f"blended {args.train_steps} steps; standard LoopParams\n")
        f.write("# columns: seed,c,separation,signal,task_err,foods_per_1k,"
                "night_home_rate,stall_frac,wall_frac,contact_frac,"
                "diet_evenness,den_evenness\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" for v in r) + "\n")
    print(f"saved {out}")

    r = np.array(rows)
    at1 = r[np.isclose(r[:, 1], 1.0)]
    print("\n=== at c=1 (3-seed mean) vs region-method reference ===")
    print(f"  separation {at1[:,2].mean():.3f}   signal {at1[:,3].mean():.3f} "
          f"(region: 0.812)   task_err {at1[:,4].mean():.3f} (region: 0.043)")
    print(f"  foods {at1[:,5].mean():.2f} (region: 6.98)   "
          f"home {at1[:,6].mean():.2f} (region: 1.00)   "
          f"stall {at1[:,7].mean():.3f} (region: 0.000)")


if __name__ == "__main__":
    main()
