"""Two follow-ups on the distribution axis (docs/idea_neuromod.md 9.4).

Part 1 -- why is the shape-mode residual larger?  Decompose the dual-task
(sin/cos) shape-mode error into BIAS (mean prediction vs target) and VARIANCE
(finite-T sampling jitter).  If the residual is variance-dominated, the
interpretation "the two variance-matched distributions differ only weakly, so
the readout must amplify a small differential signal -- and with it the
sampling noise" is supported.

Part 2 -- is the shape axis gauge-absorbable?  Train sin under ONE condition,
freeze all weights, and let per-unit multiplicative gains (g1[64], g2[64], g3)
adapt to a NEW condition on the task loss (best case for adaptation):

    intensity control   gaussian sigma0  ->  gaussian 2*sigma0
                        (known to be absorbable; E6)
    shape test          variance-matched uniform  ->  gaussian
                        (prediction: NOT fully absorbable -- an affine map
                        cannot turn the uniform's compact-support response
                        into the gaussian's smooth-tailed one)

Run from the repository root:
    .venv/bin/python tmp/neuromod_shape_gauge.py
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "shape_axis", Path(__file__).parent / "neuromod_shape_axis.py")
SA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SA)

SEEDS = [7, 11, 23]
SIGMA0 = SA.SIGMA0


class GainWrap(nn.Module):
    """Frozen AxisNNN + per-unit multiplicative gains (the E6 adaptation class)."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.g1 = nn.Parameter(torch.ones(SA.HID))
        self.g2 = nn.Parameter(torch.ones(SA.HID))
        self.g3 = nn.Parameter(torch.ones(()))

    def forward(self, x, cond):
        n, dev = self.net, x.device
        a1 = (self.g1 * n.fc1(x)).unsqueeze(1).repeat(1, SA.T_SAMPLES, 1)
        from nnn import activation
        z1 = activation.CrossingSample.apply(
            a1 + n._noise(a1.shape, cond, dev), SA.H_CROSS)
        a2 = self.g2 * n.fc2(z1)
        z2 = activation.CrossingSample.apply(
            a2 + n._noise(a2.shape, cond, dev), SA.H_CROSS)
        return self.g3 * n.fc3(z2).mean(dim=1)


def train_single(cond, seed, x, y, device, epochs, lr):
    torch.manual_seed(seed)
    net = SA.AxisNNN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        loss = ((net(x, cond) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return net

def err_of(model, x, y, cond, n_eval=1):
    with torch.no_grad():
        if n_eval == 1:
            return float(((model(x, cond) - y) ** 2).mean())
        ym = torch.stack([model(x, cond) for _ in range(n_eval)]).mean(0)
        return float(((ym - y) ** 2).mean())


def adapt_gains(net, cond, x, y, device, steps, lr):
    w = GainWrap(net).to(device)
    opt = torch.optim.Adam([w.g1, w.g2, w.g3], lr=lr)
    for _ in range(steps):
        loss = ((w(x, cond) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    w.eval()
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--adapt-steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--adapt-lr", type=float, default=0.01)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.linspace(-math.pi, math.pi, 256, device=device).unsqueeze(1)
    ys, yc = torch.sin(x), torch.cos(x)

    # ---------- Part 1: bias/variance decomposition of the dual shape task ----
    print("Part 1: shape-mode residual decomposition (dual sin/cos, seed 7)")
    ea, eb, _, _, net = SA.run_mode("shape", 7, x, ys, yc, device,
                                    args.epochs, args.lr)
    for name, y, cond in (("sin", ys, ("shape", 0.0)), ("cos", yc, ("shape", 1.0))):
        single = err_of(net, x, y, cond, 1)
        bias = err_of(net, x, y, cond, 64)
        print(f"  {name}: single-forward err={single:.4f}  "
              f"bias (64-avg)={bias:.4f}  variance share={(single-bias)/single:.0%}")

    # ---------- Part 2: gauge test --------------------------------------------
    print("\nPart 2: can per-unit gains absorb the change? (train sin, adapt)")
    full = SIGMA0 * torch.ones(SA.HID, device=device)
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        # intensity control: gaussian sigma0 -> gaussian 2 sigma0
        net_i = train_single(("region", full), seed, x, ys, device,
                             args.epochs, args.lr)
        home_i = err_of(net_i, x, ys, ("region", full), 8)
        tgt_i = ("region", 2.0 * full)
        acute_i = err_of(net_i, x, ys, tgt_i, 8)
        after_i = err_of(adapt_gains(net_i, tgt_i, x, ys, device,
                                     args.adapt_steps, args.adapt_lr),
                         x, ys, tgt_i, 8)
        # shape test: uniform -> gaussian (same intensity)
        net_s = train_single(("shape", 0.0), seed, x, ys, device,
                             args.epochs, args.lr)
        home_s = err_of(net_s, x, ys, ("shape", 0.0), 8)
        tgt_s = ("shape", 1.0)
        acute_s = err_of(net_s, x, ys, tgt_s, 8)
        after_s = err_of(adapt_gains(net_s, tgt_s, x, ys, device,
                                     args.adapt_steps, args.adapt_lr),
                         x, ys, tgt_s, 8)
        rows.append((home_i, acute_i, after_i, home_s, acute_s, after_s))
        print(f"  [seed {seed}] ({time.time()-t0:.0f}s)\n"
              f"    intensity x2 : home={home_i:.4f} acute={acute_i:.4f} "
              f"after-gains={after_i:.4f}\n"
              f"    uniform->gauss: home={home_s:.4f} acute={acute_s:.4f} "
              f"after-gains={after_s:.4f}")
    r = np.array(rows)
    print("\n3-seed mean:")
    print(f"  intensity x2 : home={r[:,0].mean():.4f} acute={r[:,1].mean():.4f} "
          f"after-gains={r[:,2].mean():.4f}  "
          f"(recovery {100*(r[:,1]-r[:,2]).mean()/max(1e-9,(r[:,1]-r[:,0]).mean()):.0f}%)")
    print(f"  uniform->gauss: home={r[:,3].mean():.4f} acute={r[:,4].mean():.4f} "
          f"after-gains={r[:,5].mean():.4f}  "
          f"(recovery {100*(r[:,4]-r[:,5]).mean()/max(1e-9,(r[:,4]-r[:,3]).mean()):.0f}%)")


if __name__ == "__main__":
    main()
