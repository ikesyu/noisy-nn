"""Channel-width check for the scale-mixture (burst) distribution (9.4).

Dual sin/cos at the sample level for the two edges the burst would carry in
the 3-behaviour benchmark: burst vs Gaussian and burst vs bimodal.  All
variance-matched to sigma0 = 0.8; burst = 0.25 N(0, 1.579^2) + 0.75 N(0, 0.15^2).

Run from the repository root:
    .venv/bin/python tmp/neuromod_shape_burst.py
"""
from __future__ import annotations
import math, time
from pathlib import Path
import numpy as np, torch
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location(
    "shape_axis", Path(__file__).parent / "neuromod_shape_axis.py")
SA = importlib.util.module_from_spec(_s); _s.loader.exec_module(SA)

S0 = SA.SIGMA0
Q, SS = 0.25, 0.15
SB = math.sqrt((S0**2 - (1-Q)*SS**2) / Q)
MU_B = math.sqrt(S0**2 - (0.3*S0)**2); SM = 0.3*S0

def draw(kind, shape, device):
    if kind == "gauss":
        return S0 * torch.randn(shape, device=device)
    if kind == "bimodal":
        sign = torch.where(torch.rand(shape, device=device) < 0.5, -1.0, 1.0)
        return sign * MU_B + SM * torch.randn(shape, device=device)
    if kind == "burst":
        wide = (torch.rand(shape, device=device) < Q).float()
        return (wide * SB + (1-wide) * SS) * torch.randn(shape, device=device)
    if kind == "skewmix":
        m = math.sqrt((S0**2 - SM**2) / 3.0)
        neg = (torch.rand(shape, device=device) < 0.75).float()
        return neg * (-m) + (1-neg) * (3.0*m) + SM * torch.randn(shape, device=device)
    raise ValueError(kind)

class PairNNN(SA.AxisNNN):
    """cond = ("pair", (kindA, kindB, w)): element-wise mixture of two kinds."""
    def _noise(self, shape, cond, device):
        kind, arg = cond
        if kind != "pair":
            return super()._noise(shape, cond, device)
        ka, kb, w = arg
        pick = (torch.rand(shape, device=device) < float(w)).float()
        return pick * draw(kb, shape, device) + (1-pick) * draw(ka, shape, device)

def run_pair(ka, kb, seed, x, ys, yc, device, epochs=20000, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    net = PairNNN().to(device)
    ca, cb = ("pair", (ka, kb, 0.0)), ("pair", (ka, kb, 1.0))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        loss = (((net(x, ca)-ys)**2).mean() + ((net(x, cb)-yc)**2).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        ea = float(((net(x, ca)-ys)**2).mean())
        eb = float(((net(x, cb)-yc)**2).mean())
        lams = [SA.implied_lambda(
            torch.stack([net(x, ("pair", (ka, kb, w))) for _ in range(16)]).mean(0),
            ys, yc) for w in SA.WS]
    return ea, eb, lams

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.linspace(-math.pi, math.pi, 256, device=device).unsqueeze(1)
    ys, yc = torch.sin(x), torch.cos(x)
    print(f"burst: {Q}*N(0,{SB:.3f}^2) + {1-Q}*N(0,{SS}^2)  (std {S0})")
    import os
    pairs = ((("skewmix", "bimodal"), ("skewmix", "burst"))
             if os.environ.get("PAIRS") == "skew" else
             (("burst", "gauss"), ("burst", "bimodal")))
    for ka, kb in pairs:
        errs = []
        for seed in SA.SEEDS:
            t0 = time.time()
            ea, eb, lams = run_pair(ka, kb, seed, x, ys, yc, device)
            errs.append((ea, eb))
            print(f"[{ka}-vs-{kb} seed {seed}] ({time.time()-t0:.0f}s) "
                  f"err_sin={ea:.4f} err_cos={eb:.4f} "
                  f"lam: {' '.join(f'{v:.2f}' for v in lams)}")
        e = np.array(errs)
        print(f"  -> mean {e[:,0].mean():.4f}/{e[:,1].mean():.4f} "
              f"(参考: uniform-gauss 0.020/0.039, bimodal-gauss 0.011/0.019, region 0.002)")

if __name__ == "__main__":
    main()
