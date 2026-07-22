"""rl_consolidate_rho -- §23.4 skill-reuse retry with rho/h-gated recruitment (§23.7).

The §23.4 two-phase curriculum (pretrain BALANCE on subnetwork A, freeze it, then learn
PUMP on subnetwork B) failed to protect the balance skill because sigma-only recruitment
LEAKS in deep layers: sigma=0 units keep firing from upstream sample fluctuations, and
the pump-trained readout-B columns read that leaked activity during balance mode.

Fix under test (idea_consolidation.md §4.5-4.6): the mobilisation dial rho drives BOTH
sigma = rho*sigma0 and h = h0/rho, so a rho=0 unit is silenced EXACTLY at any depth
(h sentinel) -- z = 0, KDE slope = 0, credit = 0 all strictly.

Arms (phase 2 both start from the SAME rho-gated phase-1 body, full-invariant freeze):
    rho    : rho-gated prototypes (the fix)
    sigma  : sigma-only prototypes (the §23.4 leaky regime, control)

Verifications:
    V1  strict silence: under P_balance the off half B has mean|z| == 0 in BOTH layers
        (rho arm); the sigma control leaks in layer 1.
    V2  balance retention after phase 2: composed policy near the top should match the
        phase-1 balance skill in the rho arm and degrade in the sigma arm.
    V3  frozen A-invariant drift == 0 (both arms; elementwise Adam + mask).
    V4  composed swing-up from the bottom (last100_up) -- the curriculum still works.

Usage:
    python tmp/rl_consolidate_rho.py p1     [--updates 200]
    python tmp/rl_consolidate_rho.py p2 --arm rho|sigma  [--updates 400]
    python tmp/rl_consolidate_rho.py report
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append("tmp")

import numpy as np
import torch

from rl import field as F
from rl.a2c_swingup import train_a2c, eval_from_bottom, _apply_rho, _set_field
from rl.envs_swingup import CartPoleSwingUp
from rl.train import RunningNorm

OUT = "tmp/out/rl_consolidate_rho"
H, HALF = 128, 64
SIGMA0, H0 = 0.6, 0.15
FORCE, XTHR = 20.0, 4.0
BAL_STARTS = [0.12, -0.15, 0.2, -0.25, 0.3, -0.1]   # handoff-realistic near-top starts

# The complete A-invariant: everything the balance function depends on under P_balance
# (§23.4 froze only l0 rows / l1 A-COLS / readout A-cols; l1 A-ROW biases+weights and the
# shared output bias were left free -- here the invariant is frozen in full).
INV_BLOCKS = ["fcs.0.weight[:64]", "fcs.0.bias[:64]", "fcs.1.weight[:64,:]",
              "fcs.1.bias[:64]", "fcs.2.weight[:,:64]", "fcs.2.bias"]


def freeze_mask_full():
    m = {}
    w0 = torch.zeros(H, 5); w0[:HALF, :] = 1
    b0 = torch.zeros(H); b0[:HALF] = 1
    w1 = torch.zeros(H, H); w1[:HALF, :] = 1            # A-l1 rows: ALL incoming weights
    b1 = torch.zeros(H); b1[:HALF] = 1
    w2 = torch.zeros(1, H); w2[:, :HALF] = 1
    b2 = torch.ones(1)                                  # shared output bias: part of balance
    return {(0, "weight"): w0, (0, "bias"): b0, (1, "weight"): w1, (1, "bias"): b1,
            (2, "weight"): w2, (2, "bias"): b2}


def invariant_drift(net_state, ref_state):
    d = {}
    d["fcs.0.weight[:64]"] = (net_state["fcs.0.weight"][:HALF] - ref_state["fcs.0.weight"][:HALF]).abs().max()
    d["fcs.0.bias[:64]"] = (net_state["fcs.0.bias"][:HALF] - ref_state["fcs.0.bias"][:HALF]).abs().max()
    d["fcs.1.weight[:64,:]"] = (net_state["fcs.1.weight"][:HALF, :] - ref_state["fcs.1.weight"][:HALF, :]).abs().max()
    d["fcs.1.bias[:64]"] = (net_state["fcs.1.bias"][:HALF] - ref_state["fcs.1.bias"][:HALF]).abs().max()
    d["fcs.2.weight[:,:64]"] = (net_state["fcs.2.weight"][:, :HALF] - ref_state["fcs.2.weight"][:, :HALF]).abs().max()
    d["fcs.2.bias"] = (net_state["fcs.2.bias"] - ref_state["fcs.2.bias"]).abs().max()
    return {k: float(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Field setters (used by the eval helpers).  Each fully specifies sigma AND h so that
# switching between arms on a shared policy object can never carry state over.
# ---------------------------------------------------------------------------
def arm_fields(arm):
    """Phase-2 prototypes [P_pump, P_balance] per arm.

    rho   : hard-disjoint rho prototypes (strict silence both ways)
    olap  : INTENTIONAL overlap -- pump mobilises everything (frozen A participates as a
            read-only vocabulary, consolidation's readout-share regime), while balance
            keeps B strictly silent (the protection stays structural)
    sigma : sigma-only prototypes = the §23.4 leaky regime (control)
    """
    if arm == "rho":
        return [F.recruit_rho(H, 1), F.recruit_rho(H, 0)], True
    if arm == "olap":
        return [[torch.ones(H), torch.ones(H)], F.recruit_rho(H, 0)], True
    return [F.recruit(H, SIGMA0, 1), F.recruit(H, SIGMA0, 0)], False


def setter_pure(arm, side):
    def s(policy, cos_theta):
        if arm == "sigma":
            policy.field = F.recruit(H, SIGMA0, side)
            for gc in policy.crossings:
                gc.h = H0
        else:
            _apply_rho(policy, F.recruit_rho(H, side), SIGMA0, H0)
    return s


def setter_gate(arm):
    fields, rho_mode = arm_fields(arm)

    def s(policy, cos_theta):
        if not rho_mode:
            for gc in policy.crossings:
                gc.h = H0
        _set_field(policy, fields, cos_theta, 6.0, 0.0, rho_mode, SIGMA0, H0)
    return s


def _rollout(policy, mean, std, field_setter, start_theta, horizon, seed, collect_z=False):
    env = CartPoleSwingUp(horizon=horizon, random_start=False, seed=seed,
                          force_mag=FORCE, x_threshold=XTHR, continuous=True)
    obs, _ = env.reset(seed=seed, start_theta=start_theta)
    cs, zmax = [], [0.0, 0.0]
    for _ in range(horizon):
        field_setter(policy, float(obs[2]))
        on = torch.clamp((torch.tensor(obs, dtype=torch.float32) - mean) / std, -5, 5)
        step = policy.rollout_step(on.unsqueeze(0), greedy=True)
        if collect_z:
            for l in (0, 1):
                zB = step.z[l][0, :, HALF:].mean(dim=0)          # [64] mean_T activity of B
                zmax[l] = max(zmax[l], float(zB.max()))
        obs, r, te, tr, _ = env.step(float(step.action.item()))
        cs.append(env.cos_theta())
    cs = np.array(cs)
    return (float(cs.mean()), float((cs > 0.9).mean()), float((cs[-100:] > 0.9).mean())), zmax


def eval_balance(policy, mean, std, field_setter, horizon=400):
    outs = [_rollout(policy, mean, std, field_setter, th, horizon, seed=i)[0]
            for i, th in enumerate(BAL_STARTS)]
    return tuple(float(np.mean([o[j] for o in outs])) for j in range(3))


def measure_silence(policy, mean, std, arm, horizon=100):
    """Max over B units / steps of the mean-T activity, per layer, under pure P_balance."""
    _, zmax = _rollout(policy, mean, std, setter_pure(arm, 0), 0.15, horizon, seed=0,
                       collect_z=True)
    return zmax


# ---------------------------------------------------------------------------
def run_p1(updates, horizon, seed=0, epu=6):
    from rl.a2c_swingup import build_policy
    P_bal = F.recruit_rho(H, 0)
    pol, _, norm, critic, cks, hist = train_a2c(
        seed=seed, H=H, sigma=SIGMA0, updates=updates, episodes_per_update=epu, horizon=horizon,
        gamma=0.99, lam=0.95, lr_actor=0.01, lr_critic=1e-3, critic_epochs=8,
        force_mag=FORCE, x_threshold=XTHR, sigma_explore=0.3, sigma_explore_end=0.05,
        fixed_field=P_bal, rho_mode=True, h0=H0, start_center=0.0, start_range=0.4,
        n_hidden_layers=2, energy_reward=False, checkpoint_every=50, verbose=True)
    # balance is the SKILL being pretrained: evaluate every checkpoint and keep the best
    # (training returns keep drifting under exploration; the greedy skill peaks earlier)
    best = None
    for upd, st in cks:
        p, mean, std_c = build_policy(st)
        bal = eval_balance(p, mean, std_c, setter_pure("rho", 0))
        print(f"  p1 ckpt upd {upd:4d}  balance: mean cos {bal[0]:+.3f}  "
              f"frac_up {bal[1]:.3f}  last100_up {bal[2]:.3f}")
        if best is None or bal[2] >= best[2][2]:
            best = (upd, st, bal)
    upd, st, bal = best
    p, mean, std_c = build_policy(st)
    sil_rho = measure_silence(p, mean, std_c, "rho")
    sil_sig = measure_silence(p, mean, std_c, "sigma")
    print(f"Phase1 BALANCE (rho field, handoff starts, best ckpt upd {upd}): "
          f"mean cos {bal[0]:+.3f}  frac_up {bal[1]:.3f}  last100_up {bal[2]:.3f}")
    print(f"V1 silence of B under P_bal  rho-gate: L0 {sil_rho[0]:.4f}  L1 {sil_rho[1]:.4f}"
          f"   |   sigma-only: L0 {sil_sig[0]:.4f}  L1 {sil_sig[1]:.4f}")
    torch.save({"body": st["net"], "norm_mean": st["norm_mean"].clone(),
                "norm_std": st["norm_std"].clone(), "H": H, "force_mag": FORCE,
                "best_upd": upd, "seed": seed,
                "balance_eval": bal, "silence_rho": sil_rho, "silence_sigma": sil_sig},
               f"{OUT}/p1_rho_s{seed}.pt")


def pick_p1():
    """Select the best phase-1 seed (balance last100_up) as the pretrained skill."""
    import glob
    import shutil
    cands = []
    for p in sorted(glob.glob(f"{OUT}/p1_rho_s*.pt")):
        d = torch.load(p, weights_only=False)
        cands.append((d["balance_eval"][2], d["balance_eval"][1], p, d))
        print(f"  {p}: seed {d['seed']} best_upd {d['best_upd']}  "
              f"balance last100_up {d['balance_eval'][2]:.3f}  frac_up {d['balance_eval'][1]:.3f}")
    best = max(cands)
    print(f"picked {best[2]}")
    shutil.copyfile(best[2], f"{OUT}/p1_rho.pt")


def run_p2(arm, updates, horizon):
    d = torch.load(f"{OUT}/p1_rho.pt", weights_only=False)
    norm = RunningNorm(5)
    norm.mean = d["norm_mean"].clone()
    norm.M2 = (d["norm_std"] ** 2).clone()
    norm.count = torch.tensor(1.0)
    fields, rho_mode = arm_fields(arm)
    pol, _, _, critic, cks, hist = train_a2c(
        seed=0, H=H, sigma=SIGMA0, updates=updates, episodes_per_update=3, horizon=horizon,
        gamma=0.99, lam=0.95, lr_actor=0.01, lr_critic=1e-3, critic_epochs=8,
        force_mag=FORCE, x_threshold=XTHR, sigma_explore=0.4, sigma_explore_end=0.1,
        fields=fields, gate_k=6.0, gate_c=0.0, rho_mode=rho_mode, h0=H0,
        init_body=d["body"], freeze_mask=freeze_mask_full(), n_hidden_layers=2,
        energy_reward=True, norm_obj=norm, update_norm=False,
        checkpoint_every=50, verbose=True)

    drift = invariant_drift(pol.net.state_dict(), d["body"])
    std = d["norm_std"]
    ret_comp = eval_balance(pol, norm.mean, std, setter_gate(arm))
    ret_pure = eval_balance(pol, norm.mean, std, setter_pure(arm, 0))
    sil = measure_silence(pol, norm.mean, std, arm)
    print(f"[{arm}] V3 invariant drift: " +
          "  ".join(f"{k}={v:.2e}" for k, v in drift.items()))
    print(f"[{arm}] V2 balance retention (handoff starts)  composed-gate: "
          f"mean cos {ret_comp[0]:+.3f} frac_up {ret_comp[1]:.3f} last100_up {ret_comp[2]:.3f}"
          f"   pure P_bal: last100_up {ret_pure[2]:.3f}"
          f"   (phase-1 was last100_up {d['balance_eval'][2]:.3f})")
    print(f"[{arm}] V1 silence of B under P_bal: L0 {sil[0]:.4f}  L1 {sil[1]:.4f}")
    print(f"[{arm}] V4 composed swing-up from BOTTOM:")
    swing = []
    for upd, st in cks:
        mc, fu, tail = eval_from_bottom(st, horizon=500)
        swing.append((upd, mc, fu, tail))
        print(f"  upd {upd:4d}  mean cos {mc:+.3f}  frac_up {fu:.3f}  last100_up {tail:.3f}")
    torch.save({"checkpoints": cks, "hist": hist, "drift": drift,
                "retention_composed": ret_comp, "retention_pure": ret_pure,
                "silence": sil, "swing": swing}, f"{OUT}/p2_{arm}.pt")
    with open(f"{OUT}/p2_{arm}.json", "w") as f:
        json.dump({"arm": arm, "drift": drift, "retention_composed": ret_comp,
                   "retention_pure": ret_pure, "silence": sil, "swing": swing}, f, indent=1)


def report():
    d1 = torch.load(f"{OUT}/p1_rho.pt", weights_only=False)
    print("=== §23.7 rho/h-gated skill reuse: summary ===")
    print(f"phase-1 balance (rho, handoff starts): last100_up {d1['balance_eval'][2]:.3f}")
    print(f"phase-1 V1 silence   rho: {d1['silence_rho']}   sigma-only: {d1['silence_sigma']}")
    for arm in ("rho", "olap", "sigma"):
        p = f"{OUT}/p2_{arm}.pt"
        if not os.path.exists(p):
            print(f"[{arm}] (not run)")
            continue
        d2 = torch.load(p, weights_only=False)
        best = max(d2["swing"], key=lambda r: r[3])
        print(f"[{arm}] retention composed last100_up {d2['retention_composed'][2]:.3f} | "
              f"pure {d2['retention_pure'][2]:.3f} | silence L1 {d2['silence'][1]:.4f} | "
              f"max drift {max(d2['drift'].values()):.2e} | "
              f"best swing-up upd {best[0]} last100_up {best[3]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["p1", "p1pick", "p2", "report"])
    ap.add_argument("--arm", choices=["rho", "olap", "sigma"], default="rho")
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    if args.phase == "p1":
        run_p1(args.updates or 200, args.horizon or 300, args.seed)
    elif args.phase == "p1pick":
        pick_p1()
    elif args.phase == "p2":
        run_p2(args.arm, args.updates or 400, args.horizon or 400)
    else:
        report()
