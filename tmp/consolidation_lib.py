"""consolidation_lib.py — noise-field self-consolidation primitives
(docs/idea_consolidation.md).

Everything a consolidation experiment needs beyond the nnn base library
lives here: a model with per-unit noise/threshold vectors, the
vanishing-path operations (anneal, snap, kill), the persistent-state
cov_jac trainer that interleaves with annealing, redundancy scoring,
per-task mobilisation fields with their descriptors (readout + bias), and
the multi-task helpers used by the soft-consolidation experiments
(§12.8–§12.9).

Deliberately kept out: task construction, greedy anneal schedules, stop
rules, figures, and argument parsing — those differ per experiment and stay
in tmp/consolidation_*.py. Kept in tmp (not in the nnn package) until the
API has settled.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from nnn import model  # noqa: E402
from nnn.credit import EPS, ManualOpt, cov_weight  # noqa: E402
from nnn.stats import Capture, kde_slope  # noqa: E402

# Threshold sentinel: a dropped unit cannot cross even under upstream
# sample fluctuations (§4.5), so z = 0 holds exactly.
H_DEAD = 1.0e6


# ============================================================
# Model with per-unit noise field
# ============================================================
class ConsolidableNNN(model.SimpleNNNSample):
    """SimpleNNNSample + per-layer, per-unit noise vectors `sigma_vecs`.

    When forward is called without explicit stds, `sigma_vecs` (a list of [H]
    tensors) is used and broadcast to [N, T, H] inside GaussianNoiseLayer.
    If `h_vecs` is present, the crossing thresholds are set per unit as well,
    so both dials of the mobilisation variable rho (sigma = rho * sigma0,
    h = h0 / rho; §4.6) are addressable.
    """

    def forward(self, x: torch.Tensor, stds: list = None) -> torch.Tensor:
        if stds is None:
            stds = self.sigma_vecs
        if getattr(self, "h_vecs", None) is not None:
            for i, gc in enumerate(self.gaussian_crossing):
                gc.h = self.h_vecs[i]          # [H]; broadcast in Crossing
        return super().forward(x, stds)


def checkpoint(net):
    return (copy.deepcopy(net.state_dict()),
            [s.clone() for s in net.sigma_vecs],
            [h.clone() for h in net.h_vecs])


def restore(net, ckpt):
    state, sigs, hs = ckpt
    net.load_state_dict(copy.deepcopy(state))
    net.sigma_vecs = [s.clone() for s in sigs]
    net.h_vecs = [h.clone() for h in hs]


def kill_unit(net, l: int, k: int) -> None:
    """Exact retirement of unit (l, k).

    sigma_k = 0 only stops the injected noise; in deep layers (l >= 1) the
    upstream sample fluctuations keep driving crossings. Raising h_k to a
    finite sentinel makes z_k = 0 exact (§4.5), which also zeroes the KDE
    slope and the credit, so the zeroed next-layer column cannot regrow.
    """
    net.sigma_vecs[l][k] = 0.0
    if l >= 1:
        net.h_vecs[l][k] = H_DEAD
    net.fcs[l + 1].weight.data[:, k] = 0.0


def predict(net, x: torch.Tensor, passes: int = 8) -> np.ndarray:
    """Ensemble prediction averaged over stochastic passes (1-D output)."""
    with torch.no_grad():
        y = torch.stack([net(x) for _ in range(passes)], dim=0).mean(dim=0)
    return y.squeeze(1).cpu().numpy()


def n_active(net) -> int:
    return int(sum(int((s > 0).sum()) for s in net.sigma_vecs))


def noise_budget(net) -> float:
    return float(sum(float((s ** 2).sum()) for s in net.sigma_vecs))


def alive(net, l: int):
    return [k for k in range(net.sigma_vecs[l].shape[0])
            if float(net.sigma_vecs[l][k]) > 0]


# ============================================================
# Persistent-state cov_jac trainer
# (same maths as fncl_driver.train_cov with method="cov_jac", jac_track=True,
#  slope=kde; the weight mirrors and optimiser state persist across calls so
#  training can interleave with annealing at one-epoch granularity.)
# ============================================================
class CovJacTrainer:
    def __init__(self, net, x, t_target, lr: float, opt: str = "adam",
                 jac_ema: float = 0.9):
        self.net, self.x, self.t = net, x, t_target
        self.lr, self.jac_ema = lr, jac_ema
        self.cap = Capture(net)
        self.opt = ManualOpt(opt)
        self.W = {}                      # weight mirrors (EMA + Kolen-Pollack)
        self.grad_masks = None           # {l: (W_mask, b_mask)}: 0 = frozen
                                         # (protects input sides under sharing)
        self.losses = []

    def step(self) -> float:
        net, x, t, cap = self.net, self.x, self.t, self.cap
        n_hidden = cap.n_hidden
        with torch.no_grad():
            y = net(x)                                   # [N, 1]; hooks fire
            ys = cap.y_samples                           # [N, T, 1] pre-ensemble
            z = [cap.z[l] for l in range(n_hidden)]
            d = [cap.d[l] for l in range(n_hidden)]
            N, T = z[0].shape[0], z[0].shape[1]

            slope = [kde_slope(cap.crossings[l], d[l]) for l in range(n_hidden)]
            slope_mean = [s.mean(dim=1) for s in slope]  # [N, H]

            meas = {"out": cov_weight(ys, z[-1], pool=True)}
            for l in range(1, n_hidden):
                meas[l] = cov_weight(d[l], z[l - 1], pool=True)
            if not self.W:
                self.W.update(meas)
            else:
                for k, v in meas.items():
                    self.W[k] = self.jac_ema * self.W[k] + (1.0 - self.jac_ema) * v

            a = [None] * n_hidden
            err_out = 2.0 * (y - t)                      # [N, 1]
            a[-1] = err_out @ self.W["out"]              # [N,m]@[m,H] -> [N,H]
            for l in range(n_hidden - 2, -1, -1):
                dd_next = a[l + 1] * slope_mean[l + 1]
                a[l] = dd_next @ self.W[l + 1]

            z_prev = [x.unsqueeze(1).expand(N, T, x.shape[1]), z[0]]
            steps = {}
            for l in range(n_hidden):
                delta = a[l].unsqueeze(1) * slope[l]     # [N, T, H]
                gW = torch.einsum("nto,nti->oi", delta, z_prev[l]) / (N * T)
                gb = delta.mean(dim=(0, 1))
                if self.grad_masks is not None and l in self.grad_masks:
                    mW, mb = self.grad_masks[l]
                    gW, gb = gW * mW, gb * mb
                steps[l] = self.opt.update(f"w{l}", net.fcs[l].weight, gW, self.lr)
                self.opt.update(f"b{l}", net.fcs[l].bias, gb, self.lr)

            z_bar = z[-1].mean(dim=1)
            gWout = torch.einsum("no,ni->oi", err_out, z_bar) / N
            gbout = err_out.mean(dim=0)
            steps["out"] = self.opt.update("wout", net.fcs[-1].weight, gWout, self.lr)
            self.opt.update("bout", net.fcs[-1].bias, gbout, self.lr)

            self.W["out"] = self.W["out"] - steps["out"]     # Kolen-Pollack
            for l in range(1, n_hidden):
                self.W[l] = self.W[l] - steps[l]

            loss = float(((y - t) ** 2).mean())
        self.losses.append(loss)
        return loss

    def run(self, epochs: int):
        for _ in range(epochs):
            self.step()

    def ema(self, w: int = 10) -> float:
        tail = self.losses[-w:]
        return float(np.mean(tail)) if tail else float("inf")

    def close(self):
        self.cap.remove()


def trainer_state(tr):
    return (copy.deepcopy(tr.W), copy.deepcopy(tr.opt.m),
            copy.deepcopy(tr.opt.v), copy.deepcopy(tr.opt.step),
            len(tr.losses))


def trainer_rollback(tr, st):
    W, m, v, step, nloss = st
    tr.W = copy.deepcopy(W)
    tr.opt.m = copy.deepcopy(m)
    tr.opt.v = copy.deepcopy(v)
    tr.opt.step = copy.deepcopy(step)
    del tr.losses[nloss:]


# ============================================================
# Activity statistics and redundancy scoring (§6.5 / §7.5)
# ============================================================
def collect_stats(net, x, passes: int = 4):
    """Per-sample activities z[l] [N, P*T, H] and expectations zbar[l] [N, H]."""
    cap = Capture(net)
    zs = [[] for _ in range(cap.n_hidden)]
    with torch.no_grad():
        for _ in range(passes):
            net(x)
            for l in range(cap.n_hidden):
                zs[l].append(cap.z[l])
    cap.remove()
    z = [torch.cat(zs[l], dim=1) for l in range(cap.n_hidden)]
    zbar = [zz.mean(dim=1) for zz in z]
    return z, zbar


def ridge_fit(Zr: torch.Tensor, zk: torch.Tensor, lam: float):
    """Centred ridge regression zk ~ Zr a + c -> (a [R], c, resid [N])."""
    zm, km = Zr.mean(dim=0), zk.mean()
    Zc, kc = Zr - zm, zk - km
    R = Zr.shape[1]
    A = Zc.T @ Zc + lam * torch.eye(R, device=Zr.device)
    a = torch.linalg.solve(A, Zc.T @ kc)
    c = km - zm @ a
    resid = zk - Zr @ a - c
    return a, c, resid


def unit_score(zbar_l: torch.Tensor, k: int, rest: list, w2: float,
               lam: float):
    """Deletion cost S_k of one unit against an explicit compensation basis.

    `w2` is the squared norm of the unit's live outgoing weights and `rest`
    the basis the compensation can actually flow to. Returns 0.0 when the
    unit has no live consumers (free to delete), None when it has consumers
    but no basis (unscorable), else the S_k of §6.5.
    """
    if w2 == 0.0:
        return 0.0
    if not rest:
        return None
    _, _, resid = ridge_fit(zbar_l[:, rest], zbar_l[:, k], lam)
    return (w2 * float((resid ** 2).sum())
            / (float((zbar_l[:, k] ** 2).sum()) + EPS))


# ============================================================
# Mobilisation fields, freezing, and region bookkeeping (§12.8–§12.9)
# ============================================================
def set_field(net, active: dict, sigma0: float, h0: float) -> None:
    """Mobilise active={l: [units]} at (sigma0, h0), silence the rest.

    Silencing keeps the weights (a temporary h-gate, unlike kill_unit): the
    silenced units have z = 0 and zero slope, so they take part in neither
    the forward pass nor learning.
    """
    H = net.sigma_vecs[0].shape[0]
    for l in (0, 1):
        sig = torch.zeros(H)
        hv = torch.full((H,), H_DEAD)
        for k in active[l]:
            sig[k] = sigma0
            hv[k] = h0
        net.sigma_vecs[l] = sig.to(net.sigma_vecs[l].device)
        net.h_vecs[l] = hv.to(net.h_vecs[l].device)


def set_field_rho(net, own: dict, voc: dict, sigma0: float, h0: float,
                  rho: float) -> None:
    """Mobilise own fully (rho=1) and voc partially at the rho dial (§4.6)."""
    H = net.sigma_vecs[0].shape[0]
    for l in (0, 1):
        sig = torch.zeros(H)
        hv = torch.full((H,), H_DEAD)
        for k in own[l]:
            sig[k] = sigma0
            hv[k] = h0
        for k in voc[l]:
            sig[k] = rho * sigma0
            hv[k] = h0 / rho
        net.sigma_vecs[l] = sig.to(net.sigma_vecs[l].device)
        net.h_vecs[l] = hv.to(net.h_vecs[l].device)


def freeze_masks(H: int, past: dict, device, share_l1: bool = False):
    """CovJacTrainer.grad_masks that freeze the input side of past regions.

    W1: past L1 rows frozen. W2: past L2 rows (the vocabulary tuning curves)
    frozen whole. By default past L1 columns are frozen for every row too
    (cross weights into live L2 rows stay 0, so the only sharing channel is
    the readout). With share_l1=True those columns stay trainable on live
    rows: new L2 units may read the past L1 bump basis (two-tier sharing,
    §12.9.9). Past tasks never see those weights (their inference silences
    the new rows), so exact zero forgetting is preserved either way. The
    readout layer is the sharing channel itself: not masked.
    """
    mW0 = torch.ones(H, 1, device=device)
    mb0 = torch.ones(H, device=device)
    mW1 = torch.ones(H, H, device=device)
    mb1 = torch.ones(H, device=device)
    for k in past[0]:
        mW0[k, :] = 0.0
        mb0[k] = 0.0
        if not share_l1:
            mW1[:, k] = 0.0
    for k in past[1]:
        mW1[k, :] = 0.0
        mb1[k] = 0.0
    return {0: (mW0, mb0), 1: (mW1, mb1)}


def zero_cross_columns(net, past: dict) -> None:
    """Zero leftover cross weights from past L1 columns into live L2 rows.

    These are remnants of the past tasks' own training era; they are
    don't-care for the past tasks (their side is dormant then) but would
    contaminate the vocabulary once past L1 units fire in a shared forward.
    """
    H = net.fcs[1].weight.shape[0]
    rows = [r for r in range(H) if r not in past[1]]
    if past[0] and rows:
        ri = torch.tensor(rows).unsqueeze(1)
        ci = torch.tensor(sorted(past[0])).unsqueeze(0)
        net.fcs[1].weight.data[ri, ci] = 0.0


def region_snapshot(net, region: dict, cols: list = None):
    """Input-side parameters a task's function depends on (forgetting check).

    The readout (wout / b_out) is excluded: it is part of the per-task
    descriptor and restored at inference. `cols` selects which W2 columns of
    the region rows are part of the task's function — by default its own L1
    region; under two-tier sharing pass the task's support L1 (own + shared
    basis). Columns outside the support are don't-care: their units are
    dormant in every field the task ever runs under (and they can carry a
    harmless Adam-momentum residue after kill_unit; see §12.9.9).
    """
    i0 = torch.tensor(region[0], dtype=torch.long)
    i1 = torch.tensor(region[1], dtype=torch.long)
    ic = i0 if cols is None else torch.tensor(sorted(cols), dtype=torch.long)
    return {
        "w1": net.fcs[0].weight.data[i0].clone(),
        "b1": net.fcs[0].bias.data[i0].clone(),
        "w2": net.fcs[1].weight.data[i1][:, ic].clone(),
        "b2": net.fcs[1].bias.data[i1].clone(),
    }


def region_drift(net, region: dict, snap, cols: list = None) -> float:
    now = region_snapshot(net, region, cols=cols)
    diffs = [float((now[k] - snap[k]).abs().max())
             for k in snap if snap[k].numel() > 0]
    return max(diffs) if diffs else 0.0


def overlaps(sup_a: dict, sup_b: dict) -> int:
    return sum(len(set(sup_a[l]) & set(sup_b[l])) for l in (0, 1))


# ============================================================
# Task descriptors (registry dicts) and their metrics
# ------------------------------------------------------------
# A task descriptor is a dict with at least: "target" (np array), "support"
# and "region" ({l: [units]}), "sigma0", "h0", "wout", "b_out", "tol".
# ============================================================
def eval_with_descriptor(net, x, r: dict, passes: int = 16):
    """Restore a task descriptor (field + readout + b_out) and predict."""
    set_field(net, r["support"], r["sigma0"], r["h0"])
    net.fcs[2].weight.data.copy_(r["wout"])
    net.fcs[2].bias.data.copy_(r["b_out"])
    return predict(net, x, passes=passes)


def vocab_energy(net, x, r: dict, registry, passes: int = 4):
    """Variance split of the readout contributions: own vs vocabulary.

    Contribution c_j(x) = wout_j * zbar_j(x); the share is
    Var_x(vocab sum) / Var_x(total sum), also broken down per source task.
    """
    set_field(net, r["support"], r["sigma0"], r["h0"])
    net.fcs[2].weight.data.copy_(r["wout"])
    net.fcs[2].bias.data.copy_(r["b_out"])
    _, zbar = collect_stats(net, x, passes=passes)
    w = r["wout"].squeeze(0)                       # [H]
    c = zbar[1] * w                                # [N, H] unit contributions
    var = lambda f: float(f.var(unbiased=False))   # noqa: E731

    own2 = r["region"][1]
    voc2 = [j for j in r["support"][1] if j not in own2]
    f_all = c[:, r["support"][1]].sum(dim=1)
    v_all = var(f_all) + EPS
    share = var(c[:, voc2].sum(dim=1)) / v_all if voc2 else 0.0
    by_source = {}
    for s in registry:
        if s is r:
            break
        src2 = [j for j in s["region"][1] if j in voc2]
        by_source[s["name"]] = (var(c[:, src2].sum(dim=1)) / v_all
                                if src2 else 0.0)
    return share, by_source


def probe_tuning_corr(net, x, target, vocab: dict, sigma0: float, h0: float,
                      passes: int = 4):
    """|corr_x(zbar_k, y_new)| of each vocabulary L2 unit (§12.9 case 3).

    Mobilises the vocabulary alone for a few forward passes (no learning,
    no parameter change) and restores the field afterwards. Under
    readout-only sharing the vocabulary can only be used linearly, so the
    tuning correlation with the new target directly estimates whether the
    new readout can use the unit.
    """
    set_back = ([s.clone() for s in net.sigma_vecs],
                [h.clone() for h in net.h_vecs])
    set_field_rho(net, {0: [], 1: []}, vocab, sigma0, h0, 1.0)
    _, zbar = collect_stats(net, x, passes=passes)
    t = torch.tensor(target, device=x.device)
    tc = t - t.mean()
    scores = {}
    for k in vocab[1]:
        zk = zbar[1][:, k]
        zc = zk - zk.mean()
        denom = float(zc.norm()) * float(tc.norm())
        scores[k] = abs(float((zc @ tc))) / (denom + EPS)
    net.sigma_vecs, net.h_vecs = set_back
    return scores


# ============================================================
# The vanishing-path inner loop (shared by every anneal variant)
# ============================================================
def anneal_unit(net, l: int, k: int, run_block, over_tol, read_act, *,
                alpha: float, max_holds: int, snap_act: float,
                max_steps: int, abort_saturated: int = None):
    """Fade unit (l, k) along the vanishing path under a closed loop.

    Per anneal step: decay sigma_k by alpha (deep layers escalate h_k by
    1/alpha — the rho dial of §4.6), then call run_block("train") once and
    run_block("hold") while over_tol() up to max_holds. Stops when the
    measured mean activity read_act() drops below snap_act or after
    max_steps. The caller decides what a block trains (one task, or an
    alternating round over several) and performs the snap/kill afterwards.

    With abort_saturated=None (default) returns the total number of hold
    blocks — byte-compatible with every earlier caller. With an integer n,
    the attempt is ABORTED once n consecutive anneal steps saturate the
    closed loop (holds hit max_holds and the EMA is still over tolerance:
    the §12.5 leading indicator, used here to cut a failing attempt short
    instead of running it to max_steps), and the return becomes the tuple
    (holds, completed).
    """
    steps, holds, sat = 0, 0, 0
    completed = True
    while True:
        net.sigma_vecs[l][k] *= alpha
        if l >= 1:
            net.h_vecs[l][k] = net.h_vecs[l][k] / alpha
        run_block("train")
        h = 0
        while over_tol() and h < max_holds:
            run_block("hold")
            h += 1
        holds += h
        steps += 1
        if abort_saturated is not None:
            sat = sat + 1 if (h >= max_holds and over_tol()) else 0
            if sat >= abort_saturated:
                completed = False
                break
        if read_act() < snap_act or steps >= max_steps:
            break
    if abort_saturated is None:
        return holds
    return holds, completed


# ============================================================
# Multi-task contexts (per-task field + readout descriptor + trainer)
# ============================================================
class TaskCtx:
    """One task's mobilisation field, readout descriptor, and trainer.

    With `support` given, activate() installs the task's own field (per-task
    sigma/h vectors) and its readout; with support=None the field is left
    untouched (shared-field mode, §12.9 case 4) and only the readout swaps.
    """

    def __init__(self, net, x, target, tol, wout, b_out, args, name: str = "",
                 support: dict = None, sigma0: float = None, h0: float = None):
        self.name = name
        self.target = target
        self.tol = tol
        self.support = ({l: sorted(support[l]) for l in (0, 1)}
                        if support is not None else None)
        if support is not None:
            H = net.sigma_vecs[0].shape[0]
            self.sig = [torch.zeros(H) for _ in (0, 1)]
            self.h = [torch.full((H,), H_DEAD) for _ in (0, 1)]
            for l in (0, 1):
                for k in support[l]:
                    self.sig[l][k] = sigma0
                    self.h[l][k] = h0
        self.wout = wout.clone()
        self.b_out = b_out.clone()
        t = torch.tensor(target, device=x.device).unsqueeze(1)
        self.trainer = CovJacTrainer(net, x, t, lr=args.ft_lr, opt=args.opt,
                                     jac_ema=args.jac_ema)

    @classmethod
    def from_registry(cls, net, x, r: dict, args, support: dict = None,
                      target=None, tol: float = None):
        """Build from a task descriptor; support=None means shared-field."""
        return cls(net, x,
                   r["target"] if target is None else target,
                   r["tol"] if tol is None else tol,
                   r["wout"], r["b_out"], args, name=r["name"],
                   support=support, sigma0=r["sigma0"], h0=r["h0"])

    def field(self, l: int):
        return [k for k in range(self.sig[l].shape[0])
                if float(self.sig[l][k]) > 0]

    def activate(self, net):
        if self.support is not None:
            for l in (0, 1):
                net.sigma_vecs[l] = self.sig[l].to(net.sigma_vecs[l].device)
                net.h_vecs[l] = self.h[l].to(net.h_vecs[l].device)
        net.fcs[2].weight.data.copy_(self.wout)
        net.fcs[2].bias.data.copy_(self.b_out)

    def deactivate(self, net):
        self.wout = net.fcs[2].weight.data.clone()
        self.b_out = net.fcs[2].bias.data.clone()

    def step(self, net):
        self.activate(net)
        loss = self.trainer.step()
        self.deactivate(net)
        return loss

    def drop(self, net, l: int, k: int):
        """Leave this task's field only (h-gate; weights untouched)."""
        self.sig[l][k] = 0.0
        self.h[l][k] = H_DEAD
        if l == 1:
            self.wout[:, k] = 0.0

    def eval(self, net, x, passes: int = 16):
        self.activate(net)
        pred = predict(net, x, passes=passes)
        return float(np.mean((pred - self.target) ** 2)), pred


def joint_round(net, order) -> list:
    """Train every task in `order` one step each (alternating rehearsal)."""
    return [ctx.step(net) for ctx in order]


def any_over_tol(ctxs) -> bool:
    return any(ctx.trainer.ema() > ctx.tol for ctx in ctxs)


def joint_checkpoint(net, ctxs):
    return (checkpoint(net),
            [(ctx.wout.clone(), ctx.b_out.clone(), trainer_state(ctx.trainer),
              ([s.clone() for s in ctx.sig], [h.clone() for h in ctx.h])
              if ctx.support is not None else None) for ctx in ctxs])


def joint_restore(net, ctxs, ck):
    restore(net, ck[0])
    for ctx, st in zip(ctxs, ck[1]):
        ctx.wout = st[0].clone()
        ctx.b_out = st[1].clone()
        trainer_rollback(ctx.trainer, st[2])
        if st[3] is not None:
            ctx.sig = [s.clone() for s in st[3][0]]
            ctx.h = [h.clone() for h in st[3][1]]


def kill_everywhere(net, ctxs, l: int, k: int):
    """Global retirement from the union: shared field + every readout."""
    kill_unit(net, l, k)                        # sigma=0, h sentinel, column 0
    if l == 1:
        for ctx in ctxs:
            ctx.wout[:, k] = 0.0
