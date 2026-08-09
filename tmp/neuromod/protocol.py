"""How the benchmark is trained and scored.

This is the experimental protocol of this problem, not NNN machinery: the model
itself comes from `nnn.model`, and nothing here knows about forward-covariance
credit, RL, consolidation, or reservoirs.  A challenge that wants a different
learning rule should build its own loop and keep using `world` and `fields`.

The protocol: same observations, same weights, different noise fields, different
targets.  Each epoch presents all three states in random order and takes one Adam
step per state, so the shared weights must hold every behaviour at once.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from nnn import model


def build_network(hidden_dim: int, n_hidden: int = 2, base_std: float = 0.8,
                  kind: str = "analytic", t: int = 64,
                  crossing_h: float = 0.2, in_dim: int = 6, out_dim: int = 2):
    """This benchmark's network shape: [6, hidden_dim * n_hidden, 2] from `nnn.model`.

    Only the structure convention belongs to this problem; the model itself is
    plain NNN.

    `kind` selects the implementation level, and the choice is not cosmetic:

        "analytic"  closed-form expected response, no threshold.  The mean field.
                    Fast, and the right choice for addressing figures, but it has
                    NO subthreshold barrier, so stochastic resonance does not
                    exist in it (`docs/idea_core.md` section 2.6).
        "sample"    real noise injection with a crossing threshold h > 0.  The
                    mechanism, and the only level at which SR appears.

    `n_hidden = 1` is worth knowing about: a zero-sigma unit is genuinely silent
    only in the first hidden layer, so single-layer models are where sigma-only
    recruitment gates exactly.  Lesion and multiplexing experiments should either
    use one hidden layer or apply the kill triple (`fields.kill_units`).
    """
    structure = [in_dim] + [hidden_dim] * n_hidden + [out_dim]
    if kind == "analytic":
        return model.SimpleNNNAnalytic(structure=structure, std=base_std)
    if kind == "sample":
        return model.SimpleNNNSample(structure=structure, std=base_std,
                                     h=crossing_h, t=t)
    raise ValueError(f"unknown model kind {kind!r}; use 'analytic' or 'sample'.")


def net_device(net) -> torch.device:
    """The device the model's parameters live on.

    Everything fed to the net -- observations, targets, noise fields -- has to
    be placed here.  The sample model runs T stochastic forward passes per
    input, which is ~65x the work of the analytic mean field, so it is only
    practical on a GPU; keeping the data placement in one helper is what makes
    the package movable.
    """
    return next(net.parameters()).device


def evaluate_vector_field(net, obs: torch.Tensor, field: torch.Tensor,
                          n_hidden: int) -> torch.Tensor:
    """Predicted velocities under one noise field, applied to every hidden layer."""
    dev = net_device(net)
    obs = obs.to(dev)
    field = field.to(dev)
    return net(obs, stds=[field] * n_hidden)


def train(net, obs: torch.Tensor, targets: dict[str, torch.Tensor],
          state_fields: dict[str, torch.Tensor], states, n_hidden: int,
          epochs: int, lr: float, chunk: int = 0, verbose: bool = True):
    """Supervised training of the shared weights across all states.

    `chunk > 0` takes Adam steps over random minibatches of that many points.
    Returns the per-state loss history.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    history = {s: [] for s in states}
    n = obs.shape[0]
    log_step = max(1, epochs // 20)

    for epoch in range(epochs):
        perm = torch.randperm(n) if chunk > 0 else None
        for state in np.random.permutation(list(states)):
            if chunk > 0:
                last = None
                for start in range(0, n, chunk):
                    idx = perm[start:start + chunk]
                    optimizer.zero_grad()
                    pred = evaluate_vector_field(net, obs[idx], state_fields[state],
                                                 n_hidden)
                    loss = criterion(pred, targets[state][idx].to(pred.device))
                    loss.backward()
                    optimizer.step()
                    last = loss.item()
                history[state].append(last)
            else:
                optimizer.zero_grad()
                pred = evaluate_vector_field(net, obs, state_fields[state], n_hidden)
                loss = criterion(pred, targets[state].to(pred.device))
                loss.backward()
                optimizer.step()
                history[state].append(loss.item())

        if verbose and (epoch % log_step == 0 or epoch == epochs - 1):
            msg = "  ".join(f"{s.split('_')[0]}={history[s][-1]:.4f}" for s in states)
            print(f"  epoch {epoch:5d}   {msg}")

    return history


def train_blended(net, pool_positions: np.ndarray, objects: dict, alphas: dict,
                  fields: dict, n_hidden: int, steps: int, bs: int = 512,
                  lr: float = 3e-4, seed: int = 0, gamma: float = 2.0,
                  wall_margin: float = 0.15, wall_kappa: float = 1.0,
                  randomize_threats: bool = True, randomize_food: bool = True,
                  verbose: bool = True):
    """Train on the RUNTIME distribution instead of the three pure corners.

    The closed loop never runs a pure state: `neuromod_weights` produces a
    convex blend of a goal field (food or shelter) with the threat field, and
    the threats WANDER.  Sec7.11 measured that a pure-trained network under
    blended fields is worse than a naive output mixer, which was tolerable for
    the near-linear 6D vector code but breaks the sector code (the nonlinear
    map does not factor across channels): symptoms were "ignores adjacent
    food", "walks into threats", "sheltering never triggers".  Each step here
    samples

        a position minibatch  x  a blend w on the runtime manifold
        (30% pure / 50% goal<->threat edge / 20% Dirichlet interior)
        x  (optionally) a fresh random threat placement in the wander box

    with the target computed from the same geometry rule.  `randomize_food`
    additionally drops each food with p=0.2 per step (sensing and target use
    the same mask): the loop DEPLETES food, and a net that never saw "the food
    I am standing on is gone" keeps the low deceleration-zone speed it learned
    there instead of accelerating to the next food (measured: direction fine
    at cos 0.96, magnitude anchored at 0.41 vs teacher 0.88 -- the post-eating
    stall).  NOTE for L4: nets trained this way have SEEN mixtures, so they
    cannot support the zero-shot interpolation claim of sec7.11; that claim
    rests on pure-trained nets.
    """
    from . import world
    from . import fields as fieldlib

    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    pool_n = pool_positions.shape[0]
    history = {"blended": []}
    log_step = max(1, steps // 20)

    for step in range(steps):
        u = rng.random()
        if u < 0.3:
            w = np.zeros(3, np.float32); w[rng.integers(3)] = 1.0
        elif u < 0.8:
            w = np.zeros(3, np.float32)
            g = 0 if rng.random() < 0.5 else 2
            a = rng.random()
            w[g] = 1.0 - a; w[1] = a
        else:
            w = rng.dirichlet((1.0, 1.0, 1.0)).astype(np.float32)

        objs_step = objects
        if randomize_threats:
            tpos = []
            while len(tpos) < objects["threat"].shape[0]:
                c = rng.uniform(-world.THREAT_BOUNDS, world.THREAT_BOUNDS, size=2)
                if all(np.linalg.norm(c - s) >= world.THREAT_KEEPOUT_RADIUS
                       for s in objects["shelter"]):
                    tpos.append(c)
            objs_step = {k: v.copy() for k, v in objects.items()}
            objs_step["threat"] = np.array(tpos, dtype=np.float32)

        fs = None
        if randomize_food:
            fs = (rng.random(objects["food"].shape[0]) > 0.2).astype(np.float32)

        alpha_w = world.blend_alpha(w, alphas)
        field_w = fieldlib.blend_fields(fields, w, world.CATEGORIES)
        idx = rng.choice(pool_n, size=bs, replace=False)
        pos_b = pool_positions[idx]
        dev = net_device(net)
        obs_b = torch.tensor(world.encode_observations(pos_b, objs_step,
                                                       food_strengths=fs),
                             dtype=torch.float32, device=dev)
        tgt_b = torch.tensor(world.make_behavior_targets(
                    pos_b, objs_step, alpha_w, gamma=gamma,
                    wall_margin=wall_margin, wall_kappa=wall_kappa,
                    food_strengths=fs),
                             dtype=torch.float32, device=dev)

        optimizer.zero_grad()
        loss = criterion(evaluate_vector_field(net, obs_b, field_w, n_hidden),
                         tgt_b)
        loss.backward()
        optimizer.step()
        history["blended"].append(loss.item())
        if verbose and (step % log_step == 0 or step == steps - 1):
            print(f"  step {step:6d}   blended={loss.item():.4f}")

    return history


def final_losses(net, obs: torch.Tensor, targets: dict[str, torch.Tensor],
                 state_fields: dict[str, torch.Tensor], states,
                 n_hidden: int) -> dict[str, float]:
    """Per-state MSE after training."""
    criterion = nn.MSELoss()
    out = {}
    with torch.no_grad():
        for state in states:
            pred = evaluate_vector_field(net, obs, state_fields[state], n_hidden)
            out[state] = float(criterion(pred, targets[state].to(pred.device)).item())
    return out


def capability(net, obs: torch.Tensor, targets: dict[str, torch.Tensor],
               fields: dict[str, torch.Tensor], categories, states,
               state_to_field: dict, n_hidden: int) -> tuple[float, float, float]:
    """The benchmark's three scores under a given set of noise fields.

        separation  mean pairwise ||y_i - y_j|| between the fields' output fields,
                    i.e. how differently the three fields drive the same input
        signal      mean stimulus-locked variation ||y - mean_obs(y)||, the part
                    of the output that actually depends on the input.  This is the
                    quantity that goes to zero at BOTH ends of a noise sweep:
                    subthreshold nothing crosses, oversaturated everything does
        task_err    mean over states of MSE against that state's trained target

    Returns them in that order.  Callers sweeping noise should report the
    gauge-invariant ratio h/sigma alongside any optimum, never sigma alone.
    """
    with torch.no_grad():
        y = {c: evaluate_vector_field(net, obs, fields[c], n_hidden).cpu().numpy()
             for c in categories}
    keys = list(categories)
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
    separation = float(np.mean([np.linalg.norm(y[a] - y[b], axis=1).mean()
                                for a, b in pairs]))
    signal = float(np.mean([
        np.linalg.norm(y[c] - y[c].mean(axis=0, keepdims=True), axis=1).mean()
        for c in keys]))
    task_err = float(np.mean([
        np.mean((y[state_to_field[s]] - targets[s].cpu().numpy()) ** 2)
        for s in states]))
    return separation, signal, task_err


def field_separation(net, obs: torch.Tensor, fields: dict[str, torch.Tensor],
                     categories, n_hidden: int) -> dict[str, float]:
    """Mean output distance between every pair of pure fields.

    If the trained network really uses the noise field to switch policy, these are
    clearly above zero; values near zero mean the field is being ignored.
    """
    with torch.no_grad():
        y = {key: evaluate_vector_field(net, obs, fields[key], n_hidden)
             for key in categories}
    out = {}
    keys = list(categories)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            out[f"{a}|{b}"] = float(torch.linalg.norm(y[a] - y[b], dim=1).mean().item())
    return out
