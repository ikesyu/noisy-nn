import sys; sys.path.append('tmp')
import torch, numpy as np
from rl.train import train, Hypers
from rl.envs_swingup import CartPoleSwingUp
from rl.policy import CartPolePolicy

def eval_bottom(state, seeds=(0, 1, 2)):
    p = CartPolePolicy(obs_dim=5, hidden=int(state['hidden']), t=int(state['t']))
    p.load_state_dict(state['net']); p.eval()
    mean, std = state['norm_mean'], state['norm_std']
    fr = []
    for s in seeds:
        env = CartPoleSwingUp(horizon=400, random_start=False, seed=s)
        obs, _ = env.reset(seed=s); cs = []
        for _ in range(400):
            x = torch.clamp((torch.tensor(obs, dtype=torch.float32) - mean) / std, -5, 5)
            a = int(p.rollout_step(x.unsqueeze(0), greedy=True).action.item())
            obs, r, te, tr, _ = env.step(a); cs.append(env.cos_theta())
        fr.append((np.mean(cs), (np.array(cs) > 0.9).mean()))
    return np.mean([m for m, _ in fr]), np.mean([f for _, f in fr])

hp = Hypers(hidden=128, t=64, total_steps=240000, lr_actor=0.02, lr_critic=0.05,
            opt='sgd', gamma=0.99, lam=0.9)
res = train('cov_jac', seed=0, hp=hp,
            env_fn=lambda: CartPoleSwingUp(horizon=400, random_start=True),
            checkpoint_every=15000, verbose=False)
torch.save({'checkpoints': res.checkpoints, 'ep_returns': res.ep_returns,
            'ep_end_steps': res.ep_end_steps}, 'tmp/out/swingup_run.pt')
print("=== eval from BOTTOM (swing-up) per checkpoint ===")
for step, st in res.checkpoints:
    mc, fu = eval_bottom(st)
    print(f"  step {step:7d}  mean cos {mc:+.3f}  frac_upright {fu:.3f}")
