import sys; sys.path.append('tmp')
import torch, numpy as np
from rl.a2c_swingup import train_a2c, eval_from_bottom
policy, env, norm, critic, cks, hist = train_a2c(
    seed=0, H=128, updates=400, episodes_per_update=3, horizon=400, gamma=0.99, lam=0.95,
    lr_actor=0.01, lr_critic=1e-3, critic_epochs=8, bottom_frac=0.5, force_mag=20.0,
    x_threshold=4.0, sigma_explore=0.4, sigma_explore_end=0.1,
    checkpoint_every=25, verbose=True)
torch.save({'checkpoints': cks, 'hist': hist}, 'tmp/out/swingup_a2c.pt')
print('=== eval from BOTTOM (mean cos / frac_up / last100_up) ===')
for upd, st in cks:
    mc, fu, tail = eval_from_bottom(st, horizon=500)
    print(f'  upd {upd:4d}  mean cos {mc:+.3f}  frac_up {fu:.3f}  last100_up {tail:.3f}')
