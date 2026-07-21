import sys; sys.path.append('tmp')
import torch, numpy as np
from rl.swingup import train_swingup, eval_from_bottom
policy, env, norm, checkpoints, hist = train_swingup(
    seed=0, H=128, episodes=900, horizon=300, gamma=0.99, lr_actor=0.03,
    bottom_frac=0.5, force_mag=20.0, x_threshold=4.0, checkpoint_every=100, verbose=True)
torch.save({'checkpoints': checkpoints, 'ep_returns': hist,
            'force_mag': 20.0, 'x_threshold': 4.0}, 'tmp/out/swingup_run.pt')
print('=== eval from BOTTOM (swing-up) ===')
for ep, st in checkpoints:
    mc, fu = eval_from_bottom(st, force_mag=20.0, x_threshold=4.0)
    print(f'  ep {ep:5d}  mean cos {mc:+.3f}  frac_upright {fu:.3f}')
