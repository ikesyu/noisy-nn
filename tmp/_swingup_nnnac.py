import sys; sys.path.append('tmp')
import torch, numpy as np
from rl.a2c_nnncritic import train_a2c_nnn
from rl.a2c_swingup import eval_from_bottom
pol,crit,norm,cks,hist=train_a2c_nnn(
    seed=0,H=128,Hc=64,updates=450,episodes_per_update=3,horizon=400,gamma=0.99,lam=0.95,
    lr_actor=0.01,lr_critic=0.02,bottom_frac=0.5,force_mag=20.0,x_threshold=4.0,
    sigma_explore=0.4,sigma_explore_end=0.1,checkpoint_every=25,verbose=True)
torch.save({'checkpoints':cks,'hist':hist},'tmp/out/swingup_nnnac.pt')
print('=== eval from BOTTOM (fully-NNN actor-critic, no backprop) ===')
for upd,st in cks:
    mc,fu,tail=eval_from_bottom(st,horizon=500)
    print(f'  upd {upd:4d}  mean cos {mc:+.3f}  frac_up {fu:.3f}  last100_up {tail:.3f}')
