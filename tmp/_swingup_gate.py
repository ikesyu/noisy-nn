import sys; sys.path.append('tmp')
import torch, numpy as np
from rl.gate_swingup import train_gate, eval_gate
from rl import field as F
P=[F.recruit(128,0.6,0,n_layers=2,quiet=0.3), F.recruit(128,0.6,1,n_layers=2,quiet=0.3)]
body,gate,critic,norm,cks,hist=train_gate(
    seed=0,H=128,Hg=48,updates=450,episodes_per_update=3,horizon=400,gamma=0.99,lam=0.95,
    lr_actor=0.01,lr_gate=0.01,lr_critic=1e-3,critic_epochs=8,bottom_frac=0.5,force_mag=20.0,
    x_threshold=4.0,sigma_explore=0.4,sigma_explore_end=0.1,gate_force=5.0,gate_sigma=0.4,
    gate_sigma_end=0.15,fields=P,checkpoint_every=25,verbose=True)
torch.save({'checkpoints':cks,'hist':hist},'tmp/out/swingup_gate.pt')
print('=== eval from BOTTOM + learned gate g(cos) ===')
for upd,st in cks:
    (mc,fu,tail),gc=eval_gate(st,horizon=500,return_gate=True)
    gc=np.array(gc); lo=gc[gc[:,0]<-0.5][:,1]; hi=gc[gc[:,0]>0.5][:,1]
    glo=round(lo.mean(),2) if len(lo) else float('nan'); ghi=round(hi.mean(),2) if len(hi) else float('nan')
    print(f'  upd {upd:4d}  mean cos {mc:+.3f}  last100_up {tail:.3f}  | gate g(pump reg)={glo} g(balance reg)={ghi}')
