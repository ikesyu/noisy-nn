import sys; sys.path.append('tmp')
import torch, numpy as np
from rl.a2c_swingup import train_a2c, eval_from_bottom, build_policy
from rl.train import RunningNorm
from rl.envs_swingup import CartPoleSwingUp
from rl import field as F
d=torch.load('tmp/out/consolidate_p1.pt', weights_only=False)
H=128; half=H//2
# fixed norm = phase-1 stats (so frozen A sees the same normalization)
norm=RunningNorm(5); norm.mean=d['norm_mean'].clone(); norm.M2=(d['norm_std']**2).clone(); norm.count=torch.tensor(1.0)
# freeze subnetwork A ([0:64]): its layer-0 inputs, its columns into layer-1, its readout cols
m={}
w0=torch.zeros(H,5); w0[:half,:]=1; m[(0,'weight')]=w0
b0=torch.zeros(H); b0[:half]=1; m[(0,'bias')]=b0
w1=torch.zeros(H,H); w1[:,:half]=1; m[(1,'weight')]=w1
w2=torch.zeros(1,H); w2[:,:half]=1; m[(2,'weight')]=w2
P_pump=F.recruit(H,0.6,1,n_layers=2)  # subnetwork B ([64:128])
P_bal =F.recruit(H,0.6,0,n_layers=2)  # subnetwork A ([0:64], frozen)
pol,_,_,critic,cks,hist=train_a2c(
    seed=0,H=H,updates=400,episodes_per_update=3,horizon=400,gamma=0.99,lam=0.95,
    lr_actor=0.01,lr_critic=1e-3,critic_epochs=8,force_mag=20.0,x_threshold=4.0,
    sigma_explore=0.4,sigma_explore_end=0.1,fields=[P_pump,P_bal],gate_k=6.0,gate_c=0.0,
    init_body=d['body'],freeze_mask=m,n_hidden_layers=2,energy_reward=True,
    norm_obj=norm,update_norm=False,checkpoint_every=50,verbose=True)
torch.save({'checkpoints':cks,'hist':hist},'tmp/out/consolidate_p2.pt')
# verify A frozen (weights unchanged)
a_unchanged = torch.allclose(pol.net.fcs[0].weight[:half], d['body']['fcs.0.weight'][:half])
print('A layer-0 weights unchanged (frozen):', bool(a_unchanged))
print('=== eval swing-up from BOTTOM ===')
for upd,st in cks:
    mc,fu,tail=eval_from_bottom(st,horizon=500)
    print(f'  upd {upd:4d}  mean cos {mc:+.3f}  frac_up {fu:.3f}  last100_up {tail:.3f}')
