import sys; sys.path.append('tmp')
import torch, numpy as np
from rl.a2c_swingup import train_a2c
from rl.envs_swingup import CartPoleSwingUp
from rl import field as F
H=128; sigma=0.6
P_bal = F.recruit(H, sigma, 0, n_layers=2)   # subnetwork A (units [0:64]) in both hidden layers
pol,_,norm,critic,cks,hist = train_a2c(
    seed=0, H=H, updates=200, episodes_per_update=3, horizon=300, gamma=0.99, lam=0.95,
    lr_actor=0.01, lr_critic=1e-3, critic_epochs=8, force_mag=20.0, x_threshold=4.0,
    sigma_explore=0.3, sigma_explore_end=0.1, fixed_field=P_bal,
    start_center=0.0, start_range=0.5, n_hidden_layers=2, energy_reward=False,
    checkpoint_every=200, verbose=True)
std = torch.sqrt(norm.M2/norm.count+norm.eps)
torch.save({'body': pol.net.state_dict(), 'norm_mean': norm.mean, 'norm_std': std,
            'H': H, 'force_mag': 20.0}, 'tmp/out/consolidate_p1.pt')
pol.field = P_bal; rng=np.random.default_rng(7); fr=[]
for s in range(6):
    env=CartPoleSwingUp(horizon=400, random_start=False, seed=s, force_mag=20.0, x_threshold=4.0, continuous=True)
    obs,_=env.reset(seed=s, start_theta=float(rng.uniform(-0.3,0.3))); cs=[]
    for _ in range(400):
        on=torch.clamp((torch.tensor(obs,dtype=torch.float32)-norm.mean)/std,-5,5)
        step=pol.rollout_step(on.unsqueeze(0),greedy=True); obs,r,te,tr,_=env.step(float(step.action.item())); cs.append(env.cos_theta())
    fr.append((np.mean(cs),(np.array(cs)>0.9).mean()))
print('Phase1 BALANCE eval (2L, pure cos): mean cos', round(np.mean([m for m,_ in fr]),3), 'frac_up', round(np.mean([f for _,f in fr]),3))
