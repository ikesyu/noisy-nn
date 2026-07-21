import sys; sys.path.append('tmp')
import torch, numpy as np
from rl.a2c_swingup import train_a2c
from rl.envs_swingup import CartPoleSwingUp
from rl import field as F
def baleval(pol, field, norm):
    std=torch.sqrt(norm.M2/norm.count+norm.eps); pol.field=field; rng=np.random.default_rng(7); fr=[]
    for s in range(5):
        env=CartPoleSwingUp(horizon=400,random_start=False,seed=s,force_mag=20.0,x_threshold=4.0,continuous=True)
        obs,_=env.reset(seed=s,start_theta=float(rng.uniform(-0.25,0.25))); cs=[]
        for _ in range(400):
            on=torch.clamp((torch.tensor(obs,dtype=torch.float32)-norm.mean)/std,-5,5)
            st=pol.rollout_step(on.unsqueeze(0),greedy=True); obs,r,te,tr,_=env.step(float(st.action.item())); cs.append(env.cos_theta())
        fr.append((np.mean(cs),(np.array(cs)>0.9).mean()))
    return round(np.mean([m for m,_ in fr]),3), round(np.mean([f for _,f in fr]),3)
cfgs=[('1L uniform',1,None),('1L P_bal',1,F.recruit(128,0.6,0,n_layers=1)),('2L P_bal',2,F.recruit(128,0.6,0,n_layers=2))]
for name,nl,fld in cfgs:
    pol,_,norm,_,_,hist=train_a2c(seed=0,H=128,updates=80,episodes_per_update=3,horizon=300,lr_actor=0.01,force_mag=20.0,sigma_explore=0.3,sigma_explore_end=0.1,fixed_field=fld,start_center=0.0,start_range=0.4,n_hidden_layers=nl,checkpoint_every=0,verbose=False)
    mc,fu=baleval(pol,fld,norm)
    print(f'{name}: ret/step {np.mean(hist[-10:])/300:+.2f} | balance mean cos {mc} frac_up {fu}')
