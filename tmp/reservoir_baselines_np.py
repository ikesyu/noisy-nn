"""Non-polynomial task probe (docs/idea_reservoir.md §13.3): does a fairly-sized
NG-RC still dominate when the target is NOT a polynomial of exact lags?

Three delayed non-polynomial targets (sin of a lag-sum, abs of a lag-diff, tanh of
a long window-mean) compared across Ours(B-mix) / LMU(A) / NG-RC (delay 20, 30) /
ESN, all on the same iid drive. Result (3 seeds): a delay-sufficient NG-RC wins on
every task; Ours only beats an UNDER-delayed NG-RC on the window task. Evidence
that the performance-superiority claim (i) does not survive a fair NG-RC baseline
on these low-dim temporal tasks -> the claim moves to the unique properties (ii).
"""
import numpy as np, reservoir as R
from reservoir.baselines import NGRC, budget_ngrc, budget_ours
from reservoir.metrics import task_nrmse

def masks(T,w=300,f=0.7):
    idx=np.arange(w,T);n=int(len(idx)*f)
    tr=np.zeros(T,bool);tr[idx[:n]]=True;te=np.zeros(T,bool);te[idx[n:]]=True;return tr,te

T=3000; x=15; W=8
def lag(u,k):
    z=np.zeros(len(u)); z[k:]=u[:-k]; return z
def winmean(u,k,w):
    z=np.zeros(len(u))
    for t in range(k+w,len(u)): z[t]=u[t-k-w:t-k].mean()
    return z

def build(u):
    return {
     'sin(pi(u_{t-15}+u_{t-1}))': np.sin(np.pi*(lag(u,x)+lag(u,1))),
     'abs(u_{t-15}-u_{t-1})':     np.abs(lag(u,x)-lag(u,1)),
     'tanh(6(mean8 u_{t-15}-.25))': np.tanh(6*(winmean(u,x,W)-0.25)),
    }

seeds=3
agg={}
for sd in range(seeds):
    rng=np.random.default_rng(sd); u=rng.uniform(0,0.5,T)
    tr,te=masks(T); A=R.LDNField(H=48,theta=60.0).run(u)
    for name,y in build(u).items():
        y=y-y.mean()
        r=agg.setdefault(name,{k:[] for k in ('ours','lmu','ng20','ng30','esn')})
        r['ours'].append(R.NoiseModulatedMap(A,y,tr,Ho=48,mix=True,seed=1).eval(te,200))
        r['lmu'].append(R.LearnedCrossingMap(A,y,tr,Ho=48,seed=1).eval(te,200))
        r['ng20'].append(task_nrmse(NGRC(20,2).features(u),y,alpha=1e-2))
        r['ng30'].append(task_nrmse(NGRC(30,2).features(u),y,alpha=1e-2))
        r['esn'].append(task_nrmse(R.LeakyESN(H=200,seed=1).run(u),y,alpha=1e-2))
    print(f'seed {sd} done',flush=True)

print(f"\n{'task':32s} {'Ours(B)':>9s} {'LMU(A)':>9s} {'NGRCd20':>9s} {'NGRCd30':>9s} {'ESN200':>9s}")
print(f"{'params->':32s} {budget_ours(48,48)['trainable']:>9d} {'~2401':>9s} "
      f"{budget_ngrc(20)['trainable']:>9d} {budget_ngrc(30)['trainable']:>9d} {'201':>9s}")
for name,r in agg.items():
    print(f"{name:32s} "+ " ".join(f"{np.mean(r[k]):9.3f}" for k in ('ours','lmu','ng20','ng30','esn')))
