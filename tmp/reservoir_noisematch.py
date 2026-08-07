"""Separate ACTIVATION SHAPE from SAMPLING COST.

User's point (correct): threshold and crossing are both SAMPLE-level (MC over
numT), so they pay a 1/sqrt(numT) estimation-variance penalty, while tanh is
deterministic.  Any verdict resting on a tanh comparison therefore conflates
shape with sampling cost.  sec13.1 caveat 1 says the same thing.

  - threshold vs crossing  : both sample, same numT -> VALID as-is.
  - anything vs tanh       : confounded.  Fix by ALSO running the mean-field
                             (analytic) versions of threshold and crossing, which
                             are deterministic like tanh.

analytic threshold = Phi(p) ; analytic crossing = 2 Phi(p)(1-Phi(p)).
The analytic-vs-sample gap of the SAME activation measures the sampling cost.
"""
import numpy as np
import torch
import torch.nn as nn
from reservoir.tasks import narma_x
from reservoir.fields import LDNField
from reservoir.readout import standardize_fit, nrmse
from reservoir.moment import masks, LambdaAct, SignField, parity_task, AnaThr, AnaCross

class Net(nn.Module):
    def __init__(self, Hin, H, depth, act, gam, numT=64, seed=0):
        super().__init__(); torch.manual_seed(seed)
        d=[Hin]+[H]*depth
        self.ls=nn.ModuleList([nn.Linear(d[i],d[i+1]) for i in range(depth)])
        self.bns=nn.ModuleList([nn.BatchNorm1d(d[i+1],affine=False) for i in range(depth)])
        self.cs=nn.ParameterList([nn.Parameter(torch.zeros(d[i+1])) for i in range(depth)])
        self.out=nn.Linear(d[-1],1); self.act,self.g,self.numT=act,gam,numT
    def _f(self,p):
        a=self.act
        if a=="tanh": return torch.tanh(p)
        if a=="thr_ana": return AnaThr.apply(p)
        if a=="cross_ana": return AnaCross.apply(p)
        return LambdaAct.apply(p, 0.0 if a=="thr_smp" else 1.0, 0.0, self.numT)
    def forward(self,x):
        for L,bn,c in zip(self.ls,self.bns,self.cs): x=self._f((bn(L(x))-c)*self.g)
        return self.out(x).squeeze(-1)

def run(A,y,tr,te,depth,act,g,H=48,steps=900,bs=256,lr=3e-3,seed=0):
    mu,sd=standardize_fit(A[tr]); X=torch.tensor((A-mu)/sd,dtype=torch.float32)
    yt=torch.tensor(y,dtype=torch.float32)
    ti=np.where(tr)[0]; ei=np.where(te)[0]
    rng=np.random.default_rng(seed); torch.manual_seed(seed)
    net=Net(A.shape[1],H,depth,act,g,seed=seed)
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-4)
    for _ in range(steps):
        b=torch.tensor(rng.choice(ti,size=bs,replace=False))
        net.train(); opt.zero_grad(); ((net(X[b])-yt[b])**2).mean().backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p=[net(X[torch.tensor(ei[i:i+1000])]).numpy() for i in range(0,len(ei),1000)]
    return nrmse(y[ei],np.concatenate(p))

ACTS=("thr_smp","thr_ana","cross_smp","cross_ana","tanh")
G=(0.3,0.5,0.7,1.0,1.5,2.0,3.0); S=3; T=3000
def bench(name,A,y,tr,te,depth):
    best={}
    for a in ACTS:
        v=[np.mean([run(A,y,tr,te,depth,a,g,seed=s) for s in range(S)]) for g in G]
        i=int(np.argmin(v)); best[a]=(v[i],G[i])
    print(f"  {name:<16}"+"  ".join(f"{a}={best[a][0]:.3f}(g{best[a][1]:g})" for a in ACTS))
    cs,ca=best["cross_smp"][0],best["cross_ana"][0]
    ts,ta=best["thr_smp"][0],best["thr_ana"][0]
    print(f"      sampling cost: threshold {ts-ta:+.3f} | crossing {cs-ca:+.3f}"
          f"   || noise-matched vs tanh: thr_ana-tanh {ta-best['tanh'][0]:+.3f}, "
          f"cross_ana-tanh {ca-best['tanh'][0]:+.3f}", flush=True)

def main():
    tr,te=masks(T)
    u,y=parity_task(T); Ap=SignField(H=32,gain=8.0).run(u)
    print("PARITY (sign field)"); bench("depth1",Ap,y,tr,te,1); bench("depth2",Ap,y,tr,te,2)
    rng=np.random.default_rng(0); u2=rng.uniform(0,0.5,T)
    lag=lambda L: np.concatenate([np.zeros(L),u2[:T-L]])
    ya=np.abs(lag(2)-0.25)+np.abs(lag(5)-0.25); ya=ya-ya.mean()
    Aa=LDNField(H=48,theta=60.0).run(u2)
    print("ABSUM (LDN)"); bench("depth1",Aa,ya,tr,te,1)
    un,yn=narma_x(T,10,seed=0); yn=yn-yn.mean(); An=LDNField(H=48,theta=60.0).run(un)
    print("NARMA-10 (LDN)"); bench("depth1",An,yn,tr,te,1)

if __name__ == "__main__":   # guard added so `from reservoir_noisematch import ...`
    main()                   # (reservoir_numT.py) no longer re-runs the benchmark
