"""Does raising numT recover the crossing's SHAPE advantage on NARMA?

Noise-matched run (reservoir_noisematch.py) showed on NARMA-10 depth 1:
    analytic crossing 0.217  <<  tanh 0.371 ~ analytic threshold 0.377
but
    sample  crossing 0.355  ~=  tanh          (sampling cost +0.138)
i.e. the crossing has the best SHAPE here, and numT=64 estimation variance eats
the advantage.  Prediction: the sample curve should approach the analytic 0.217
as numT grows.  (Contrast: on parity d2 and absum the sampling cost is NEGATIVE
-- the MC noise HELPS -- consistent with sec13.1's noise-as-resource.)
"""
import numpy as np, torch, torch.nn as nn
from reservoir.tasks import narma_x
from reservoir.fields import LDNField
from reservoir.readout import standardize_fit, nrmse
from reservoir_noisematch import Net, run, ACTS

T=3000; S=3
u,y=narma_x(T,10,seed=0); y=y-y.mean()
from reservoir_lambda_local import masks
tr,te=masks(T); A=LDNField(H=48,theta=60.0).run(u)

def run_numT(act,g,numT,seed):
    mu,sd=standardize_fit(A[tr]); X=torch.tensor((A-mu)/sd,dtype=torch.float32)
    yt=torch.tensor(y,dtype=torch.float32); ti=np.where(tr)[0]; ei=np.where(te)[0]
    rng=np.random.default_rng(seed); torch.manual_seed(seed)
    net=Net(A.shape[1],48,1,act,g,numT=numT,seed=seed)
    opt=torch.optim.Adam(net.parameters(),lr=3e-3,weight_decay=1e-4)
    for _ in range(900):
        b=torch.tensor(rng.choice(ti,size=256,replace=False))
        net.train(); opt.zero_grad(); ((net(X[b])-yt[b])**2).mean().backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p=[net(X[torch.tensor(ei[i:i+1000])]).numpy() for i in range(0,len(ei),1000)]
    return nrmse(y[ei],np.concatenate(p))

print("NARMA-10 + LDN, depth 1: does numT recover the crossing's shape advantage?")
print("  analytic reference: crossing 0.217 / threshold 0.377 / tanh 0.371")
print(f"  {'numT':>6} {'cross_smp':>10} {'thr_smp':>9}")
for numT in (32, 64, 128, 256, 512):
    c=np.mean([run_numT("cross_smp",0.3,numT,s) for s in range(S)])
    t=np.mean([run_numT("thr_smp",0.3,numT,s) for s in range(S)])
    print(f"  {numT:6d} {c:10.3f} {t:9.3f}", flush=True)
