"""(b) v3 confirmation: 5 seeds on the key cells, plus the shape diagnostic that
explains WHY the optimum is interior (mixture beats BOTH pure readings)."""
import numpy as np, torch
from reservoir_lambda_local import masks, DelayField
from reservoir_lambda_local2 import local_task
from reservoir_lambda_local3 import train_eval

def shape(lam):
    Q = np.linspace(1e-4, 1-1e-4, 20001)
    z = (1-lam)*Q + lam*2*Q*(1-Q)
    peak = z.max(); plateau = (1-lam)*1.0
    return peak, plateau, peak - plateau

def main():
    T=3000; u,y=local_task(T,lag=2); tr,te=masks(T)
    X1 = DelayField(H=6).run(u)[:, 2:3]
    print("=== shape of zbar_lambda: 'bump on a step' ===")
    print("  lam   peak   plateau(s->+inf)   lobe depth")
    for lam in (0.0,0.2,0.4,0.6,0.8,1.0):
        p,pl,d = shape(lam)
        print(f"  {lam:4.1f}  {p:5.3f}   {pl:5.3f}            {d:+.3f}")
    print("\n=== 5-seed confirmation, gain init 1.0, depth 1, 1-D input ===")
    print("  H\\lam   0.00    0.60    1.00   | tanh")
    for H in (8, 16):
        row=[]
        for lam in (0.0,0.6,1.0):
            e=[train_eval(X1,y,tr,te,H,lam,gain0=1.0,seed=s) for s in range(5)]
            row.append((np.mean(e),np.std(e)))
        t=[train_eval(X1,y,tr,te,H,0.0,act="tanh",gain0=1.0,seed=s) for s in range(5)]
        print(f"  {H:<3d} " + "  ".join(f"{m:.2f}±{sd:.2f}" for m,sd in row)
              + f" | {np.mean(t):.2f}±{np.std(t):.2f}", flush=True)

if __name__=="__main__":
    main()
