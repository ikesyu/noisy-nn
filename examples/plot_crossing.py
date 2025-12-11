import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# データ生成用関数
def gen(x, s):
    y1 = 0*x
    y2 = 0*x
    if s != 0:
        P=norm.cdf(x , loc =0, scale=s)
        p= norm.pdf(x, loc=0, scale=s)
        y1 = 4*P*(1-P)
        y2 = 4*(1-2*P)*p
    return y1, y2

x = np.linspace(-2, 2, 100)
y1, y2 = gen(x, 1.0)

# 横並びの2つのグラフを作成
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

for i in np.linspace(0, 1.0, 10):
    y1, y2 = gen(x, i)
    axes[0].plot(x, y1, label=f"s={i:.1f}")
    axes[1].plot(x, y2, label=f"s={i:.1f}")

#fig.legend(loc="upper center", ncol=5)

plt.tight_layout()
plt.show()