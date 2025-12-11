import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import erf, sqrt

# ノイズの標準偏差
sigma = 0.5

# 図とサブプロットの作成（横に3つ並べる）
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# ---- 左のグラフ：ノイズ付き正弦波 ----
ax_left = axs[0]
left_line1, = ax_left.plot([], [], lw=2, label="Noisy Sine", color='blue')
left_line2, = ax_left.plot([], [], lw=2, label="Original Sine", color='red')
ax_left.set_xlim(0, 100)
ax_left.set_ylim(-2, 2)
ax_left.grid(True)
ax_left.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)


# ---- 中央のグラフ：二値化（0を閾値）した結果 ----
ax_mid = axs[1]
mid_line, = ax_mid.plot([], [], lw=2, color='green')
ax_mid.set_xlim(0, 100)
ax_mid.set_ylim(-0.5, 1.5)  # 0 と 1 の値なので余裕を持った表示に
ax_mid.grid(True)
ax_mid.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

# ---- 右のグラフ：元の正弦波と理論上の確率 P(y>0) ----
ax_right = axs[2]
right_line1, = ax_right.plot([], [], lw=2, label="Original Sine", color='red')
right_line2, = ax_right.plot([], [], lw=2, label="Estimated", color='purple')
ax_right.set_xlim(0, 100)
ax_right.set_ylim(-2, 2)
ax_right.grid(True)
ax_right.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

# 各グラフのデータを格納するリスト
xdata = []
left_ydata1 = []
left_ydata2 = []
mid_ydata = []
right_ydata1 = []  # 元の正弦波
right_ydata2 = []  # 理論上の確率

# 初期化関数（アニメーション開始前の状態を設定）
def init():
    left_line1.set_data([], [])
    left_line2.set_data([], [])
    mid_line.set_data([], [])
    right_line1.set_data([], [])
    right_line2.set_data([], [])
    return left_line1, left_line2, mid_line, right_line1, right_line2

# 更新関数：各フレームごとに呼ばれる
def update(frame):
    # 新たな x 軸の値を追加（例：フレーム番号をそのまま x 軸の値として使用）
    xdata.append(frame)
    
    # 元の正弦波の値を計算
    sine_val = np.sin(frame / 10.0)
    left_ydata2.append(sine_val)
    
    # 左のグラフ：ノイズを加えた正弦波
    noisy_val = sine_val + np.random.normal(scale=sigma)
    left_ydata1.append(noisy_val)
    
    # 中央のグラフ：0 を閾値に二値化（noisy_val >= 0 なら 1，そうでなければ 0）
    bin_val = 1 if noisy_val >= 0 else 0
    mid_ydata.append(bin_val)
    
    # 右のグラフ：
    # 1) ノイズを加える前の正弦波
    right_ydata1.append(sine_val)
    # 2) 理論上、ノイズを加えた場合に y>0 となる確率をもとにスケーリング
    #prob_val = 0.5 * (1 + erf(sine_val / (sigma * sqrt(2))))
    prob_val = erf(sine_val / (sigma * sqrt(2)))
    right_ydata2.append(prob_val)
    
    # 各ラインのデータ更新
    left_line1.set_data(xdata, left_ydata1)
    left_line2.set_data(xdata, left_ydata2)
    mid_line.set_data(xdata, mid_ydata)
    right_line1.set_data(xdata, right_ydata1)
    right_line2.set_data(xdata, right_ydata2)
    
    # x 軸のウィンドウ幅が 100 になるように、描画が右端に達したら左へスクロール
    if frame > 100:
        for ax in axs:
            ax.set_xlim(frame - 100, frame)
    
    return left_line1, left_line2, mid_line, right_line1, right_line2

# FuncAnimation によりアニメーション作成（frames: 0～600 を 1200 分割）
ani = animation.FuncAnimation(
    fig, update,
    frames=np.linspace(0, 600, 1200),
    init_func=init,
    blit=True,
    interval=50,
    repeat=False
)

#ani.save("anim.gif",writer='imagemagick')
#ani.save("anim.mp4",writer='ffmpeg')
plt.show()