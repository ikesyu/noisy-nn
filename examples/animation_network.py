import torch
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import math

# ニューラルネットワークの層構造を指定
#layer_structure = [4, 6, 4, 2]  # 入力層4, 隠れ層6, 隠れ層4, 出力層2
layer_structure = [1, 20, 20, 1]  # 入力層4, 隠れ層6, 隠れ層4, 出力層2

# σの初期値（変更しやすいように変数化）
sigma = 1.0

# カラーマップの設定（0～σの値を色にマッピング）
cmap = plt.cm.viridis

def draw_neural_network(ax, layer_sizes, neuron_colors=None, fixed_size=(6, 6), neuron_radius=0.1, input_output_scale=0.6):
    """
    指定された構造のニューラルネットワークを、指定のサイズ内で均等に配置して描画する関数。
    
    :param ax: Matplotlib の Axes オブジェクト
    :param layer_sizes: 各層のニューロン数をリストで指定 [入力層, 隠れ層1, ..., 出力層]
    :param neuron_colors: 各ニューロンの色をリストで指定（省略時は青）
    :param fixed_size: 図のサイズ（幅, 高さ）
    :param neuron_radius: ニューロンの半径
    :param input_output_scale: 入力層と出力層の幅スケール（1未満で中間層より狭くする）
    :return: ニューロンとエッジのオブジェクトリスト（アニメーション用）
    """
    n_layers = len(layer_sizes)
    width, height = fixed_size
    x_spacing = width / (n_layers + 1)  # X軸方向の間隔
    max_layer_size = max(layer_sizes)
    y_spacing = height / (max_layer_size + 1)  # Y軸方向の間隔

    node_positions = []
    neuron_patches = []  # ニューロンの描画オブジェクト
    edge_lines = []  # エッジの描画オブジェクト
    
    # ニューロンの色設定（デフォルトは青）
    if neuron_colors is None:
        neuron_colors = ['blue'] * sum(layer_sizes)
    
    neuron_index = 0  # ニューロンのインデックス（色を適用するため）

    # 各層のノードを配置
    for i, layer_size in enumerate(layer_sizes):
        x_pos = (i + 1) * x_spacing  # X位置の計算

        # 入力層と出力層の幅を狭める（ただし1つしかない場合は中央）
        if layer_size == 1:
            y_positions = [height / 2]  # 1つだけの場合は中央に配置
        else:
            scale = input_output_scale if (i == 0 or i == n_layers - 1) else 1.0
            y_positions = np.linspace(
                (height * (1 - scale)) / 2 + y_spacing,
                height - (height * (1 - scale)) / 2 - y_spacing,
                layer_size
            )

        node_positions.append(list(zip([x_pos] * layer_size, y_positions)))

        # ノードの描画
        for x, y in node_positions[i]:
            circle = plt.Circle((x, y), radius=neuron_radius, color=neuron_colors[neuron_index], ec='black', lw=1.5, zorder=3)
            ax.add_patch(circle)
            neuron_patches.append(circle)
            neuron_index += 1

    # 層間の接続を描画
    for i in range(n_layers - 1):
        for (x1, y1) in node_positions[i]:
            for (x2, y2) in node_positions[i + 1]:
                line, = ax.plot([x1, x2], [y1, y2], 'k-', lw=0.5, alpha=0.6, zorder=2)
                edge_lines.append(line)

    # 軸の設定
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    return neuron_patches, edge_lines

#def generate_tensor_list(layer_sizes, sigma):
#    """ 各層のニューロンの活性度を表す PyTorch の 1次元テンソルのリストを生成 """
#    return [sigma * torch.rand(size=(size,)) for size in layer_sizes]

def generate_tensor_list(layer_sizes, sigma, delta=0.1):
    """
    各層のニューロンの活性度を表す1次元テンソルのリストを生成する関数

    パラメータ:
      layer_sizes (list of int): 各層のサイズのリスト（例: [1, 20, 20, 1]）
      sigma (float): 最初と最後の層のテンソルに用いるランダム生成のスケール
      delta (float): 呼び出しごとに進める角度の増分（デフォルトは0.1ラジアン）

    戻り値:
      list of torch.Tensor: 各層の1次元テンソル
         - 先頭層と最終層は sigma * torch.rand(...) によりランダム生成
         - 中間層は、初回は先頭半分が1, 後半が0となり、呼び出すたびに
           値が滑らかに変化し、最終的に先頭半分が0, 後半が1となる状態を経由し、
           再び元に戻る周期的な挙動を示す
    """
    # 関数属性を利用して状態（角度theta）を保持
    if not hasattr(generate_tensor_list, "theta"):
        generate_tensor_list.theta = 0.0  # 初期状態：theta = 0

    theta = generate_tensor_list.theta
    # 中間層の値を決める。θ=0のときは (1,0)、θ=πで (0,1) となる
    val_first_half = 0.5 * (1 + math.cos(theta))
    val_second_half = 0.5 * (1 - math.cos(theta))

    tensor_list = []
    for i, size in enumerate(layer_sizes):
        if i == 0 or i == len(layer_sizes) - 1:
            # 最初と最後はランダムなテンソル
            #tensor = sigma * torch.rand(size=(size,))
            tensor = torch.ones(1)
        else:
            # 中間層は指定した初期パターンから始まり、徐々に変化
            tensor = torch.empty(size)
            half = size // 2  # 奇数の場合は切り捨て
            # 最初の半分を val_first_half に，残りを val_second_half に設定
            tensor[:half] = val_first_half
            tensor[half:] = val_second_half
        tensor_list.append(tensor)

    # 呼び出すたびにthetaを更新（周期的に0～2πを循環）
    generate_tensor_list.theta = (theta + delta) % (2 * math.pi)
    return tensor_list


def tensor_to_colors(tensor_list, sigma):
    """ テンソルの値を 0～σ の範囲で正規化し、カラーマップに変換 """
    return [cmap(val.item() / sigma) for tensor in tensor_list for val in tensor]

# アニメーション用の更新関数
def update(frame):
    global neuron_patches, sigma

    # 新しいテンソルのリストを生成（周期的変化をシミュレート）
    tensor_list = generate_tensor_list(layer_structure, sigma)

    # テンソルをカラーマップの色に変換
    colors = tensor_to_colors(tensor_list, sigma)

    # 各ニューロンの色を更新
    for patch, color in zip(neuron_patches, colors):
        patch.set_facecolor(color)

    return neuron_patches

# 図の作成
fig, ax = plt.subplots(figsize=(6, 6))
neuron_patches, edge_lines = draw_neural_network(ax, layer_structure)

# アニメーションの作成
ani = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=False)

#ani.save("anim.mp4",writer='ffmpeg')
plt.show()
