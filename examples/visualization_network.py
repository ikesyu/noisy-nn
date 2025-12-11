import matplotlib.pyplot as plt
import numpy as np


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

# ニューラルネットワークの層構造を指定
layer_structure = [4, 6, 4, 2]  # 入力層4, 隠れ層6, 隠れ層4, 出力層2

# 図の作成
fig, ax = plt.subplots(figsize=(6, 6))
neuron_patches, edge_lines = draw_neural_network(ax, layer_structure)
plt.show()

# `neuron_patches` を使うことで色の変更やアニメーションが容易に可能になる
