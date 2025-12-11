import matplotlib.pyplot as plt
import numpy as np

def raster_plot(binary_matrix, line_height=0.8):
    """
    Rasterプロットを描画する関数
    
    Args:
        binary_matrix (2D array-like): shape = (n_neurons, n_timepoints)
            0と1からなる2次元配列。行がニューロン、列が時間。
        line_height (float): 各スパイクの縦線の長さ（デフォルト0.8）
    """
    binary_matrix = np.array(binary_matrix)
    n_neurons, n_timepoints = binary_matrix.shape

    plt.figure(figsize=(10, n_neurons * 0.25))

    for neuron_idx in range(n_neurons):
        spike_times = np.where(binary_matrix[neuron_idx] == 1)[0]
        for t in spike_times:
            plt.vlines(t, neuron_idx + 1 - line_height/2, neuron_idx + 1 + line_height/2, color="black")

    plt.ylim(0.5, n_neurons + 0.5)
    plt.yticks(np.arange(1, n_neurons + 1), labels=np.arange(1, n_neurons + 1))
    plt.xlabel("Time")
    plt.ylabel("Neuron")
    plt.show()


# サンプルデータ（ニューロン20 x 時間200、ランダムにスパイク発生）
np.random.seed(0)
sample_data = (np.random.rand(20, 200) < 0.05).astype(int)  # スパイク確率5%

raster_plot(sample_data)