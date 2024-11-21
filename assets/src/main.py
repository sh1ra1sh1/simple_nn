from neural_network import NeuralNetwork
from utils import load_data, plot_losses, plot_decision_boundary

# ハイパーパラメータ
layer_sizes = [2, 20, 10, 1]  # 入力層、隠れ層1、隠れ層2、出力層
learning_rate = 0.01
epochs = 12500 + 1

# データの読み込み
X, y = load_data('../data/data.csv')

# ニューラルネットワークの初期化
nn = NeuralNetwork(layer_sizes, learning_rate)

# ニューラルネットワークの学習
losses = nn.train(X, y, epochs)

# 学習曲線のプロット
plot_losses(losses)

# 決定境界のプロット
plot_decision_boundary(X, y, nn)
