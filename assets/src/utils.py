import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# データをCSVから読み込み、標準化
def load_data(filepath):
    df = pd.read_csv(filepath, header=None)
    temp_df = df.to_numpy().reshape([152, 3])
    data = pd.DataFrame(temp_df, columns=['x1', 'x2', 'y'])

    X = data[['x1', 'x2']].values  # 入力特徴量
    y = data['y'].values.reshape(-1, 1)  # ターゲット変数

    # データを標準化
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y

# 学習曲線をプロット
def plot_losses(losses):
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_curve.png')  # 学習曲線を保存
    plt.show()

# 決定境界をプロット
def plot_decision_boundary(X, y, nn):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # 予測結果を整形してプロット
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.2, colors=['red', 'blue'])
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolors='k', marker='o')
    plt.title('Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig('decision_boundary.png')  # 決定境界を保存
    plt.show()
