import numpy as np

# ニューラルネットワークの定義
class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # 重みとバイアスをランダムに初期化
        np.random.seed(0)
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.biases.append(np.zeros(layer_sizes[i+1]))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        activations = [X]
        inputs = []
        
        # 順伝播
        for W, b in zip(self.weights, self.biases):
            inputs.append(np.dot(activations[-1], W) + b)
            if len(inputs) < len(self.weights):
                activations.append(self.relu(inputs[-1]))
            else:
                activations.append(self.sigmoid(inputs[-1]))
        
        return inputs, activations
    
    def backward(self, X, y, inputs, activations):
        layer_deltas = []
        error = activations[-1] - y
        layer_deltas.append(error * self.sigmoid_derivative(activations[-1]))
        
        # 逆伝播
        for i in range(len(self.layer_sizes) - 2, 0, -1):
            error = layer_deltas[-1].dot(self.weights[i].T)
            layer_deltas.append(error * self.relu_derivative(activations[i]))
        
        layer_deltas.reverse()
        
        # 勾配降下法で重みとバイアスを更新
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * activations[i].T.dot(layer_deltas[i])
            self.biases[i] -= self.learning_rate * np.sum(layer_deltas[i], axis=0)
    
    def train(self, X, y, epochs):
        losses = []
        
        # トレーニングループ
        for epoch in range(epochs):
            inputs, activations = self.forward(X)
            self.backward(X, y, inputs, activations)
            
            loss = np.mean(np.square(y - activations[-1]))
            losses.append(loss)
            
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
        
        return losses
    
    def predict(self, X):
        _, activations = self.forward(X)
        return (activations[-1] > 0.5).astype(int)
