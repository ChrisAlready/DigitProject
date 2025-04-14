import numpy as np

class SimpleNeuralNetwork:

    def __init__(self, input_size = 784, hidden_size = 128, output_size = 10):

        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def summary(self):

        print("w1 shape: ", self.w1.shape)
        print("b1 shape: ", self.b1.shape)
        print("w2 shape: ", self.w2.shape)
        print("b2 shape: ", self.b2.shape)

    def relu(self,x):

        return np.maximum(0, x)

    def softmax(self, x):

        exp_x = np.exp(x - np.max(x, axis = 1, keepdims = True))
        return exp_x / np.sum(exp_x, axis = 1, keepdims = True)

    def forward(self, x):

        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1,self.w2) + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2

    def compute_loss(y_true, y_pred):

        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss

    if __name__ == "__main__":

        y_pred = np.array([
            [0.1, 0.05, 0.05, 0.7, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01]
        ])

        y_true = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        ])

        loss = compute_loss(y_true, y_pred)

        print("交叉熵损失： ", loss)