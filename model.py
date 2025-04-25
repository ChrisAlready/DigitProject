import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class SimpleNeuralNetwork:

    def __init__(self, input_size = 784, hidden_size = 128, output_size = 10):

        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

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

    def compute_loss(self, y_true, y_pred):

        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss

    def backward(self, x, y_true, learning_rate = 0.1):

        m = y_true.shape[0]

        dz2 = self.a2 - y_true
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis = 0, keepdims = True) / m

        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * (self.z1 > 0)
        dw1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis = 0, keepdims = True) / m

        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1

    def train(self, x_train, y_train, epochs = 10, learning_rate = 0.1):

        for epoch in range(epochs):

            y_pred = self.forward(x_train)

            loss = self.compute_loss(y_train, y_pred)

            self.backward(x_train, y_train, learning_rate)

            print(f"Epoch {epoch + 1} / {epochs}, Loss: {loss: 4f}")


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 784) / 255.0
    x_test = x_test.reshape(-1, 784) / 255.0
    y_train = to_categorical(y_train, num_classes = 10)
    y_test = to_categorical(y_test, num_classes = 10)

    nn = SimpleNeuralNetwork()
    nn.train(x_train, y_train, epochs = 50, learning_rate = 0.1)

    y_pred = nn.forward(x_test)
    predictions = np.argmax(y_pred, axis = 1)
    labels = np.argmax(y_test, axis = 1)
    accuracy = np.mean(predictions == labels)
    print("Test for accuracy: ", accuracy)