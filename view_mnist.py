import numpy as np
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


plt.imshow(x_train[0], cmap = 'grey')
plt.title(f"Label: {y_train[0]}")
plt.show()