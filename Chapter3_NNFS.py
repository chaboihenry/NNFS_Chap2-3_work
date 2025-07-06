import nnfs
nnfs.init()
import numpy as np
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from layer_dense import Layer_Dense

# Generate the spiral data as in the book
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
layer1 = Layer_Dense(2, 3)
# Perform a forward pass of our training data through this layer
layer1.forward(X)
# See the output of the first 5 samples
print(layer1.output[:5])

# Plot the data (as in the book)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()

