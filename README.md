# Neural Networks from Scratch 

A comprehensive implementation of neural networks following Harrison Kinsley and Daniel Kukieła's "Neural Networks from Scratch" book (up to chapter 4).

## Overview

This repository contains code implementations and explanations for building neural networks from fundamental components, using both raw Python and NumPy optimizations.

## Chapter 2: Coding our First Neurons

### Single Neuron Implementation

The fundamental building block that processes inputs through weights and a bias:

```python
# Single Neuron
inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)  # 35.7
```

### Layer of Neurons

Multiple neurons processing the same inputs in parallel:

```python
# Layer of Neurons
inputs = [1.2, 5.1, 2.1]

weights = [
    [0.1, 0.1, -0.3],  # Neuron 1
    [0.1, 0.2, 0.0],   # Neuron 2
    [0.0, 1.3, 0.1]    # Neuron 3
]

biases = [2, 3, 0.5]

outputs = [
    inputs[0]*weights[0][0] + inputs[1]*weights[0][1] + inputs[2]*weights[0][2] + biases[0],
    inputs[0]*weights[1][0] + inputs[1]*weights[1][1] + inputs[2]*weights[1][2] + biases[1],
    inputs[0]*weights[2][0] + inputs[1]*weights[2][1] + inputs[2]*weights[2][2] + biases[2]
]
print(outputs)  # [2.56, 4.32, 7.01]
```

### Using NumPy for Optimization

#### Single Neuron with NumPy

```python
import numpy as np

inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

output = np.dot(inputs, weights) + bias
print(output)  # 35.7
```

#### Layer of Neurons with NumPy

```python
import numpy as np

inputs = [1.2, 5.1, 2.1]
weights = [
    [0.1, 0.1, -0.3],
    [0.1, 0.2, 0.0],
    [0.0, 1.3, 0.1]
]
biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases
print(output)  # [2.56, 4.32, 7.01]
```

### Batch Processing with NumPy

Processing multiple samples simultaneously:

```python
import numpy as np

# 5 samples, each with 3 features
inputs = [
    [1.2, 5.1, 2.1],
    [3.1, 2.1, 8.7],
    [0.1, 0.1, -0.3],
    [0.1, 0.2, 0.0],
    [0.0, 1.3, 0.1]
]

weights = [
    [0.1, 0.1, -0.3],
    [0.1, 0.2, 0.0],
    [0.0, 1.3, 0.1] 
]

biases = [2, 3, 0.5]

# Shape: (5 samples × 3 neurons)
layer_output = np.dot(inputs, np.array(weights).T) + biases
print(layer_output)
```

## Chapter 3: Adding Layers

Firstly, we use the simplest method to add a layer by performing the calculations manually ourselves. Following the example in the book,
We'll create a hidden layer consisting of four inputs with 3 neurons.

```python
import numpy as np

# Hidden Layer: 4 Inputs, 3 Neurons
inputs = [[
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]
]]

weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
biases = [2, 3, 0.5]
# Output for the 2nd layer
weights2 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]
]
biases2 = [-1, 2, -0.5]

layer1_output = np.dot(inputs, np.array(weights).T) + biases
# Note layer1_output is input for the 2nd layer
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

print(layer2_output)
```

### Training Data

In most cases, real data is input to the neural network; however, for training purposes, we use the nnfs package to import a dataset to 
utilize as inputs in our neural network that we will later build. 

```python
#nnfs package contains functions that we can use to create data 
from nnfs.datasets import spiral_data

import nnfs 
import matplotlib.pyplot as plt

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()
```
### Dense Layer Class

So far, we've only dealt with dense or fully connected layers. Recall that weights must be initialized for any model. For example, 
The textbook uses a random initialization for the weights, but it is possible to use predetermined parameters already discovered
by other neural network models. 

In this example, the weights and biases are randomly initialized, following textbook results. 

```python
import numpy as np

class Layer_Dense: 
    # initialization of the layer
    def __init__(self, n_inputs, n_neurons):
        # copy the weights and biases from nnfs textbook 
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
```
Now that we have created a class, an object in Python, we can use this new class and the methods that come along with it to create a hidden
layer and do the hardcoded calculations. So we use the previous spiral data to generate some data to create a new layer and perform a 
forward pass using these new classes created. 

```python
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
```

This ends chapter 3. Next is the application of activation functions to neural networks. 

