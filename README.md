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

Firstly, we use the simplest method to add a layer by manually doing the calculations ourselves. 

```python
```


