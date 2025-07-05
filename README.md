# Neural Networks from Scratch 

A comprehensive implementation of all the material up to chapter 4 (page 70), encountered in Neural Networks from Scratch by Harrison Kinsley and Daniel Kukieła.

## Overview

This repository contains code and explanations for building single neurons and a layer of neurons, manually and with Numpy.

## Chapter 2: Coding our First Neurons

A singular neuron is a fundamental building block of neural networks. In this chapter, we see how to code and implement a basic neuron that:

- Processes multiple input values
- Applies weights to each input
- Adds a bias term
- Computes a weighted sum output

```python
# Single Neuron
# Output = input * weight + bias

# Inputs
inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

# Output
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)
```
Furthermore, we proceed to a layer of neurons, also known as groups of neurons. We previously reproduced the code, but now we aim to produce an output that contains a value for each of the three distinct neurons. 

```python
# A Layer of Neurons
# Use the same inputs as before, but now we have 3 neurons

# Inputs
inputs = [1.2, 5.1, 2.1]

# Now we have 3 neurons, so we need 3 sets of weights
weights = [
    [0.1, 0.1, -0.3], # Neuron 1
    [0.1, 0.2, 0.0], # Neuron 2
    [0.0, 1.3, 0.1] # Neuron 3
]

# Now we need 3 biases
biases = [2, 3, 0.5]

# Outputs
outputs = [
    # Neuron 1
    inputs[0]*weights[0][0] + inputs[1]*weights[0][1] + inputs[2]*weights[0][2] + biases[0],
    # Neuron 2
    inputs[0]*weights[1][0] + inputs[1]*weights[1][1] + inputs[2]*weights[1][2] + biases[1],
    # Neuron 3
    inputs[0]*weights[2][0] + inputs[1]*weights[2][1] + inputs[2]*weights[2][2] + biases[2]
]
print(outputs)
```
We are then introduced to tensors, arrays, and vectors to help us contain all values for the inputs, weights, and biases so that we can utilize the NumPy library to use dot product and vector addition operations to simultaneously calculate the neuron output of a layer, unlike the method shown before that calculates the neuron outputs one by one in an inefficient manner. 

Therefore, we move on to coding a single neuron but with NumPy. Use the same random numbers for inputs, weights, and biases as before. 

```python
import numpy as np
# A single neuron using Numpy

inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

output = np.dot(inputs, weights) + bias
print(output)
# Notice that the output is the same as the previous example with manual calculations
```

And moving on, we can also code a layer of neurons with Numpy. Using the same numbers as before...

```python
import numpy as np
# A Layer of Neurons using Numpy

inputs = [1.2, 5.1, 2.1]
weights = [
    [0.1, 0.1, -0.3],
    [0.1, 0.2, 0.0],
    [0.0, 1.3, 0.1]
]

biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases
print(output)
# Notice the output is the same as the previous example with manual calculations, the difference of .0000000...1 is
# due to the precision of the floating point numbers
```
We are then introduced to "A Batch of Data," where we are given more than one sample observation for the neural network to produce outputs. This leads to the use of the Matrix product and transposition operation to facilitate the computation of all the neurons for all the sample observations given. 

Therefore, we code a layer of neurons with a batch of data with Numpy

```python
# A Layer of (3)Neurons using Numpy but with multiple inputs (batch of data)

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

# np.array(weights).T is the transpose of the weights matrix 
# which is the same as the weights matrix but with the rows and columns swapped
layer_output = np.dot(inputs, np.array(weights).T) + biases
print(layer_output)
# note we have 5 samples in the batch, and 3 neurons in the layer, so we expect 5x3 output
```

## Chapter 3: Adding Layers












