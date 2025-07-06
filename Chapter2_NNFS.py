import numpy as np

# Single Neuron

inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)  # 35.7

########################################################################################

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

########################################################################################

# A signle neuron using Numpy

inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

output = np.dot(inputs, weights) + bias
print(output)  # 35.7
# notice the output is the same as the previous example with manual calculations

########################################################################################

# A Layer of Neurons using Numpy

inputs = [1.2, 5.1, 2.1]
weights = [
    [0.1, 0.1, -0.3],
    [0.1, 0.2, 0.0],
    [0.0, 1.3, 0.1]
]
biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases
print(output)  # [2.56, 4.32, 7.01]
# notice the output is the same as the previous example with manual calculations, 
# the difference of .0000000...1 is due to the preicison of the floating point numbers

########################################################################################

# A Layer of Neurons using Numpy but with multiple inputs (batch of data)

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
# note we have 5 samples in the batch, and 3 neurons in the layer, so we expect 5x3 output





