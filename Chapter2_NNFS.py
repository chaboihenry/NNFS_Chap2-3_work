import numpy as np

# Single Neuron
# Output = input * weight + bias

# Inputs
inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

# Output
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)

# A Layer of Neurons
# use the same inputs as before, but now we have 3 neurons

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

# A signle neuron using Numpy

inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

output = np.dot(inputs, weights) + bias
print(output)
# notice the output is the same as the previous example with manual calculations

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
# notice the output is the same as the previous example with manual calculations, the difference of .0000000...1 is due to the preicison of the floating point numbers

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



