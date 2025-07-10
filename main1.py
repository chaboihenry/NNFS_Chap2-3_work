"""Main script for NNFS textbook."""

import nnfs
nnfs.init()
import numpy as np
import matplotlib.pyplot as plt

from layer.layer_dense import Layer_Dense
from neuron.neuron import Neuron
from activation_function import Activation_ReLU, Activation_Softmax
from losses.losses import Loss_CategoricalCrossentropy

def main():
    """Run simple neural network experiments."""
    
    # Single neuron example like in textbook
    inputs = np.array([1.2, 5.1, 2.1])
    weights = np.array([3.1, 2.1, 8.7])
    bias = 3
    
    neuron = Neuron(weights, bias)
    neuron.forward(inputs)
    print(inputs)
    print(weights)
    print(bias)
    print(neuron.output)
    
    #########################################################
    # Chapter 3 final code output
    #########################################################
    from nnfs.datasets import spiral_data
    X, y = spiral_data(samples=100, classes=3)
    
    layer1 = Layer_Dense(2, 3)
    layer1.forward(X)
    
    print(layer1.output[:5])
    
    #########################################################
    # Chapter 4 final code output 
    #########################################################
    # Create dataset
    X, y = spiral_data(samples=100, classes=3)
    
    # Create Dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 3)
    
    # Create ReLU activation (to be used with Dense layer):
    activation1 = Activation_ReLU()
    
    # Create second Dense layer with 3 input features (as we take output
    # of previous layer here) and 3 output values
    dense2 = Layer_Dense(3, 3)
    
    # Create Softmax activation (to be used with Dense layer):
    activation2 = Activation_Softmax()
    
    # Make a forward pass of our training data through this layer
    dense1.forward(X)
    
    # Make a forward pass through activation function
    # it takes the output of first dense layer here
    activation1.forward(dense1.output)
    
    # Make a forward pass through second Dense layer
    # it takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)
    
    # Make a forward pass through activation function
    # it takes the output of second dense layer here
    activation2.forward(dense2.output)
    
    # Output of the first few samples:
    print(activation2.output[:5])
    
    #########################################################
    # Chapter 5 final code output
    #########################################################
    # Create loss function
    loss_function = Loss_CategoricalCrossentropy()
    
    # Calculate loss
    loss = loss_function.calculate(activation2.output, y)
    
    # Print loss
    print(loss)
    
    # Calculate accuracy
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)
    
    # Print accuracy
    print(accuracy)
    
    # Plot the data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    plt.title('Spiral Data')
    plt.show()

if __name__ == "__main__":
    main() 
