import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses import Activation_Softmax_Loss_CategoricalCrossentropy

"""Main script for NNFS textbook."""

import nnfs
nnfs.init()
import numpy as np
import matplotlib.pyplot as plt

from layer import Layer_Dense
from neuron import Neuron
from activation_function import Activation_ReLU, Activation_Softmax
from losses import Loss_CategoricalCrossentropy
from optimizers import Optimizer_ADAM

def main():
    """Run simple neural network experiments."""
    
    # Single neuron example like in textbook
    inputs = np.array([1.2, 5.1, 2.1])
    weights = np.array([3.1, 2.1, 8.7])
    bias = 3
    
    neuron = Neuron(weights, bias)
    neuron.forward(inputs)
    # print(inputs)
    # print(weights)
    # print(bias)
    # print(neuron.output)
    
    #########################################################
    # Chapter 3 final code output
    #########################################################
    from nnfs.datasets import spiral_data
    X, y = spiral_data(samples=100, classes=3)
    
    layer1 = Layer_Dense(2, 3)
    layer1.forward(X)
    
    # print(layer1.output[:5])
    
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
    # print(activation2.output[:5])
    
    #########################################################
    # Chapter 5 final code output
    #########################################################
    # Create loss function
    loss_function = Loss_CategoricalCrossentropy()
    
    # Calculate loss
    loss = loss_function.calculate(activation2.output, y)
    
    # Print loss
    # print(loss)
    
    # Calculate accuracy
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)
    
    # Print accuracy
    # print(accuracy)

    #########################################################
    # Chapter 6 final code output
    #########################################################
   
    X, y = spiral_data(samples=100, classes=3)
    dense1 = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()
    loss_function = Loss_CategoricalCrossentropy()
    
    lowest_loss = 99999
    # Best weights and biases found so far
    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases = dense2.biases.copy()
    
    for iteration in range(10000):
        
        # Update weights and biases
        dense1.weights += 0.05 * np.random.randn(2, 3)
        dense1.biases += 0.05 * np.random.randn(1, 3)
        dense2.weights += 0.05 * np.random.randn(3, 3)
        dense2.biases += 0.05 * np.random.randn(1, 3)
        
        # Perform forward pass
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        
        # Calculate loss
        loss = loss_function.calculate(activation2.output, y)
        
        # Calculate accuracy from output of act2 and targets
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == y)
        
        # If loss is smaller - print and save weights and biases aside
        if loss < lowest_loss:
            # print(f'New set of weights found, iteration: {iteration} loss: {loss:.7f} acc: {accuracy}')
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            lowest_loss = loss
        # Revert weights and biases
        else:
            dense1.weights = best_dense1_weights.copy()
            dense1.biases = best_dense1_biases.copy()
            dense2.weights = best_dense2_weights.copy()
            dense2.biases = best_dense2_biases.copy()
    
    #########################################################
    # Chapter 9 final code output
    #########################################################
    # Create dataset
    X, y = spiral_data(samples=100, classes=3)
    dense1 = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(3, 3)
    # Create Softmax classifier's combined loss and activation
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    # Forward pass through the second dense layer
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    # See the output of the first 5 samples:
    #print(loss_activation.output[:5])
    # print loss value
    #print(f'loss: {loss}')
    # Calculate accuracy from output of act2 and targets
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    # Print accuracy
    print(f'acc: {accuracy}')

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Print gradients
    #print(dense1.dweights)
    #print(dense1.dbiases)
    #print(dense2.dweights)
    #print(dense2.dbiases)
    
    ######################################################
    # Chapter 10 & 11 final code output
    ######################################################
    # Create dataset
    X, y = spiral_data(samples=100, classes=3)
    dense1 = Layer_Dense(2, 64)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 3)
    # Create Softmax classifier's combined loss and activation
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    # Create optimizer
    optimizer = Optimizer_ADAM(learning_rate=0.05, decay=1e-3)

    # Training loop
    for epoch in range(10001):
        # Forward pass
        # Perform a forward pass of our training data through this layer
        dense1.forward(X)
        # Perform a forward pass through activation function
        # takes the output of first dense layer here
        activation1.forward(dense1.output)
        # Perform a forward pass through second Dense layer
        # takes outputs of activation function of first layer as inputs    
        dense2.forward(activation1.output)
        # Perform a forward pass through the activation/loss function
        # takes the output of second dense layer here and returns loss
        loss = loss_activation.forward(dense2.output, y)

        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)

        # Print loss and accuracy every 1000 epochs
        if not epoch % 1000:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}', 
                  f'lr: {optimizer.current_learning_rate:.5f}')
        
        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

    # Create test dataset
    X_test, y_test = spiral_data(samples=100, classes=3)
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y_test)
        
    # Calculate accuracy of the testing dataset
    predicitons = np.argmax(loss_activation.output, axis=1)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)
    accuracy = np.mean(predicitons == y_test)

    # Print accuracy and loss of the testing dataset
    print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

if __name__ == "__main__":
    main() 
