import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Main script for personality classification using NNFS neural network."""

import nnfs
nnfs.init()
import numpy as np
import pandas as pd

from layer import Layer_Dense
from activation_function import Activation_ReLU, Activation_Softmax
from losses import Activation_Softmax_Loss_CategoricalCrossentropy
from optimizers import Optimizer_SGD, Optimizer_ADAM

def main():
    print("Personality Classification - Optimizer Comparison with Validation")
    
    # Load and preprocess data
    df = pd.read_csv('datasets/personality_dataset.csv')
    df['Stage_fear'] = (df['Stage_fear'] == 'Yes').astype(int)
    df['Drained_after_socializing'] = (df['Drained_after_socializing'] == 'Yes').astype(int)
    
    X = df.drop('Personality', axis=1).values
    y = (df['Personality'].values == 'Extrovert').astype(int)
    
    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Split data 80-20 for training and testing
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Data shape: {X.shape}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Classes: {len(np.unique(y))}")

    ######################################################
    # Training with SGD Optimizer
    ######################################################
    print("\n" + "="*50)
    print("Training with SGD Optimizer")
    print("="*50)
    
    # Create network
    dense1 = Layer_Dense(X_train.shape[1], 16)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(16, 16)
    activation2 = Activation_ReLU()
    dense3 = Layer_Dense(16, 2)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    
    # Create SGD optimizer
    optimizer = Optimizer_SGD(learning_rate=0.01, decay=1e-4)

    # Training loop
    for epoch in range(10001):
        # Forward pass
        dense1.forward(X_train)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        dense3.forward(activation2.output)
        loss = loss_activation.forward(dense3.output, y_train)

        # Calculate accuracy
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y_train.shape) == 2:
            y_train = np.argmax(y_train, axis=1)
        accuracy = np.mean(predictions == y_train)

        # Print loss and accuracy every 1000 epochs
        if not epoch % 1000:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}', 
                  f'lr: {optimizer.current_learning_rate:.5f}')
        
        # Backward pass
        loss_activation.backward(loss_activation.output, y_train)
        dense3.backward(loss_activation.dinputs)
        activation2.backward(dense3.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)
        optimizer.post_update_params()

    # Test SGD model on unseen data
    print("\n SGD Validation Results:")
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    loss = loss_activation.forward(dense3.output, y_test)
    
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_test)
    print(f'SGD validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

    ######################################################
    # Training with ADAM Optimizer
    ######################################################
    print("\n" + "="*50)
    print("Training with ADAM Optimizer")
    print("="*50)
    
    # Create network
    dense1 = Layer_Dense(X_train.shape[1], 16)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(16, 16)
    activation2 = Activation_ReLU()
    dense3 = Layer_Dense(16, 2)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    # Create ADAM optimizer
    optimizer = Optimizer_ADAM(learning_rate=0.01, decay=1e-4)

    # Training loop
    for epoch in range(10001):
        # Forward pass
        dense1.forward(X_train)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        dense3.forward(activation2.output)
        loss = loss_activation.forward(dense3.output, y_train)

        # Calculate accuracy
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y_train.shape) == 2:
            y_train = np.argmax(y_train, axis=1)
        accuracy = np.mean(predictions == y_train)

        # Print loss and accuracy every 1000 epochs
        if not epoch % 1000:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}', 
                  f'lr: {optimizer.current_learning_rate:.5f}')
        
        # Backward pass
        loss_activation.backward(loss_activation.output, y_train)
        dense3.backward(loss_activation.dinputs)
        activation2.backward(dense3.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)
        optimizer.post_update_params()
    # Test ADAM model on unseen data
    print("\n ADAM Validation Results:")
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    loss = loss_activation.forward(dense3.output, y_test)
    
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_test)
    print(f'ADAM validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

if __name__ == "__main__":
    main() 