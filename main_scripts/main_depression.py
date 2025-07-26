import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Main script for depression vs ME/CFS classification using NNFS neural network."""

import nnfs
nnfs.init()
import numpy as np
import pandas as pd

from layer import Layer_Dense
from activation_function import Activation_ReLU, Activation_Softmax
from losses import Activation_Softmax_Loss_CategoricalCrossentropy
from optimizers import Optimizer_SGD, Optimizer_ADAM

def main():
    print("Depression vs ME/CFS Classification - Optimizer Comparison with Validation")
    
    # Load and preprocess data
    df = pd.read_csv('datasets/me_cfs_vs_depression_dataset.csv')
    
    # Fill missing numeric values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Convert categorical variables to numeric
    df['pem_present'] = (df['pem_present'] == 1).astype(int)
    df['meditation_or_mindfulness'] = (df['meditation_or_mindfulness'] == 'Yes').astype(int)
    df['gender'] = (df['gender'] == 'Male').astype(int)
    work_mapping = {'Working': 0, 'Partially working': 1, 'Not working': 2}
    social_mapping = {'Very low': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very high': 4}
    exercise_mapping = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Daily': 4}
    df['work_status'] = df['work_status'].replace(work_mapping).fillna(1).astype(int)
    df['social_activity_level'] = df['social_activity_level'].replace(social_mapping).fillna(2).astype(int)
    df['exercise_frequency'] = df['exercise_frequency'].replace(exercise_mapping).fillna(2).astype(int)
    
    # Prepare features and labels
    X = df.drop(['diagnosis', 'age'], axis=1).to_numpy()
    
    # Encode labels: 0=ME/CFS, 1=Depression, 2=Both
    diagnosis_mapping = {'ME/CFS': 0, 'Depression': 1, 'Both': 2}
    y = df['diagnosis'].replace(diagnosis_mapping).astype(int).to_numpy()
    
    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Split data 80-20 for training and testing
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Classes: {len(np.unique(y_train))}")

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
    dense3 = Layer_Dense(16, 3)
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
    dense3 = Layer_Dense(16, 3)
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