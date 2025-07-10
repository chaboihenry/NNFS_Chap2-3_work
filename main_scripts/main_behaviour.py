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
from losses import Loss_CategoricalCrossentropy

def main():
    print("personality classification")
    
    # Load and preprocess data
    df = pd.read_csv('Datasets/personality_dataset.csv')
    df['Stage_fear'] = (df['Stage_fear'] == 'Yes').astype(int)
    df['Drained_after_socializing'] = (df['Drained_after_socializing'] == 'Yes').astype(int)
    
    X = df.drop('Personality', axis=1).values
    y = (df['Personality'].values == 'Extrovert').astype(int)
    
    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    print(f"data shape: {X.shape}")
    print(f"classes: {len(np.unique(y))}")

    dense1 = Layer_Dense(X.shape[1], 16)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(16, 16)
    activation2 = Activation_ReLU()
    dense3 = Layer_Dense(16, 2)
    activation3 = Activation_Softmax()
    loss_function = Loss_CategoricalCrossentropy()

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    
    loss = loss_function.calculate(activation3.output, y)
    predictions = np.argmax(activation3.output, axis=1)
    accuracy = np.mean(predictions == y)
    
    print(f"loss: {loss:.6f}")
    print(f"accuracy: {accuracy:.6f}")

if __name__ == "__main__":
    main() 
