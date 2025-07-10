import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Main script for external dataset classification using NNFS neural network."""

import nnfs
nnfs.init()
import numpy as np
from nnfs.datasets import vertical_data

from layer import Layer_Dense
from activation_function import Activation_ReLU, Activation_Softmax
from losses import Loss_CategoricalCrossentropy

def main():
    print("vertical data classification")
    
    X, y = vertical_data(samples=200, classes=2)
    print(f"data shape: {X.shape}")
    print(f"classes: {len(np.unique(y))}")

    dense1 = Layer_Dense(2, 16)
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