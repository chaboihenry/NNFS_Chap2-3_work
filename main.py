"""Main script for NNFS textbook."""

import nnfs
nnfs.init()
import numpy as np
import matplotlib.pyplot as plt

from src import Layer_Dense, Neuron

def main():
    """Run simple neural network experiments."""
    
    print(" Single Neuron Example ")
    # Single neuron example like in textbook
    inputs = np.array([1.2, 5.1, 2.1])
    weights = np.array([3.1, 2.1, 8.7])
    bias = 3
    
    neuron = Neuron(weights, bias)
    neuron.forward(inputs)
    print(f"Inputs: {inputs}")
    print(f"Weights: {weights}")
    print(f"Bias: {bias}")
    print(f"Neuron output: {neuron.output}")
    
    print("\nLayer Example (Textbook Output) ")
    # Reproduces textbook output
    from nnfs.datasets import spiral_data
    X, y = spiral_data(samples=100, classes=3)
    
    layer1 = Layer_Dense(2, 3)
    layer1.forward(X)
    
    print("\nLayer output (first 5 samples):")
    print(layer1.output[:5])
    
    # Plot the data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    plt.title('Spiral Data')
    plt.show()

if __name__ == "__main__":
    main() 