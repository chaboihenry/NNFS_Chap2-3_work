"""Explore vertical_data from nnfs to see what it looks like."""

import nnfs
nnfs.init()
import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import vertical_data

def main():
    print("exploring vertical_data")
    
    # Generate the data
    X, y = vertical_data(samples=200, classes=2)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"unique y values: {np.unique(y)}")
    print(f"first 10 X values: {X[:10]}")
    print(f"first 10 y values: {y[:10]}")
    
    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title('Vertical Data Visualization')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main() 