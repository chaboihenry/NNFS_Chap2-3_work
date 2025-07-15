"""Simple tests for loss functions."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from losses import Loss_CategoricalCrossentropy

def test_categorical_crossentropy():
    """Test categorical crossentropy loss."""
    loss_function = Loss_CategoricalCrossentropy()
    
    # Test with softmax outputs and categorical labels
    softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                [0.1, 0.5, 0.4],
                                [0.02, 0.9, 0.08]])
    class_targets = np.array([0, 1, 1])
    
    loss = loss_function.calculate(softmax_outputs, class_targets)
    assert loss > 0
    assert np.isfinite(loss)
    
    # Test individual sample losses
    sample_losses = loss_function.forward(softmax_outputs, class_targets)
    assert len(sample_losses) == 3
    assert np.all(sample_losses > 0)

def test_categorical_crossentropy_onehot():
    """Test categorical crossentropy with one-hot labels."""
    loss_function = Loss_CategoricalCrossentropy()
    
    softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                [0.1, 0.5, 0.4]])
    onehot_targets = np.array([[1, 0, 0],
                               [0, 1, 0]])
    
    loss = loss_function.calculate(softmax_outputs, onehot_targets)
    assert loss > 0
    assert np.isfinite(loss)

def test_categorical_crossentropy_with_network():
    """Test loss function with network output."""
    import nnfs
    nnfs.init()
    from layer import Layer_Dense
    from activation_function import Activation_ReLU, Activation_Softmax
    from nnfs.datasets import spiral_data
    
    # Create network like textbook
    X, y = spiral_data(samples=100, classes=3)
    dense1 = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()
    
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    # Calculate loss
    loss_function = Loss_CategoricalCrossentropy()
    loss = loss_function.calculate(activation2.output, y)
    assert loss > 0
    assert np.isfinite(loss) 