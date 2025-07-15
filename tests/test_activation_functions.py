"""Simple tests for activation functions - forward and backward passes."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from activation_function import (
    Activation_Step, 
    Activation_Linear, 
    Activation_Sigmoid, 
    Activation_ReLU, 
    Activation_Softmax
)

def test_step_activation():
    """Test step activation forward and backward pass."""
    step = Activation_Step()
    
    # Forward pass
    inputs = np.array([1.0, -2.0, 0.0, 3.0])
    output = step.forward(inputs)
    expected = np.array([1, 0, 0, 1])
    np.testing.assert_array_equal(output, expected)
    
    # Backward pass
    dvalues = np.array([0.1, 0.2, 0.3, 0.4])
    dinputs = step.backward(dvalues)
    expected = np.zeros_like(dvalues)  # Step has zero derivative
    np.testing.assert_array_equal(dinputs, expected)

def test_linear_activation():
    """Test linear activation forward and backward pass."""
    linear = Activation_Linear()
    
    # Forward pass
    inputs = np.array([1.0, -2.0, 0.0, 3.0])
    output = linear.forward(inputs)
    np.testing.assert_array_equal(output, inputs)
    
    # Backward pass
    dvalues = np.array([0.1, 0.2, 0.3, 0.4])
    dinputs = linear.backward(dvalues)
    np.testing.assert_array_equal(dinputs, dvalues)  # Linear has derivative of 1

def test_sigmoid_activation():
    """Test sigmoid activation forward and backward pass."""
    sigmoid = Activation_Sigmoid()
    
    # Forward pass
    inputs = np.array([0.0, 1.0, -1.0, 2.0])
    output = sigmoid.forward(inputs)
    expected = 1 / (1 + np.exp(-inputs))
    np.testing.assert_array_almost_equal(output, expected, decimal=6)
    
    # Backward pass
    dvalues = np.array([0.1, 0.2, 0.3, 0.4])
    dinputs = sigmoid.backward(dvalues)
    assert np.all(np.isfinite(dinputs))
    assert dinputs.shape == dvalues.shape
    assert np.all(dinputs <= 0.25 * dvalues)  # Sigmoid derivative max is 0.25

def test_relu_activation():
    """Test ReLU activation forward and backward pass."""
    relu = Activation_ReLU()
    
    # Forward pass
    inputs = np.array([1.0, -2.0, 0.0, 3.0])
    output = relu.forward(inputs)
    expected = np.array([1.0, 0.0, 0.0, 3.0])
    np.testing.assert_array_equal(output, expected)
    
    # Backward pass
    dvalues = np.array([0.1, 0.2, 0.3, 0.4])
    dinputs = relu.backward(dvalues)
    expected = np.array([0.1, 0.0, 0.0, 0.4])  # Positive inputs get gradient, negative get zero
    np.testing.assert_array_equal(dinputs, expected)

def test_softmax_activation():
    """Test softmax activation forward and backward pass."""
    softmax = Activation_Softmax()
    
    # Forward pass (2D input required)
    inputs = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    output = softmax.forward(inputs)
    
    # Check probabilities sum to 1
    for i in range(output.shape[0]):
        np.testing.assert_almost_equal(np.sum(output[i]), 1.0, decimal=6)
    
    # Backward pass
    dvalues = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    dinputs = softmax.backward(dvalues)
    assert np.all(np.isfinite(dinputs))
    assert dinputs.shape == dvalues.shape

def test_layer_dense_backward():
    """Test dense layer backward pass."""
    from layer import Layer_Dense
    
    layer = Layer_Dense(3, 2)
    inputs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dvalues = np.array([[0.1, 0.2], [0.3, 0.4]])
    
    layer.forward(inputs)
    layer.backward(dvalues)
    
    # Check gradients exist and have correct shapes
    assert hasattr(layer, 'dweights')
    assert hasattr(layer, 'dbiases')
    assert hasattr(layer, 'dinputs')
    assert layer.dweights.shape == layer.weights.shape
    assert layer.dbiases.shape == layer.biases.shape
    assert layer.dinputs.shape == inputs.shape

def test_loss_backward():
    """Test loss function backward pass."""
    from losses import Loss_CategoricalCrossentropy
    
    loss = Loss_CategoricalCrossentropy()
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    y_true = np.array([0, 1])
    dvalues = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    
    loss.forward(y_pred, y_true)
    dinputs = loss.backward(dvalues, y_true)
    
    assert hasattr(loss, 'dinputs')
    assert dinputs.shape == dvalues.shape 