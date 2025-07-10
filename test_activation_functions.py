"""Simple tests for activation functions."""

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
    """Test step activation function."""
    step = Activation_Step()
    
    # Test with single neuron output
    inputs = np.array([1.0, -2.0, 0.0, 3.0])
    output = step.forward(inputs)
    expected = np.array([1, 0, 0, 1])
    np.testing.assert_array_equal(output, expected)
    
    # Test with layer output
    layer_output = np.array([[1.0, -1.0], [0.5, -0.5]])
    step_output = step.forward(layer_output.flatten())
    assert len(step_output) == layer_output.size

def test_linear_activation():
    """Test linear activation function."""
    linear = Activation_Linear()
    
    # Test with single neuron output
    inputs = np.array([1.0, -2.0, 0.0, 3.0])
    output = linear.forward(inputs)
    np.testing.assert_array_equal(output, inputs)
    
    # Test with layer output
    layer_output = np.array([[1.0, 2.0], [3.0, 4.0]])
    linear_output = linear.forward(layer_output)
    np.testing.assert_array_equal(linear_output, layer_output)

def test_sigmoid_activation():
    """Test sigmoid activation function."""
    sigmoid = Activation_Sigmoid()
    
    # Test with single neuron output
    inputs = np.array([0.0, 1.0, -1.0])
    output = sigmoid.forward(inputs)
    expected = 1 / (1 + np.exp(-inputs))
    np.testing.assert_array_almost_equal(output, expected, decimal=6)
    
    # Test with layer output
    layer_output = np.array([[1.0, 2.0], [3.0, 4.0]])
    sigmoid_output = sigmoid.forward(layer_output)
    assert sigmoid_output.shape == layer_output.shape
    assert np.all(sigmoid_output >= 0) and np.all(sigmoid_output <= 1)

def test_relu_activation():
    """Test ReLU activation function."""
    relu = Activation_ReLU()
    
    # Test with single neuron output
    inputs = np.array([1.0, -2.0, 0.0, 3.0])
    output = relu.forward(inputs)
    expected = np.array([1.0, 0.0, 0.0, 3.0])
    np.testing.assert_array_equal(output, expected)
    
    # Test with layer output
    layer_output = np.array([[1.0, -1.0], [0.5, -0.5]])
    relu_output = relu.forward(layer_output)
    assert relu_output.shape == layer_output.shape
    assert np.all(relu_output >= 0)

def test_softmax_activation():
    """Test softmax activation function."""
    softmax = Activation_Softmax()
    
    # Test with single neuron output (reshape to 2D)
    inputs = np.array([[1.0, 2.0, 3.0]])
    output = softmax.forward(inputs)
    np.testing.assert_almost_equal(np.sum(output[0]), 1.0, decimal=6)
    
    # Test with layer output
    layer_output = np.array([[1.0, 2.0], [3.0, 4.0]])
    softmax_output = softmax.forward(layer_output)
    assert softmax_output.shape == layer_output.shape
    for i in range(softmax_output.shape[0]):
        np.testing.assert_almost_equal(np.sum(softmax_output[i]), 1.0, decimal=6) 