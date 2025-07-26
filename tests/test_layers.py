"""Tests for the Layer_Dense class."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from layer import Layer_Dense, Layer
from neuron import Neuron

def test_layer_dense_initialization():
    """Test that Layer_Dense initializes correctly."""
    layer = Layer_Dense(3, 2)
    
    assert layer.weights.shape == (3, 2)
    assert layer.biases.shape == (1, 2)
    assert layer.biases[0, 0] == 0  # Biases should be zero

def test_layer_dense_forward():
    """Test that forward pass works correctly."""
    layer = Layer_Dense(2, 3)
    
    # Test input
    inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    # Perform forward pass
    layer.forward(inputs)
    
    # Check output shape
    assert layer.output.shape == (2, 3)
    
    # Check that output is not all zeros (weights are random)
    assert not np.allclose(layer.output, 0)

def test_layer_dense_reproducible():
    """Test that results are reproducible with nnfs.init()."""
    import nnfs
    nnfs.init()
    
    # Set a specific seed for reproducibility
    np.random.seed(0)
    
    layer1 = Layer_Dense(2, 3)
    inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
    layer1.forward(inputs)
    output1 = layer1.output.copy()
    
    # Reset seed and create another layer
    np.random.seed(0)
    layer2 = Layer_Dense(2, 3)
    layer2.forward(inputs)
    output2 = layer2.output
    
    # Results should be identical with same seed
    np.testing.assert_array_almost_equal(output1, output2)

def test_layer_textbook_example():
    """Test the textbook layer example with specific weights and biases."""
    import nnfs
    nnfs.init()
    
    # Textbook example inputs
    inputs = np.array([1.2, 5.1, 2.1])
    
    # Textbook weights and biases
    weights = np.array([
        [0.1, 0.1, -0.3],
        [0.1, 0.2, 0.0],
        [0.0, 1.3, 0.1]
    ])
    biases = np.array([2, 3, 0.5])
    
    # Manual calculation from textbook
    expected = np.dot(weights, inputs) + biases
    
    # Test with our layer (using same initialization)
    layer = Layer_Dense(3, 3)
    layer.weights = weights.T  # Transpose to match our implementation
    layer.biases = biases.reshape(1, -1)
    
    layer.forward(inputs.reshape(1, -1))  # Reshape to 2D
    
    np.testing.assert_array_almost_equal(layer.output[0], expected)

def test_batch_processing():
    """Test batch processing as shown in textbook."""
    import nnfs
    nnfs.init()
    
    # Batch of inputs (5 samples, 3 features each)
    inputs = np.array([
        [1.2, 5.1, 2.1],
        [3.1, 2.1, 8.7],
        [0.1, 0.1, -0.3],
        [0.1, 0.2, 0.0],
        [0.0, 1.3, 0.1]
    ])
    
    layer = Layer_Dense(3, 3)
    layer.forward(inputs)
    
    # Check output shape: (5 samples, 3 neurons)
    assert layer.output.shape == (5, 3)
    
    # Check that all outputs are reasonable (not all zeros, not all same)
    assert not np.allclose(layer.output, 0)
    assert not np.allclose(layer.output, layer.output[0, 0])

def test_simple_layer():
    """Test the simple Layer class with neurons."""
    # Create some neurons
    neuron1 = Neuron([0.1, 0.2], 1.0)
    neuron2 = Neuron([0.3, 0.4], 2.0)
    
    # Create a layer with these neurons
    layer = Layer([neuron1, neuron2])
    
    # Test forward pass
    inputs = np.array([1.0, 2.0])
    layer.forward(inputs)
    
    # Check outputs
    assert len(layer.output) == 2
    assert abs(layer.output[0] - (1.0 * 0.1 + 2.0 * 0.2 + 1.0)) < 1e-10
    assert abs(layer.output[1] - (1.0 * 0.3 + 2.0 * 0.4 + 2.0)) < 1e-10 