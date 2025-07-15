"""Simple tests for the Neuron class."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from neuron import Neuron

def test_neuron_initialization():
    """Test that Neuron initializes correctly."""
    weights = [1.0, 2.0, 3.0]
    bias = 0.5
    neuron = Neuron(weights, bias)
    assert len(neuron.weights) == 3
    assert neuron.bias == 0.5

def test_neuron_forward():
    """Test that neuron forward pass works."""
    weights = [0.5, 0.3]
    bias = 1.0
    neuron = Neuron(weights, bias)
    inputs = np.array([1.0, 2.0])
    neuron.forward(inputs)
    expected = 1.0 * 0.5 + 2.0 * 0.3 + 1.0  # manual calculation
    assert abs(neuron.output - expected) < 1e-10

def test_neuron_textbook_example():
    """Test the textbook neuron example."""
    inputs = np.array([1.2, 5.1, 2.1])
    weights = np.array([3.1, 2.1, 8.7])
    bias = 3
    
    neuron = Neuron(weights, bias)
    neuron.forward(inputs)
    
    # Manual calculation from textbook
    expected = 1.2 * 3.1 + 5.1 * 2.1 + 2.1 * 8.7 + 3
    assert abs(neuron.output - expected) < 1e-10

def test_layer_of_neurons_manual():
    """Test the textbook layer of neurons example (manual calculation)."""
    inputs = np.array([1.2, 5.1, 2.1])
    
    # Three neurons with different weights and biases
    weights1 = [0.1, 0.1, -0.3]
    weights2 = [0.1, 0.2, 0.0]
    weights3 = [0.0, 1.3, 0.1]
    
    bias1, bias2, bias3 = 2, 3, 0.5
    
    # Create three neurons
    neuron1 = Neuron(weights1, bias1)
    neuron2 = Neuron(weights2, bias2)
    neuron3 = Neuron(weights3, bias3)
    
    # Forward pass
    neuron1.forward(inputs)
    neuron2.forward(inputs)
    neuron3.forward(inputs)
    
    # Manual calculations from textbook
    expected1 = 1.2 * 0.1 + 5.1 * 0.1 + 2.1 * (-0.3) + 2
    expected2 = 1.2 * 0.1 + 5.1 * 0.2 + 2.1 * 0.0 + 3
    expected3 = 1.2 * 0.0 + 5.1 * 1.3 + 2.1 * 0.1 + 0.5
    
    assert abs(neuron1.output - expected1) < 1e-10
    assert abs(neuron2.output - expected2) < 1e-10
    assert abs(neuron3.output - expected3) < 1e-10 