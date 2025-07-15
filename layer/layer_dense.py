import numpy as np

class Layer_Dense:
    """Dense layer that reproduces textbook output exactly."""
    
    def __init__(self, n_inputs, n_neurons):
        """Initialize with the same weights and biases as textbook."""
        # Use the exact same initialization as textbook
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        """Forward pass through the layer."""
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases 
    
    def backward(self, dvalues):
        """Backward pass through the layer."""
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradients on inputs
        self.dinputs = np.dot(dvalues, self.weights.T)