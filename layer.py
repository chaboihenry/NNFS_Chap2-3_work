import numpy as np

class Layer:
    """Simple layer containing a list of neurons."""
    
    def __init__(self, neurons):
        """Initialize with a list of neurons."""
        self.neurons = neurons
    
    def forward(self, inputs):
        """Forward pass through all neurons."""
        outputs = []
        for neuron in self.neurons:
            neuron.forward(inputs)
            outputs.append(neuron.output)
        self.output = outputs

class Layer_Dense:
    """Dense layer that reproduces textbook output exactly."""
    
    def __init__(self, n_inputs, n_neurons):
        """Initialize with the same weights and biases as textbook."""
        # Use the exact same initialization as textbook
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        """Forward pass through the layer."""
        self.output = np.dot(inputs, self.weights) + self.biases 