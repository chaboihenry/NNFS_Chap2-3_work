import numpy as np

class Neuron:
    """General single neuron as shown in NNFS textbook."""
    
    def __init__(self, weights, bias):
        """Initialize neuron with specific weights and bias."""
        self.weights = np.array(weights)
        self.bias = bias
    
    def forward(self, inputs):
        """Forward pass through the neuron."""
        self.output = np.dot(inputs, self.weights) + self.bias 
