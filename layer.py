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