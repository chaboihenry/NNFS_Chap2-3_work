import numpy as np

class Layer:
    """Simple layer containing a list of neurons."""
    
    def __init__(self, neurons):
        """Initialize with a list of neurons."""
        self.neurons = neurons
    
    def forward(self, inputs):
        """Forward pass through all neurons."""
        self.inputs = inputs
        outputs = []
        for neuron in self.neurons:
            neuron.forward(inputs)
            outputs.append(neuron.output)
        self.output = outputs
    
    def backward(self, dvalues):
        """Backward pass through all neurons."""
        # Each neuron gets its own gradient from dvalues
        self.dinputs = 0
        for i, neuron in enumerate(self.neurons):
            neuron.backward(dvalues[i])
            self.dinputs += neuron.dinputs 