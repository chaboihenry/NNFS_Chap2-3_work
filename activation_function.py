import numpy as np 

class Activation_Step:
    """Step activation function."""
    
    def forward(self, inputs):
        """Forward pass of the step activation function."""
        self.output = np.array([int(x > 0) for x in inputs])
        return self.output

class Activation_Linear:
    """Linear activation function."""

    def forward(self, inputs):
        """Forward pass of the linear activation function."""
        self.output = inputs
        return self.output
    
class Activation_Sigmoid:
    """Sigmoid activation function."""

    def forward(self, inputs):
        """Forward pass of the sigmoid activation function."""
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

class Activation_ReLU:
    """ReLU activation function."""

    def forward(self, inputs):
        """Forward pass of the ReLU activation function."""
        self.output = np.maximum(0, inputs)
        return self.output
    
class Activation_Softmax:
    """Softmax activation function."""

    def forward(self, inputs):
        """Forward pass of the softmax activation function."""
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
