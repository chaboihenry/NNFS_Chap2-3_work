import numpy as np 

class Activation_Step:
    """Step activation function."""
    
    def forward(self, inputs):
        """Forward pass of the step activation function."""
        self.inputs = inputs
        self.output = np.array([int(x > 0) for x in inputs])
        return self.output
    
    def backward(self, dvalues):
        """Backward pass of the step activation function."""
        # Step function has zero derivative everywhere except at x=0
        # where it's undefined.
        self.dinputs = np.zeros_like(dvalues)
        return self.dinputs

class Activation_Linear:
    """Linear activation function."""

    def forward(self, inputs):
        """Forward pass of the linear activation function."""
        self.inputs = inputs
        self.output = inputs
        return self.output
    
    def backward(self, dvalues):
        """Backward pass of the linear activation function."""
        # Linear function has derivative of 1 everywhere
        self.dinputs = dvalues.copy()
        return self.dinputs
    
class Activation_Sigmoid:
    """Sigmoid activation function."""

    def forward(self, inputs):
        """Forward pass of the sigmoid activation function."""
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output
    
    def backward(self, dvalues):
        """Backward pass of the sigmoid activation function."""
        # Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        self.dinputs = dvalues * (1 - self.output) * self.output
        return self.dinputs

class Activation_ReLU:
    """ReLU activation function."""

    # Forward pass of the ReLU activation function.
    def forward(self, inputs):
        """Forward pass of the ReLU activation function."""
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward(self, dvalues):
        """Backward pass of the ReLU activation function."""
        # make copy of values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs
    
class Activation_Softmax:
    """Softmax activation function."""

    def forward(self, inputs):
        """Forward pass of the softmax activation function."""
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
    
    def backward(self, dvalues):
        """Backward pass of the softmax activation function."""
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the arry of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
        return self.dinputs


    
