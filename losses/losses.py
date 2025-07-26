import numpy as np
from activation_function import Activation_Softmax

# Common loss class
class Loss:
    def forward(self, output, y):
        return np.array([])
        
    def calculate(self, output, y):
        sample_losses = self.forward(output, y) # Calculate sample losses
        data_loss = np.mean(sample_losses) # Calculate mean loss
        return data_loss
    
    def backward(self, dvalues, y_true):
        """Backward pass of the loss function."""
        pass

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass of the categorical cross-entropy loss function.
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # Clip data to prevent division by 0
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1) 
        negative_log_likelihoods = -np.log(correct_confidences) # Calculate negative log likelihood
        return negative_log_likelihoods 
    # Backward pass of the categorical cross-entropy loss function.
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        labels = len(dvalues[0])
        # If lables are spares, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        return self.dinputs

# Softmax classifer - combined softmax activation and categorical cross-entropy loss
class Activation_Softmax_Loss_CategoricalCrossentropy():
    # creates activation and loss function objects 
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    
    # Forward Pass
    def forward(self, inputs, y_true):
        # Output layer's activation functon
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate loss
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        return self.dinputs

