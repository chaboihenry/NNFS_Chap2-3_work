import numpy as np

# Common loss class
class Loss:
    def forward(self, output, y):
        return np.array([])
        
    def calculate(self, output, y):
        sample_losses = self.forward(output, y) # Calculate sample losses
        data_loss = np.mean(sample_losses) # Calculate mean loss
        return data_loss

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # Clip data to prevent division by 0
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1) 
        negative_log_likelihoods = -np.log(correct_confidences) # Calculate negative log likelihood
        return negative_log_likelihoods 
