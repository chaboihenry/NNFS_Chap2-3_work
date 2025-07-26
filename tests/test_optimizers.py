import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from layer import Layer_Dense
from optimizers import Optimizer_SGD, Optimizer_ADAGrad, Optimizer_RMSProp, Optimizer_ADAM

def test_optimizers():
    """simple test to see if optimizers work"""
    
    print("testing optimizers...")
    
    # make a simple layer
    layer = Layer_Dense(2, 3)
    
    # fake gradients (like from backward pass)
    layer.dweights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    layer.dbiases = np.array([[0.01, 0.02, 0.03]])
    
    print(f"layer weights: {layer.weights.shape}")
    print(f"layer biases: {layer.biases.shape}")
    print()
    
    # test SGD
    print("testing SGD...")
    sgd = Optimizer_SGD(learning_rate=0.01)
    sgd.pre_update_params()
    sgd.update_params(layer)
    sgd.post_update_params()
    print("sgd works!")
    print()
    
    # test SGD with momentum
    print("testing SGD with momentum...")
    sgd_momentum = Optimizer_SGD(learning_rate=0.01, momentum=0.9)
    sgd_momentum.pre_update_params()
    sgd_momentum.update_params(layer)
    sgd_momentum.post_update_params()
    print("sgd with momentum works!")
    print()
    
    # test AdaGrad
    print("testing AdaGrad...")
    adagrad = Optimizer_ADAGrad(learning_rate=0.01)
    adagrad.pre_update_params()
    adagrad.update_params(layer)
    adagrad.post_update_params()
    print("adagrad works!")
    print()
    
    # test RMSProp
    print("testing RMSProp...")
    rmsprop = Optimizer_RMSProp(learning_rate=0.01)
    rmsprop.pre_update_params()
    rmsprop.update_params(layer)
    rmsprop.post_update_params()
    print("rmsprop works!")
    print()
    
    # test Adam
    print("testing Adam...")
    adam = Optimizer_ADAM(learning_rate=0.01)
    adam.pre_update_params()
    adam.update_params(layer)
    adam.post_update_params()
    print("adam works!")
    print()
    
    print("all optimizers work! :)")

if __name__ == "__main__":
    test_optimizers() 