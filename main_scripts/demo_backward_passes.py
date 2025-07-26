import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from activation_function import Activation_Step, Activation_Linear, Activation_Sigmoid, Activation_ReLU, Activation_Softmax
from layer import Layer_Dense
from losses import Loss_CategoricalCrossentropy

print("Step activation:")
a = Activation_Step()
x = np.array([1.0, -2.0, 0.0, 3.0])
forward = a.forward(x)
print(forward)
dvalues = np.array([0.1, 0.2, 0.3, 0.4])
backward = a.backward(dvalues)
print(backward)

print("\nLinear activation:")
a = Activation_Linear()
x = np.array([1.0, -2.0, 0.0, 3.0])
forward = a.forward(x)
print(forward)
dvalues = np.array([0.1, 0.2, 0.3, 0.4])
backward = a.backward(dvalues)
print(backward)

print("\nSigmoid activation:")
a = Activation_Sigmoid()
x = np.array([0.0, 1.0, -1.0, 2.0])
forward = a.forward(x)
print(forward)
dvalues = np.array([0.1, 0.2, 0.3, 0.4])
backward = a.backward(dvalues)
print(backward)

print("\nReLU activation:")
a = Activation_ReLU()
x = np.array([1.0, -2.0, 0.0, 3.0])
forward = a.forward(x)
print(forward)
dvalues = np.array([0.1, 0.2, 0.3, 0.4])
backward = a.backward(dvalues)
print(backward)

print("\nSoftmax activation:")
a = Activation_Softmax()
x = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
forward = a.forward(x)
print(forward)
dvalues = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
backward = a.backward(dvalues)
print(backward)

print("\nDense layer backward pass:")
layer = Layer_Dense(3, 2)
x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
dvalues = np.array([[0.1, 0.2], [0.3, 0.4]])
layer.forward(x)
layer.backward(dvalues)
print(layer.dweights)
print(layer.dbiases)
print(layer.dinputs)

print("\nLoss backward pass:")
loss = Loss_CategoricalCrossentropy()
y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
y_true = np.array([0, 1])
dvalues = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
loss.forward(y_pred, y_true)
backward = loss.backward(dvalues, y_true)
print(loss.dinputs) 