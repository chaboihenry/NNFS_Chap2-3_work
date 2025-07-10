# Neural Networks from Scratch - Learning Journey

This repository contains my implementation of neural networks following the "Neural Networks from Scratch" textbook by Harrison Kinsley and Daniel Kukieła. I've built everything from the ground up, learning the fundamentals of how neural networks actually work.

## What I've Learned (Chapters 1-5)

### Chapter 1-2: The Building Blocks - Neurons

I started with the most basic component: a single neuron. A neuron takes inputs, multiplies them by weights, adds a bias, and produces an output.

**Key Files:**
- `neuron/neuron.py` - My single neuron implementation
- `tests/test_neuron.py` - Tests to make sure it works correctly

**What it does:**
```python
# Single neuron example
inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3
output = 35.7  # inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
```

Then I learned how to create layers of neurons that process the same inputs in parallel, which is much more powerful.

### Chapter 3: Adding Layers

I learned how to stack neurons into layers and connect them together. This is where things start getting interesting.

**Key Files:**
- `layer/layer.py` - General layer class (list of neurons)
- `layer/layer_dense.py` - Dense layer implementation
- `tests/test_layers.py` - Tests for layer functionality

**What I built:**
- Dense layers that can handle multiple inputs and outputs
- Proper weight initialization (small random values)
- Forward pass through layers
- Batch processing (multiple samples at once)

The dense layer became my main building block for creating neural networks.

### Chapter 4: Activation Functions

This was a game-changer. I learned that neurons need activation functions to introduce non-linearity and make the network actually learn complex patterns.

**Key Files:**
- `activation_function/activation_function.py` - All activation functions
- `tests/test_activation_functions.py` - Tests for each activation function

**Activation Functions I implemented:**
- **Step**: Simple on/off function (0 or 1)
- **Linear**: Identity function (output = input)
- **Sigmoid**: S-shaped curve that squashes values between 0 and 1
- **ReLU**: Most popular - max(0, x) - simple but effective
- **Softmax**: Converts outputs to probabilities (sums to 1)

**Why they matter:**
- Without activation functions, neural networks are just linear transformations
- Different activation functions are better for different tasks
- ReLU is fast and works well for hidden layers
- Softmax is perfect for the final layer in classification

### Chapter 5: Loss Functions

Now I learned how to measure how wrong my network's predictions are. This is crucial for training.

**Key Files:**
- `losses/losses.py` - Loss function implementations
- `tests/test_losses.py` - Tests for loss calculations

**What I built:**
- Base `Loss` class for common functionality
- `Loss_CategoricalCrossentropy` for classification tasks
- Handles both categorical labels and one-hot encoded labels
- Prevents numerical issues with clipping

**Why loss matters:**
- Loss tells us how far off our predictions are
- We need this to train the network (reduce loss = improve predictions)
- Categorical crossentropy is perfect for classification problems

## Complete Network Example

By Chapter 5, I can build a complete neural network:

```python
# Create network
dense1 = Layer_Dense(2, 3)           # 2 inputs, 3 neurons
activation1 = Activation_ReLU()      # ReLU activation
dense2 = Layer_Dense(3, 3)           # 3 inputs, 3 neurons  
activation2 = Activation_Softmax()   # Softmax for probabilities

# Forward pass
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Calculate loss and accuracy
loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
accuracy = np.mean(np.argmax(activation2.output, axis=1) == y)

print(f"Loss: {loss}")      # ~1.098
print(f"Accuracy: {accuracy}")  # ~0.333 (random guessing)
```

## Project Structure

```
├── neuron/
│   ├── neuron.py          # Single neuron implementation
│   └── __init__.py
├── layer/
│   ├── layer.py           # General layer class
│   ├── layer_dense.py     # Dense layer implementation
│   └── __init__.py
├── activation_function/
│   ├── activation_function.py  # All activation functions
│   └── __init__.py
├── losses/
│   ├── losses.py          # Loss function implementations
│   └── __init__.py
├── tests/
│   ├── test_neuron.py     # Neuron tests
│   ├── test_layers.py     # Layer tests
│   ├── test_activation_functions.py  # Activation function tests
│   └── test_losses.py     # Loss function tests
├── main1.py               # Complete example (Chapters 1-5) with spiral_data
├── main2.py               # Future: External dataset example
└── README.md              # This file
```

## Key Insights

1. **Start Simple**: Single neurons → layers → activation functions → loss functions
2. **Test Everything**: I wrote tests for each component to make sure they work correctly
3. **Modular Design**: Each component is in its file, making it easy to understand and modify
4. **Textbook Accuracy**: My implementations produce the same outputs as the textbook examples
5. **Foundation Matters**: Understanding these basics makes everything else much clearer

## Running the Code

```bash
# Install dependencies
pip install numpy matplotlib nnfs pytest

# Run the complete example
python main1.py

# Run all tests
python -m pytest tests/ -v
```
