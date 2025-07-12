# Neural Networks from Scratch - Learning Journey

This repository contains my implementation of neural networks following the "Neural Networks from Scratch" textbook by Harrison Kinsley and Daniel Kukieła. I've built everything from the ground up, learning the fundamentals of how neural networks actually work.

## What I've Learned (Chapters 1-7)

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

### Chapter 6: Optimization - Finding Better Weights

This chapter introduced me to the concept of optimization - how to actually improve the network's performance by finding better weights and biases.

**Key Concepts:**
- **Random Search**: A simple but inefficient optimization method
- **Weight Adjustment**: Randomly tweaking weights and keeping improvements
- **Loss Minimization**: The goal is to find weights that minimize loss

**What I implemented:**
- Random search optimization in `main_spiral.py`
- Only keeping weight changes that improve (lower) the loss
- Tracking the best weights found so far
- Iterative improvement over many attempts

**Example from the spiral dataset:**
```python
# Random search optimization
for iteration in range(10000):
    # Randomly adjust weights
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    
    # Test new weights
    # ... forward pass ...
    
    # Keep if better, otherwise revert
    if loss < lowest_loss:
        print(f'New set of weights found, iteration: {iteration} loss: {loss:.7f} acc: {accuracy}')
        # Save best weights
        lowest_loss = loss
    else:
        # Revert to previous best weights
        dense1.weights = best_weights.copy()
```

**Results:**
- Started with loss ~1.098 and accuracy ~33%
- After 10,000 iterations: loss ~1.028 and accuracy ~44%
- Shows that even random search can improve performance

**Why this matters:**
- This is the foundation of training neural networks
- Real training uses smarter methods (gradient descent), but the principle is the same
- Demonstrates that neural networks can learn by finding better weights

### Chapter 7: Derivatives - The Math Behind Optimization

Chapter 7 focused on the mathematical foundation needed for proper optimization: derivatives and calculus.

**Key Concepts:**
- **Derivatives**: How much a function changes when its input changes
- **Chain Rule**: How to find derivatives of composite functions
- **Gradients**: The direction of steepest increase in multi-variable functions

**What I learned:**
- How to calculate derivatives of simple functions
- The relationship between derivatives and optimization
- Why derivatives are crucial for efficient training (gradient descent)

**Simple derivative example:**
```python
# For function f(x) = 2x²
def f(x):
    return 2 * x**2

# Derivative is f'(x) = 4x
def derivative_f(x):
    return 4 * x

# At x = 3: f'(3) = 12
# This tells us the function is increasing rapidly at x = 3
```

**Why derivatives matter:**
- They tell us which direction to adjust weights for maximum improvement
- This leads to gradient descent (much better than random search)
- Foundation for backpropagation (coming in later chapters)
- Makes training thousands of times more efficient

**Connection to neural networks:**
- Instead of random weight changes, we can calculate exactly how to change weights
- Derivatives show us the "slope" of the loss function
- We can follow the slope downhill to minimize loss faster

## Classification Scripts

I've created several main scripts that demonstrate the neural network on different classification tasks:

### 1. `main_spiral.py` - Textbook Example
- Uses the spiral dataset from NNFS
- Demonstrates the complete network from Chapters 1-6
- Shows single neuron, layer operations, full network, and optimization
- Includes random search optimization that improves accuracy from ~33% to ~44%

### 2. `main_vertical.py` - Simple Classification
- Uses vertical data from NNFS
- 2-class classification problem
- Simple and clean implementation

### 3. `main_behaviour.py` - Personality Classification
- **Dataset**: `Datasets/personality_datasert.csv`
- **Task**: Classify introvert vs extrovert based on personality traits
- **Features**: 7 features (time spent alone, social events, etc.)
- **Architecture**: 3-layer network (7 → 16 → 16 → 2)
- **Results**: ~59% accuracy

### 4. `main_depression.py` - Medical Classification
- **Dataset**: `Datasets/me_cfs_vs_depression_dataset.csv`
- **Task**: 3-class classification (Depression, ME/CFS, Both)
- **Features**: 14 features (sleep quality, brain fog, pain scores, etc.)
- **Architecture**: 3-layer network (14 → 16 → 16 → 3)
- **Results**: ~46% accuracy (reasonable for 3-class problem)

## Complete Network Example

By Chapter 6, I can build a complete neural network with optimization:

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

# With optimization (Chapter 6)
# After 10,000 iterations of random search:
# Loss: ~1.028, Accuracy: ~0.44
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
├── main_scripts/
│   ├── main_spiral.py     # Textbook spiral data example
│   ├── main_vertical.py   # Simple 2-class classification
│   ├── main_behaviour.py  # Personality classification (introvert/extrovert)
│   └── main_depression.py # Medical classification (Depression/ME/CFS/Both)
├── Datasets/
│   ├── personality_datasert.csv           # Personality traits dataset
│   └── me_cfs_vs_depression_dataset.csv   # Medical diagnosis dataset
├── tests/
│   ├── test_neuron.py     # Neuron tests
│   ├── test_layers.py     # Layer tests
│   ├── test_activation_functions.py  # Activation function tests
│   └── test_losses.py     # Loss function tests
├── explore_vertical_data.py  # Data exploration script
└── README.md              # This file
```

## Running the Code

```bash
# Install dependencies
pip install numpy matplotlib nnfs pytest pandas

# Run textbook examples
python main_scripts/main_spiral.py
python main_scripts/main_vertical.py

# Run real-world classification tasks
python main_scripts/main_behaviour.py
python main_scripts/main_depression.py

# Run all tests
python -m pytest tests/ -v
```

## Dataset Information

### Personality Dataset
- **Purpose**: Classify introvert vs extrovert personality types
- **Features**: 7 features including time spent alone, social event attendance, stage fear, etc.
- **Samples**: ~2,900 individuals
- **Classes**: 2 (Introvert, Extrovert)

### Medical Dataset  
- **Purpose**: Classify between Depression, ME/CFS, and Both conditions
- **Features**: 14 features including sleep quality, brain fog, physical pain, stress levels, etc.
- **Samples**: ~1,000 patients
- **Classes**: 3 (Depression, ME/CFS, Both)
