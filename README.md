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

## Classification Scripts

I've created several main scripts that demonstrate the neural network on different classification tasks:

### 1. `main_spiral.py` - Textbook Example
- Uses the spiral dataset from NNFS
- Demonstrates the complete network from Chapters 1-5
- Shows single neuron, layer operations, and full network

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
