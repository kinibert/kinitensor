# 🧠 KiniTensor
KiniTensor is a simple deep learning library for educational purposes.
It provides a high-level, easy to use API for building and training neural networks.
Please note that this project is only for educational purposes and is not intended for production use.
It is NOT optimized for speed or efficiency.

## Installation
You can install KiniTensor using pip:

```bash
pip install git+https://github.com/kinibert/kinitensor
```

## Features
- **Modular Architecture:** A neural network is defined by a list of Layer Objects.
- Activation Functions, Loss Function and Initialisation Functions can be easily changed.
- **Multi-Layer Perceptron:** The type of neural network available in KiniTensor
- **Batch Processing:** Batch Size can be configured to speed up training/execution
- **High-Level API:** Training and executing models with a user-friendly high-level API
- **Implementation in NumPy:** The complete logic part was implemented from scratch using only NumPy.
  The scikit-learn dependency is only used for getting training data and evaluating an example model on the MNIST Dataset.

## Tutorial
To create a Neural Network (Multi-Layer Perception is currently the only supported type), you need to import:
- Activation function (imported with `import kinitensor.activation_functions`)
- Loss function (imported with `import kinitensor.loss_functions`)
- Initialisation function (imported with `import kinitensor.initialisation_functions`)
- Layers (imported with `import kinitensor.layers`)
- NeuralNetwork Object (imported with `from kinitensor.models import NeuralNetwork`)

First, create a Neural Network. It takes its definition as a parameter in form of a list.
The list defines the architecture of the Network, where each list item corresponds to a layer in the network.

**Important:**
- The number of output neurons of a layer MUST be the same as the number of input neurons from the previous layer
- The last layer MUST be an `OutputLayer`

## MNIST Results
**Note:** To reproduce these results, clone this repository, install all dependencies and run `test.py` 

**Architecture:**  
Input (784 Inputs) -> Dense Layer (LeakyReLU Activation, 128 Neurons) -> Dense Layer (LeakyReLU Activation, 32 Neurons) -> Dense Layer (Sigmoid Activation, 10 Neurons) -> Output (10 Outputs)

| Metric | Result |
| :--- | :--- |
| Learning Rate | 0.6 |
| Epochs | 30 |
| Batch Size | 64 |
| Parameters | 13.008 |
| **Total accuracy** | **98,04%**  |

## Example
Here is a small Neural Network with 2 hidden layers and around 13.000 parameters:

```python
# Imports here!!

nn = NeuralNetwork([
    DenseLayer(784, 16, XavierInitialisation()),
    ActivationLayer(LeakyReLU()),
    DenseLayer(16, 16, XavierInitialisation()),
    ActivationLayer(LeakyReLU()),
    DenseLayer(16, 10, XavierInitialisation()),
    ActivationLayer(Sigmoid()),
    OutputLayer(MeanSquaredLoss())
])
```

Now we can train it:

```python
x_train = # Train Inputs
y_train = # Train Outputs
nn.train(x_train, y_train)
```

Now we can test it on new data:
```python
x_test = # Test Inputs
y_test = # Test Outputs
y_pred = nn.predict(x_test)
```

## License
This project is licensed under the [LGPL-3.0-or-later](https://www.gnu.org/licenses/lgpl-3.0.html) license. You are free to use, modify, and distribute this software under the terms of this license.
