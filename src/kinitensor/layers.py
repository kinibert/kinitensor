from abc import ABC, abstractmethod
import numpy as np

class BaseLayer(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

class ActivationLayer(BaseLayer):
    def __init__(self, activation_function):
        self.activation_function = activation_function
        self.output = None
        self.input = None

    def forward(self, x):
        self.input = x
        self.output = self.activation_function(x)
        return self.output

    def backward(self, gradient, lr):
        return gradient * self.activation_function.derivative(self.output)

class DenseLayer(BaseLayer):
    def __init__(self, input_size, output_size, init_function):
        self.w = init_function(input_size, output_size)
        self.b = np.zeros((1, output_size))
        self.output = None
        self.input = None

    def forward(self, x):
        self.input = x
        self.output = x @ self.w + self.b
        return self.output

    def backward(self, gradient, lr):
        new_gradient = gradient @ self.w.T
        self.w -= lr * (self.input.T @ gradient)
        self.b -= lr * np.sum(gradient, axis=0, keepdims=True)
        return new_gradient

class OutputLayer(BaseLayer):
    def __init__(self, loss_function):
        self.loss_function = loss_function
        self.output = None

    def forward(self, x):
        self.output = x
        return x

    def loss(self, expected_output):
        return self.loss_function(self.output, expected_output)

    def backward(self, expected_output, lr):
        return self.loss_function.derivative(self.output, expected_output)