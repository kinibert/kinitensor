import numpy as np
from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

class Linear(ActivationFunction):
    def __call__(self, x):
        return x

    def derivative(self, x):
        return 1

class Sigmoid(ActivationFunction):
    def __call__(self, x):
        np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        sigmoid_x = self.__call__(x)
        return sigmoid_x * (1 - sigmoid_x)

class ReLU(ActivationFunction):
    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)

class LeakyReLU(ActivationFunction):
    def __init__(self, alpha = 0.01):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x > 0, x, x * self.alpha)

    def derivative(self, x):
        return np.where(x > 0, 1, self.alpha)

class Tanh(ActivationFunction):
    def __call__(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
