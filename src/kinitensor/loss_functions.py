from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def __call__(self, output, expected_output):
        pass

    @abstractmethod
    def derivative(self, output, expected_output):
        pass


class MeanPowerLoss(LossFunction):
    def __init__(self, power):
        self.power = power

    def __call__(self, output, expected_output):
        return np.mean(np.pow(output - expected_output, self.power))

    def derivative(self, output, expected_output):
        return (self.power / output.size) * (output - expected_output)


class MeanSquaredLoss(MeanPowerLoss):
    def __init__(self):
        super().__init__(2)


class BinaryCrossEntropy(LossFunction):
    def __call__(self, output, expected_output):
        output = np.clip(output, 1e-15, 1.0 - 1e-15)
        return np.mean(-(expected_output * np.log(output) + (1.0 - expected_output) * np.log(1.0 - output)))

    def derivative(self, output, expected_output):
        output = np.clip(output, 1e-15, 1.0 - 1e-15)
        return (-(expected_output / output) + ((1.0 - expected_output) / (1.0 - output))) / output.size
