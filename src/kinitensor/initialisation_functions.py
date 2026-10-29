from abc import ABC, abstractmethod
import numpy as np

rng = np.random.default_rng(seed=67)

class InitialisationFunction(ABC):
    @abstractmethod
    def __call__(self, input_size, output_size):
        pass


class XavierInitialisation(InitialisationFunction):
    def __call__(self, input_size, output_size):
        g = np.sqrt(6 / (input_size + output_size))
        return rng.uniform(low=-g, high=g, size=(input_size, output_size))