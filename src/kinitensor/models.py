import joblib
from tqdm import tqdm
import numpy as np
from datetime import datetime

class NeuralNetwork():
    def __init__(self, layers):
        self.layers = layers

    def train(self, x, y, epochs=10, batch_size=32, lr=0.1, beta=0.95):
        end_index = (len(x) // batch_size) * batch_size
        for epoch in tqdm(range(epochs), desc="Model training in progress"):
            for i in range(0, end_index, batch_size):
                x_batch = x[i:i+batch_size]
                y_batch = y[i : i + batch_size]
                for layer in self.layers:
                    x_batch = layer.forward(x_batch)
                grad = y_batch
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, lr)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
