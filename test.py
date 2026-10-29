from kinitensor.layers import ActivationLayer, OutputLayer, DenseLayer
from kinitensor.loss_functions import MeanSquaredLoss, BinaryCrossEntropy
from kinitensor.initialisation_functions import XavierInitialisation
from kinitensor.activation_functions import Sigmoid, ReLU, LeakyReLU, Tanh
from kinitensor.models import NeuralNetwork
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

TRAIN_TEST_RATIO = 0.8

print("Downloading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')

x = mnist.data.astype(np.int32) / 255.0
y = np.eye(10)[mnist.target.astype(np.int32)]

test_start_index = int(len(x)*TRAIN_TEST_RATIO)
x_train, x_test = x[:test_start_index], x[test_start_index:]
y_train, y_test = y[:test_start_index], y[test_start_index:]

model = NeuralNetwork([
    DenseLayer(784, 128, XavierInitialisation()),
    ActivationLayer(LeakyReLU(alpha=0.05)),
    DenseLayer(128, 32, XavierInitialisation()),
    ActivationLayer(LeakyReLU(alpha=0.05)),
    DenseLayer(32, 10, XavierInitialisation()),
    ActivationLayer(Sigmoid()),
    OutputLayer(MeanSquaredLoss())
])


model.train(x_train, y_train, lr=0.6, epochs=30, batch_size=64)

y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print()
print("Classification Report:")
print(classification_report(y_true, y_pred))
print()
print("\n🧩 Verwechslungsmatrix:")
print(confusion_matrix(y_true, y_pred))