import numpy as np

from layers.Layer import Layer
from layers.utils import xavier_initialization


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        super().__init__()
        self.name = 'Dense'
        self.learning_rate = learning_rate

        self.weights = xavier_initialization((input_units, output_units), input_units, output_units)
        self.biases = np.zeros(output_units)

    def forward(self, input):
        """
        input shape: [batch, input_units]
        output shape: [batch, output units]
        """
        return np.dot(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)

        grad_weights = np.dot(input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)

        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input

    def get_state(self):
        return self.name, {'weights': self.weights, 'biases': self.biases}

    def set_state(self, params_dict):
        self.weights = params_dict['weights']
        self.biases = params_dict['biases']
