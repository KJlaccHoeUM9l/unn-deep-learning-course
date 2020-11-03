import numpy as np

from Layer import Layer


class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'Tanh'

    def forward(self, input):
        return np.tanh(input)

    def backward(self, input, grad_output):
        y = self.forward(input)
        return (1. - y ** 2) * grad_output
