import numpy as np

from layers.Layer import Layer


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'Sigmoid'

    def forward(self, input):
        return 1. / (1. + np.exp(-input))

    def backward(self, input, grad_output):
        y = self.forward(input)
        return y * (1. - y) * grad_output
