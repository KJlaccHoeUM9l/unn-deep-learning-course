import numpy as np

from Layer import Layer


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'Flatten'

    def forward(self, input):
        n_samples, n_channels, height, width = input.shape
        return input.reshape((n_samples, n_channels * height * width))

    def backward(self, input, grad_output):
        return grad_output.reshape(input.shape)
