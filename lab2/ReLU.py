import numpy as np

from Layer import Layer


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'ReLU'

    def forward(self, input):
        output = np.maximum(0, input)
        return output

    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output * relu_grad
