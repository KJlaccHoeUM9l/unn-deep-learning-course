import numpy as np

from layers.Layer import Layer


class Dropout(Layer):
    def __init__(self, p):
        super().__init__()
        self.name = 'Dropout'
        self.p = p
        self.__mask_cache = None

    def forward(self, input):
        self.__mask_cache = np.random.binomial(1, self.p, size=input.shape) / self.p
        out = input * self.__mask_cache
        return out

    def backward(self, input, grad_output):
        return grad_output * self.__mask_cache
