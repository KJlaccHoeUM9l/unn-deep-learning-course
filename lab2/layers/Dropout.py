import numpy as np

from layers.Layer import Layer


class Dropout(Layer):
    def __init__(self, p):
        super().__init__()
        self.name = 'Dropout'
        self.p = p
        self.mask_cache = None

    def forward(self, X):
        self.mask_cache = np.random.binomial(1, self.p, size=X.shape) / self.p
        out = X * self.mask_cache
        return out

    def backward(self, input, grad_output):
        return grad_output * self.mask_cache
