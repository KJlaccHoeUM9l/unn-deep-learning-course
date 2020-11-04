import numpy as np

from layers.Layer import Layer
from layers.utils import im2col_indices, col2im_indices


class PoolingLayer(Layer):
    def __init__(self, name, size=2, stride=2):
        super().__init__()
        self.name = name
        self.size = size
        self.stride = stride

        self.forward_pool_function = None
        self.backward_pool_function = None

    def forward(self, input):
        n, d, h, w = input.shape
        h_out = int((h - self.size) / self.stride) + 1
        w_out = int((w - self.size) / self.stride) + 1

        X_reshaped = input.reshape(n * d, 1, h, w)
        X_col = im2col_indices(X_reshaped, self.size, self.size, padding=0, stride=self.stride)

        out, pool_cache = self.forward_pool_function(X_col)

        out = out.reshape(h_out, w_out, n, d)
        out = out.transpose(2, 3, 0, 1)

        return out

    def backward(self, input, grad_output):
        n, d, w, h = input.shape

        X_reshaped = input.reshape(n * d, 1, h, w)
        X_col = im2col_indices(X_reshaped, self.size, self.size, padding=0, stride=self.stride)
        dX_col = np.zeros_like(X_col)
        dout_col = grad_output.transpose(2, 3, 0, 1).ravel()

        _, pool_cache = self.forward_pool_function(X_col)
        dX = self.backward_pool_function(dX_col, dout_col, pool_cache)

        dX = col2im_indices(dX_col, (n * d, 1, h, w), self.size, self.size, padding=0, stride=self.stride)
        dX = dX.reshape(input.shape)

        return dX
