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

        self.__input_reshaped = None
        self.__input_col = None
        self.__max_idx_cache = None

    def forward(self, input):
        n, d, h, w = input.shape
        h_out = int((h - self.size) / self.stride) + 1
        w_out = int((w - self.size) / self.stride) + 1

        self.__input_reshaped = input.reshape(n * d, 1, h, w)
        self.__input_col = im2col_indices(self.__input_reshaped, self.size, self.size, padding=0, stride=self.stride)

        out, self.__max_idx_cache = self.forward_pool_function(self.__input_col)
        out = out.reshape(h_out, w_out, n, d)
        out = out.transpose(2, 3, 0, 1)

        return out

    def backward(self, input, grad_output):
        n, d, w, h = input.shape

        grad_input_col = np.zeros_like(self.__input_col)
        grad_output_col = grad_output.transpose(2, 3, 0, 1).ravel()

        grad_input = self.backward_pool_function(grad_input_col, grad_output_col, self.__max_idx_cache)
        grad_input = col2im_indices(grad_input, (n * d, 1, h, w), self.size, self.size, padding=0, stride=self.stride)
        grad_input = grad_input.reshape(input.shape)

        return grad_input
