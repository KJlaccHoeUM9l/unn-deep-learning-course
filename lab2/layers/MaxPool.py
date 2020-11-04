import numpy as np

from layers.PoolingLayer import PoolingLayer


def forward_max_pool(input_col):
    max_idx = np.argmax(input_col, axis=0)
    out = input_col[max_idx, range(max_idx.size)]
    return out, max_idx


def backward_max_pool(grad_input_col, grad_output_col, max_idx_cache):
    grad_input_col[max_idx_cache, range(grad_output_col.size)] = grad_output_col
    return grad_input_col


class MaxPool(PoolingLayer):
    def __init__(self, size=2, stride=2):
        super().__init__('MaxPool', size=size, stride=stride)
        self.forward_pool_function = forward_max_pool
        self.backward_pool_function = backward_max_pool
