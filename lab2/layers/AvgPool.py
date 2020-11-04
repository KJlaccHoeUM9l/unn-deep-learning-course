import numpy as np

from layers.PoolingLayer import PoolingLayer


def forward_avg_pool(input_col):
    return np.mean(input_col, axis=0), None


def backward_avg_pool(grad_input_col, grad_output_col, _):
    grad_input_col[:, range(grad_output_col.size)] = 1. / grad_input_col.shape[0] * grad_output_col
    return grad_input_col


class AvgPool(PoolingLayer):
    def __init__(self, size=2, stride=2):
        super().__init__('AvgPool', size=size, stride=stride)
        self.forward_pool_function = forward_avg_pool
        self.backward_pool_function = backward_avg_pool
