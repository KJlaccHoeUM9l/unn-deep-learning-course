import numpy as np

from layers.PoolingLayer import PoolingLayer


def maxpool(X_col):
    max_idx = np.argmax(X_col, axis=0)
    out = X_col[max_idx, range(max_idx.size)]
    return out, max_idx


def dmaxpool(dX_col, dout_col, pool_cache):
    dX_col[pool_cache, range(dout_col.size)] = dout_col
    return dX_col


class MaxPool(PoolingLayer):
    def __init__(self, size=2, stride=2):
        super().__init__('MaxPool', size=size, stride=stride)
        self.forward_pool_function = maxpool
        self.backward_pool_function = dmaxpool
