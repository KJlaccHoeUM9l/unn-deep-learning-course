import numpy as np

from layers.PoolingLayer import PoolingLayer


def avgpool(X_col):
    out = np.mean(X_col, axis=0)
    cache = None
    return out, cache


def davgpool(dX_col, dout_col, pool_cache):
    dX_col[:, range(dout_col.size)] = 1. / dX_col.shape[0] * dout_col
    return dX_col


class AvgPool(PoolingLayer):
    def __init__(self, size=2, stride=2):
        super().__init__('AvgPool', size=size, stride=stride)
        self.forward_pool_function = avgpool
        self.backward_pool_function = davgpool
