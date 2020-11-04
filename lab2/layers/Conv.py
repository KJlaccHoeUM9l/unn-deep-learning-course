import numpy as np

from typing import Tuple

from layers.Layer import Layer
from layers.utils import im2col_indices, col2im_indices


class Conv(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], stride=1, padding=0, learning_rate=0.1):
        super().__init__()
        self.name = 'Conv'
        self.learning_rate = learning_rate
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.filters = np.random.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]) * 0.01
        self.biases = np.zeros(out_channels)

    def forward(self, X):
        """
        input shape: [batch, in_channels, in_height, in_width]
        output shape: [batch, out_channels, ..., ...]
        """
        n_filters, d_filter, h_filter, w_filter = self.filters.shape
        n_x, d_x, h_x, w_x = X.shape
        h_out = int((h_x - h_filter + 2 * self.padding) / self.stride) + 1
        w_out = int((w_x - w_filter + 2 * self.padding) / self.stride) + 1

        X_col = im2col_indices(X, h_filter, w_filter, padding=self.padding, stride=self.stride)
        W_col = self.filters.reshape(n_filters, -1)

        out = W_col @ X_col + self.biases.reshape(-1, 1)
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)

        return out

    def backward(self, X, dout):
        n_filter, d_filter, h_filter, w_filter = self.filters.shape

        db = np.sum(dout, axis=(0, 2, 3))

        X_col = im2col_indices(X, h_filter, w_filter, padding=self.padding, stride=self.stride)
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dW = dout_reshaped @ X_col.T
        dW = dW.reshape(self.filters.shape)

        W_reshape = self.filters.reshape(n_filter, -1)
        dX_col = W_reshape.T @ dout_reshaped
        dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=self.padding, stride=self.stride)

        self.filters = self.filters - self.learning_rate * dW
        self.biases = self.biases - self.learning_rate * db

        return dX
