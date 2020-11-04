import numpy as np

from typing import Tuple

from layers.Layer import Layer
from layers.utils import xavier_initialization, im2col_indices, col2im_indices


class Conv(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], stride=1, padding=0, learning_rate=0.1):
        super().__init__()
        self.name = 'Conv'
        self.learning_rate = learning_rate
        self.stride = stride
        self.padding = padding

        self.filters = xavier_initialization((out_channels, in_channels, kernel_size[0], kernel_size[1]), in_channels, out_channels)
        self.biases = np.zeros(out_channels)

        self.input_col = None

    def forward(self, input):
        """
        input shape: [batch, in_channels, in_height, in_width]
        output shape: [batch, out_channels, ..., ...]
        """
        n_filters, d_filter, h_filter, w_filter = self.filters.shape
        n_x, d_x, h_x, w_x = input.shape
        h_out = int((h_x - h_filter + 2 * self.padding) / self.stride) + 1
        w_out = int((w_x - w_filter + 2 * self.padding) / self.stride) + 1

        self.input_col = im2col_indices(input, h_filter, w_filter, padding=self.padding, stride=self.stride)
        filters_col = self.filters.reshape(n_filters, -1)

        out = np.dot(filters_col, self.input_col) + self.biases.reshape(-1, 1)
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)

        return out

    def backward(self, input, grad_output):
        n_filters, d_filter, h_filter, w_filter = self.filters.shape

        grad_biases = np.sum(grad_output, axis=(0, 2, 3))

        grad_output_reshaped = grad_output.transpose(1, 2, 3, 0).reshape(n_filters, -1)
        grad_filters = np.dot(grad_output_reshaped, self.input_col.T)
        grad_filters = grad_filters.reshape(self.filters.shape)

        filters_reshaped = self.filters.reshape(n_filters, -1)
        grad_input_col = np.dot(filters_reshaped.T, grad_output_reshaped)
        grad_input = col2im_indices(grad_input_col, input.shape, h_filter, w_filter, padding=self.padding, stride=self.stride)

        self.filters = self.filters - self.learning_rate * grad_filters
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input
