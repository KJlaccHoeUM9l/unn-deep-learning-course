import numpy as np

from layers.Layer import Layer


def running_average(running, new, gamma=.9):
    return gamma * running + (1. - gamma) * new


class BatchNormalization(Layer):
    def __init__(self, train_mode=True, learning_rate=0.1):
        super().__init__()
        self.name = 'BatchNormalization'
        self.train_mode = train_mode
        self.learning_rate = learning_rate

        self.gamma = 1
        self.beta = 0

        self.__eps = 0.0000001
        self.__running_mean = 0
        self.__running_var = 1
        self.__input_norm = None
        self.__batch_mean = None
        self.__batch_var = None

    def forward(self, input):
        self.__batch_mean = np.mean(input, axis=0)
        self.__batch_var = np.var(input, axis=0)
        self.__running_mean = running_average(self.__running_mean, self.__batch_mean)
        self.__running_var = running_average(self.__running_var, self.__batch_var)

        if self.train_mode:
            self.__input_norm = (input - self.__batch_mean) / np.sqrt(self.__batch_var + self.__eps)
            out = self.gamma * self.__input_norm + self.beta
        else:
            input_norm = (input - self.__running_mean) / np.sqrt(self.__running_var + self.__eps)
            out = self.gamma * input_norm + self.beta

        return out

    def backward(self, input, grad_output):
        n, d, _, _ = input.shape

        input_mu = input - self.__batch_mean
        std_inv = 1. / np.sqrt(self.__batch_var + self.__eps)

        grad_input_norm = grad_output * self.gamma
        grad_var = np.sum(grad_input_norm * input_mu, axis=0) * -.5 * std_inv ** 3
        grad_mu = np.sum(grad_input_norm * -std_inv, axis=0) + grad_var * np.mean(-2. * input_mu, axis=0)

        grad_input = (grad_input_norm * std_inv) + (grad_var * 2 * input_mu / n) + (grad_mu / n)
        grad_gamma = np.sum(grad_output * self.__input_norm, axis=0)
        grad_beta = np.sum(grad_output, axis=0)

        self.gamma = self.gamma - self.learning_rate * grad_gamma
        self.beta = self.beta - self.learning_rate * grad_beta

        return grad_input

    def get_state(self):
        return self.name, {'gamma': self.gamma, 'beta': self.beta}

    def set_state(self, params_dict):
        self.gamma = params_dict['gamma']
        self.beta = params_dict['beta']
