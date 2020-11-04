import numpy as np

from layers.Layer import Layer


def exp_running_avg(running, new, gamma=.9):
    return gamma * running + (1. - gamma) * new


class BatchNormalization(Layer):
    def __init__(self, learning_rate=0.1):
        super().__init__()
        self.name = 'BatchNormalization'
        self.learning_rate = learning_rate
        self.running_mean = 0
        self.running_var = 1
        self.gamma = 1
        self.beta = 0
        self.train = True
        self.eps = 0.0000001
        self.forward_cache = None

    def forward(self, X):
        if self.train:
            mu = np.mean(X, axis=0)
            var = np.var(X, axis=0)

            X_norm = (X - mu) / np.sqrt(var + self.eps)
            out = self.gamma * X_norm + self.beta

            self.forward_cache = (X_norm, mu, var, self.gamma, self.beta)
            self.running_mean = exp_running_avg(self.running_mean, mu)
            self.running_var = exp_running_avg(self.running_var, var)
        else:
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * X_norm + self.beta
            self.forward_cache = None

        return out

    def backward(self, X, grad_output):
        X_norm, mu, var, gamma, beta = self.forward_cache

        N, D, _, _ = X.shape

        X_mu = X - mu
        std_inv = 1. / np.sqrt(var + self.eps)

        dX_norm = grad_output * gamma
        dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv ** 3
        dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

        dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
        dgamma = np.sum(grad_output * X_norm, axis=0)
        dbeta = np.sum(grad_output, axis=0)

        self.gamma = self.gamma - self.learning_rate * dgamma
        self.beta = self.beta - self.learning_rate * dbeta

        return dX
