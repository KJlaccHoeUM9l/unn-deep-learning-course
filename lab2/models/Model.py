import numpy as np


class Model:
    def __init__(self):
        self.name = 'BaseNet'
        self.learning_rate = 0.1
        self.network = None

    def train_on_batch(self, X, y, loss_function):
        layer_activations = self.__forward(X)
        layer_inputs = [X] + layer_activations
        logits = layer_activations[-1]

        # Compute the loss and the initial gradient
        loss = loss_function.compute_loss(logits, y)
        loss_grad = loss_function.compute_grad(logits, y)

        current_grad = loss_grad
        for layer_ind in reversed(range(len(self.network))):
            current_grad = self.network[layer_ind].backward(layer_inputs[layer_ind], current_grad)

        return np.mean(loss)

    def predict(self, X):
        logits = self.__forward(X)[-1]
        return logits.argmax(axis=-1)

    def __forward(self, X):
        input = X
        activations = []
        for layer in self.network:
            input = layer.forward(input)
            activations.append(input)
        return activations
