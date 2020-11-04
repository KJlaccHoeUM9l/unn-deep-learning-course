import os
import pickle
import numpy as np


class Model:
    def __init__(self):
        self.name = 'BaseNet'
        self.learning_rate = 0.1
        self.network = None

        self.__accuracy = 0.

    def train_on_batch(self, X, y, loss_function):
        layer_activations = self.__forward(X)
        layer_inputs = [X] + layer_activations
        logits = layer_activations[-1]

        loss = loss_function.compute_loss(logits, y)
        loss_grad = loss_function.compute_grad(logits, y)

        current_grad = loss_grad
        for layer_ind in reversed(range(len(self.network))):
            current_grad = self.network[layer_ind].backward(layer_inputs[layer_ind], current_grad)

        return np.mean(loss)

    def predict(self, X, batch_step=1000):
        n_iterations = 1 + int(len(X) / batch_step)
        result = np.array([], dtype=int)
        for iteration in range(n_iterations):
            start_ind = batch_step * iteration
            result = np.append(result, self.__predict_on_batch(X[start_ind:min(start_ind + batch_step, len(X)), :]))
        return result

    def save_state_dict(self, accuracy, save_root_path='.'):
        if accuracy >= self.__accuracy:
            self.__accuracy = accuracy
            model_state = {}
            for layer_ind, layer in enumerate(self.network):
                layer_name, params_dict = layer.get_state()
                model_state[layer_ind] = {'layer_name': layer_name, 'params_dict': params_dict}

            with open(os.path.join(save_root_path, '{}_weights.tar'.format(self.name)), 'wb') as handle:
                pickle.dump(model_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state_dict(self, weights_path):
        with open(weights_path, 'rb') as handle:
            model_state = pickle.load(handle)
        for layer_ind, layer_state_dict in model_state.items():
            layer_params = layer_state_dict['params_dict']
            if layer_params is not None:
                self.network[layer_ind].set_state(layer_params)

    def __forward(self, X):
        input = X
        activations = []
        for layer in self.network:
            input = layer.forward(input)
            activations.append(input)
        return activations

    def __predict_on_batch(self, X):
        logits = self.__forward(X)[-1]
        return logits.argmax(axis=-1)
