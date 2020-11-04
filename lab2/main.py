import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange

from DataManager import DataManager
from models import get_model
from loss_functions import SoftmaxCCE


def forward(network, X):
    activations = []
    input = X
    for layer in network:
        input = layer.forward(input)
        activations.append(input)
    return activations


def predict(network, X):
    logits = forward(network, X)[-1]
    return logits.argmax(axis=-1)


def train(network, X, y, loss_function):
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations  # layer_input[i] is an input for network[i]
    logits = layer_activations[-1]

    # Compute the loss and the initial gradient
    loss = loss_function.compute_loss(logits, y)
    loss_grad = loss_function.compute_grad(logits, y)

    current_grad = loss_grad
    for layer_ind in reversed(range(len(network))):
        current_grad = network[layer_ind].backward(layer_inputs[layer_ind], current_grad)

    return np.mean(loss)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def main():
    np.random.seed(47)

    data_manager = DataManager('/home/agladyshev/Documents/UNN/DL/Datasets/cifar-10-batches-py/data_batch_1', like_images=True)
    X_train, X_val, y_train, y_val = data_manager.get_train_data()

    network = get_model('maxpool')

    train_log = []
    val_log = []

    for epoch in range(25):
        for x_batch, y_batch in iterate_minibatches(X_train, y_train, batchsize=32, shuffle=True):
            loss = train(network, x_batch, y_batch, SoftmaxCCE)

        train_log.append(np.mean(predict(network, X_train) == y_train))
        val_log.append(np.mean(predict(network, X_val) == y_val))

        print("Epoch", epoch)
        print("Loss: ", loss)
        print("Train accuracy: ", train_log[-1])
        print("Val accuracy: ", val_log[-1])

    plt.plot(train_log, label='train accuracy')
    plt.plot(val_log, label='val accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
