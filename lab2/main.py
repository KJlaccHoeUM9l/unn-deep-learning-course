import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange

from DataManager import DataManager
from models.SimpleNet import SimpleNet
from models.SimpleConvNet import SimpleConvNet
from models.SimpleMaxPoolNet import SimpleMaxPoolNet
from models.Lenet5 import Lenet5
from models.Lenet5XXX import Lenet5XXX

from loss_functions import SoftmaxCCE


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

    # network = SimpleNet(learning_rate=0.1)
    # network = SimpleConvNet(learning_rate=0.1)
    # network = SimpleMaxPoolNet(learning_rate=0.1)
    # network = Lenet5(learning_rate=0.1)
    network = Lenet5XXX(learning_rate=0.1)

    data_manager = DataManager('/home/agladyshev/Documents/UNN/DL/Datasets/cifar-10-batches-py/data_batch_1', like_images=True)
    X_train, X_val, y_train, y_val = data_manager.get_train_data()

    train_log = []
    val_log = []
    for epoch in range(10):
        for x_batch, y_batch in iterate_minibatches(X_train, y_train, batchsize=32, shuffle=True):
            loss = network.train_on_batch(x_batch, y_batch, SoftmaxCCE)

        train_log.append(np.mean(network.predict(X_train) == y_train))
        val_log.append(np.mean(network.predict(X_val) == y_val))

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
