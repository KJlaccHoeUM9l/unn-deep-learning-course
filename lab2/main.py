import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange

from DataManager import DataManager
from models.Lenet5XXX import Lenet5XXX
from models.NoName import NoName

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
    dataset_root_path = '/home/agladyshev/Documents/UNN/DL/Datasets/cifar-10-batches-py'
    weights_root_path = './weights'
    weights_path = None#os.path.join(weights_root_path, 'Lenet5XXX_weights_51_acc.tar')

    network = Lenet5XXX(learning_rate=0.1)
    # network = NoName(learning_rate=0.1)
    if weights_path is not None:
        network.load_state_dict(weights_path)

    data_manager = DataManager(dataset_root_path, like_images=True)
    X_train, X_val, y_train, y_val = data_manager.get_train_data()
    X_test, y_test = data_manager.get_test_data()

    train_log = []
    val_log = []
    for epoch in range(10):
        for x_batch, y_batch in iterate_minibatches(X_train, y_train, batchsize=128, shuffle=True):
            loss = network.train_on_batch(x_batch, y_batch, SoftmaxCCE)

        train_log.append(np.mean(network.predict(X_train, batch_step=32) == y_train))
        val_log.append(np.mean(network.predict(X_val, batch_step=32) == y_val))

        network.save_state_dict(val_log[-1], weights_root_path)

        print("Epoch", epoch)
        print("Loss: ", loss)
        print("Train accuracy: ", train_log[-1])
        print("Val accuracy: ", val_log[-1])

    print("Test accuracy: ", np.mean(network.predict(X_test, batch_step=32) == y_test))

    plt.plot(train_log, label='train accuracy')
    plt.plot(val_log, label='val accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
