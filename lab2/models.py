from layers.Dense import Dense
from layers.Conv import Conv
from layers.ReLU import ReLU
from layers.Sigmoid import Sigmoid
from layers.Tanh import Tanh
from layers.Flatten import Flatten
from layers.MaxPool import MaxPool
from layers.AvgPool import AvgPool
from layers.BatchNormalization import BatchNormalization
from layers.Dropout import Dropout


def get_model(name='simple'):
    learning_rate = 0.1
    network = []
    if name == 'simple':
        network.append(Dense(3072, 100, learning_rate=learning_rate))
        network.append(ReLU())
        network.append(Dense(100, 200, learning_rate=learning_rate))
        network.append(ReLU())
        network.append(Dense(200, 100, learning_rate=learning_rate))
        network.append(ReLU())
        network.append(Dense(100, 10, learning_rate=learning_rate))
    elif name == 'conv':
        network.append(Conv(3, 16, (5, 5), learning_rate=learning_rate))
        network.append(Flatten())
        network.append(Dense(12544, 1000, learning_rate=learning_rate))
        network.append(Sigmoid())
        network.append(Dense(1000, 10, learning_rate=learning_rate))
    elif name == 'maxpool':
        network.append(Conv(3, 8, (5, 5), learning_rate=learning_rate))
        network.append(Tanh())
        network.append(MaxPool(2, 2))
        network.append(Flatten())
        network.append(Dense(1568, 10, learning_rate=learning_rate))
    elif name == 'lenet5':
        network.append(Conv(3, 6, (5, 5), learning_rate=learning_rate))
        network.append(AvgPool(2, 2))
        network.append(Tanh())
        network.append(Conv(6, 16, (5, 5), learning_rate=learning_rate))
        network.append(AvgPool(2, 2))
        network.append(Tanh())
        network.append(Conv(16, 120, (5, 5), learning_rate=learning_rate))
        network.append(Tanh())
        network.append(Flatten())
        network.append(Dense(120, 84, learning_rate=learning_rate))
        network.append(Tanh())
        network.append(Dense(84, 10, learning_rate=learning_rate))
    elif name == 'test':
        network.append(Conv(3, 6, (5, 5), learning_rate=learning_rate))
        network.append(ReLU())
        network.append(BatchNormalization())
        network.append(MaxPool(2, 2))
        # network.append(Dropout(0.2))
        network.append(Conv(6, 16, (5, 5), learning_rate=learning_rate))
        network.append(ReLU())
        network.append(BatchNormalization())
        network.append(MaxPool(2, 2))
        # network.append(Dropout(0.3))
        network.append(Conv(16, 120, (5, 5), learning_rate=learning_rate))
        network.append(ReLU())
        network.append(Flatten())
        network.append(Dense(120, 84, learning_rate=learning_rate))
        network.append(ReLU())
        network.append(Dense(84, 10, learning_rate=learning_rate))
    else:
        raise ValueError('incorrect net name')
    return network
