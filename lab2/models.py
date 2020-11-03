from Dense import Dense
from Conv import Conv
from ReLU import ReLU
from Sigmoid import Sigmoid
from Tanh import Tanh
from Flatten import Flatten


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
    else:
        raise ValueError('incorrect net name')
    return network
