from models.Model import Model

from layers.Dense import Dense
from layers.Conv import Conv
from layers.MaxPool import MaxPool
from layers.Flatten import Flatten
from layers.ReLU import ReLU
from layers.BatchNormalization import BatchNormalization
from layers.Dropout import Dropout


class Lenet5XXX(Model):
    def __init__(self, learning_rate, use_dropout=False):
        super().__init__()
        self.name = 'Lenet5XXX'
        self.learning_rate = learning_rate

        self.network = []
        self.network.append(Conv(3, 6, (5, 5), learning_rate=learning_rate))
        self.network.append(ReLU())
        self.network.append(BatchNormalization())
        self.network.append(MaxPool(2, 2))
        if use_dropout:
            self.network.append(Dropout(0.2))
        self.network.append(Conv(6, 16, (5, 5), learning_rate=learning_rate))
        self.network.append(ReLU())
        self.network.append(BatchNormalization())
        self.network.append(MaxPool(2, 2))
        if use_dropout:
            self.network.append(Dropout(0.3))
        self.network.append(Conv(16, 120, (5, 5), learning_rate=learning_rate))
        self.network.append(ReLU())
        self.network.append(Flatten())
        self.network.append(Dense(120, 84, learning_rate=learning_rate))
        self.network.append(ReLU())
        self.network.append(Dense(84, 10, learning_rate=learning_rate))
