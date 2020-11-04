from models.Model import Model

from layers.Dense import Dense
from layers.Conv import Conv
from layers.MaxPool import MaxPool
from layers.Flatten import Flatten
from layers.ReLU import ReLU
from layers.BatchNormalization import BatchNormalization
from layers.Dropout import Dropout


class NoName(Model):
    def __init__(self, learning_rate):
        super().__init__()
        self.name = 'NoName'
        self.learning_rate = learning_rate

        self.network = []
        self.network.append(Conv(3, 64, (3, 3), learning_rate=learning_rate))
        self.network.append(ReLU())
        self.network.append(MaxPool(2, 1))
        self.network.append(BatchNormalization())

        self.network.append(Conv(64, 128, (3, 3), learning_rate=learning_rate))
        self.network.append(ReLU())
        self.network.append(MaxPool(2, 1))
        self.network.append(BatchNormalization())

        self.network.append(Conv(128, 256, (3, 3), learning_rate=learning_rate))
        self.network.append(ReLU())
        self.network.append(MaxPool(2, 1))
        self.network.append(BatchNormalization())

        self.network.append(Flatten())

        self.network.append(Dense(135424, 128))
        self.network.append(ReLU())
        self.network.append(Dropout(0.7))
        self.network.append(BatchNormalization())

        self.network.append(Dense(128, 512))
        self.network.append(ReLU())
        self.network.append(Dropout(0.7))
        self.network.append(BatchNormalization())

        self.network.append(Dense(512, 1024))
        self.network.append(ReLU())
        self.network.append(Dropout(0.7))
        self.network.append(BatchNormalization())

        self.network.append(Dense(1024, 10))
