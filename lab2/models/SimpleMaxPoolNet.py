from models.Model import Model

from layers.Dense import Dense
from layers.Conv import Conv
from layers.Flatten import Flatten
from layers.Tanh import Tanh
from layers.MaxPool import MaxPool


class SimpleMaxPoolNet(Model):
    def __init__(self, learning_rate):
        super().__init__()
        self.name = 'SimpleMaxPoolNet'
        self.learning_rate = learning_rate

        self.network = []
        self.network.append(Conv(3, 8, (5, 5), learning_rate=learning_rate))
        self.network.append(Tanh())
        self.network.append(MaxPool(2, 2))
        self.network.append(Flatten())
        self.network.append(Dense(1568, 10, learning_rate=learning_rate))
