from models.Model import Model

from layers.Dense import Dense
from layers.Conv import Conv
from layers.Flatten import Flatten
from layers.Sigmoid import Sigmoid


class SimpleConvNet(Model):
    def __init__(self, learning_rate):
        super().__init__()
        self.name = 'SimpleConvNet'
        self.learning_rate = learning_rate

        self.network = []
        self.network.append(Conv(3, 16, (5, 5), learning_rate=learning_rate))
        self.network.append(Flatten())
        self.network.append(Dense(12544, 1000, learning_rate=learning_rate))
        self.network.append(Sigmoid())
        self.network.append(Dense(1000, 10, learning_rate=learning_rate))
