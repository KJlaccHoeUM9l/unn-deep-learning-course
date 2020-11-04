from models.Model import Model

from layers.Dense import Dense
from layers.Conv import Conv
from layers.Flatten import Flatten
from layers.Tanh import Tanh
from layers.AvgPool import AvgPool


class Lenet5(Model):
    def __init__(self, learning_rate):
        super().__init__()
        self.name = 'Lenet5'
        self.learning_rate = learning_rate

        self.network = []
        self.network.append(Conv(3, 6, (5, 5), learning_rate=learning_rate))
        self.network.append(AvgPool(2, 2))
        self.network.append(Tanh())
        self.network.append(Conv(6, 16, (5, 5), learning_rate=learning_rate))
        self.network.append(AvgPool(2, 2))
        self.network.append(Tanh())
        self.network.append(Conv(16, 120, (5, 5), learning_rate=learning_rate))
        self.network.append(Tanh())
        self.network.append(Flatten())
        self.network.append(Dense(120, 84, learning_rate=learning_rate))
        self.network.append(Tanh())
        self.network.append(Dense(84, 10, learning_rate=learning_rate))
