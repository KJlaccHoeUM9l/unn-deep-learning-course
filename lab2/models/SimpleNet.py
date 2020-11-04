from models.Model import Model

from layers.Dense import Dense
from layers.ReLU import ReLU


class SimpleNet(Model):
    def __init__(self, learning_rate):
        super().__init__()
        self.name = 'SimpleNet'
        self.learning_rate = learning_rate

        self.network = []
        self.network.append(Dense(3072, 100, learning_rate=self.learning_rate))
        self.network.append(ReLU())
        self.network.append(Dense(100, 200, learning_rate=self.learning_rate))
        self.network.append(ReLU())
        self.network.append(Dense(200, 100, learning_rate=self.learning_rate))
        self.network.append(ReLU())
        self.network.append(Dense(100, 10, learning_rate=self.learning_rate))
