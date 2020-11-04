class Layer:
    def __init__(self):
        self.name = 'layer'

    def forward(self, input):
        pass

    def backward(self, input, grad_output):
        pass

    def get_state(self):
        return self.name, None

    def set_state(self, params_dict):
        pass

    def __str__(self):
        return self.name
