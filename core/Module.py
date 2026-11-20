class Module:

    def __init__(self):
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def parameters(self):
        return []

    def gradients(self):
        return []
    def regularizable_parameters(self):
        return []
    def add_regularization_gradient(self, reg_grad):
        pass
    def train(self):
        self.training = True

    def eval(self):
        self.training = False