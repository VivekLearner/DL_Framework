from core.Module import Module


class Sequential(Module):

    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x
    
    def backward(self, grad):
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad

    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params

    def gradients(self):
        grads = []
        for module in self.modules:
            grads.extend(module.gradients())
        return grads
    def regularizable_parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.regularizable_parameters())
        return params
    def add_regularization_gradients(self, reg_gradients):
        index = 0
        for module in self.modules:
            params = module.regularizable_parameters()
            if params:
                for _ in params:
                    module.add_regularization_gradient(
                        reg_gradients[index]
                    )
                    index += 1
    def train(self):
        self.training = True
        for module in self.modules:
            module.train()

    def eval(self):
        self.training = False
        for module in self.modules:
            module.eval()
    