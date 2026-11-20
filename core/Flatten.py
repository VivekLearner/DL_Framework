from core.Module import Module
class Flatten(Module):

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)

    def parameters(self):
        return []

    def gradients(self):
        return []