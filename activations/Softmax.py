import numpy as np
from core.Module import Module
class Softmax(Module):
    def __init__(self):
        self.output = None 
    def forward(self, x):
        x_shifted = x - np.max(x,axis=-1,keepdims=True)
        exp_x = np.exp(x_shifted)
        self.output = exp_x/np.sum(exp_x,axis=-1,keepdims=True)
        return self.output

    def backward(self, grad):
        softmax_der = self.output*(grad-np.sum(grad*self.output,axis=-1,keepdims=True))
        return softmax_der

    def parameters(self):
        return []

    def gradients(self):
        return []