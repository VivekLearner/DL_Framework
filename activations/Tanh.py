import numpy as np
from core.Module import Module
class Tanh(Module):
    def __init__(self):
        self.input = None 
    def forward(self, x):
        self.output =  np.tanh(x)
        return self.output

    def backward(self, grad):
        tanh_der = 1 - (self.output**2)
        return grad * tanh_der

    def parameters(self):
        return []

    def gradients(self):
        return []