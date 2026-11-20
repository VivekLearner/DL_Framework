import numpy as np
from core.Module import Module
class Sigmoid(Module):
    def __init__(self):
        self.input = None 
    def forward(self, x):
        self.output =  1/(1+np.exp(-x))
        return self.output

    def backward(self, grad):
        sigmoid_der = self.output * (1-self.output)
        return grad * sigmoid_der

    def parameters(self):
        return []

    def gradients(self):
        return []