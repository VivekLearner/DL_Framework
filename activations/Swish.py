import numpy as np
from core.Module import Module
class Swish(Module):
    def __init__(self):
        self.input = None 
        self.output = None 
        self.sigmoid = None 
    def forward(self, x):
        self.input = x 
        self.sigmoid =  1/(1+np.exp(-x))
        self.output = x * self.sigmoid
        return self.output

    def backward(self, grad):
        sigmoid_der = self.sigmoid * (1-self.sigmoid)
        swish_der = sigmoid_der + self.input *sigmoid_der
        return grad * swish_der

    def parameters(self):
        return []

    def gradients(self):
        return []