import numpy as np
from core.Module import Module
class ReLU(Module):
    def __init__(self):
        self.input = None 
    def forward(self,x):
        self.input = x 
        return np.maximum(0,x)
    def backward(self,grad):
        relu_grad = (self.input>0).astype(float)
        return relu_grad*grad 

    def parameters(self):
        return []
    def gradients(self):
        return []