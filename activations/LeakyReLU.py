import numpy as np
from core.Module import Module
class LeakyReLU(Module):
    def __init__(self,alpha = 0.01):
        self.alpha = alpha
        self.input = None 
    def forward(self,x):
        self.input = x 
        return np.where(x>0,x,self.alpha*x)
    def backward(self,grad):
        leakyrelu_grad = np.where(self.input>0,1.0,self.alpha)
        return leakyrelu_grad*grad 

    def parameters(self):
        return []
    def gradients(self):
        return []