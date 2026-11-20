import numpy as np
from core.Module import Module
class MSE(Module):
    def __init__(self):
        self.y_true=None 
        self.y_pred = None
    def forward(self,y_pred,y_true):
        self.y_true=y_true
        self.y_pred = y_pred
        return np.mean((y_pred-y_true)**2)
    def backward(self,grad):
        n = self.y_true.size 
        return (2/n) *(self.y_pred - self.y_true)
    def parameters(self):
        return []
    def gradients(self):
        return []