import numpy as np
from core.Module import Module
class MSE(Module):
    def __init__(self):
        self.y_true=None 
        self.y_pred = None
    def forward(self,y_pred,y_true):
        self.y_true=y_true
        self.y_pred = y_pred
        return np.abs(y_pred-y_true)
    def backward(self,grad):
        n = self.y_true.size 
        return np.sign(self.y_pred-self.y_true) / n 
    def parameters(self):
        return []
    def gradients(self):
        return []