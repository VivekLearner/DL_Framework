import numpy as np
from core.Module import Module
class BCE(Module):
    def __init__(self,eps=1e-8):
        self.eps = eps
        self.y_true=None 
        self.y_pred = None
    def forward(self,y_pred,y_true):
        self.y_true=y_true
        self.y_pred = np.clip(y_pred,self.eps,1-self.eps)
        loss=  -(y_true*np.log(y_pred)
                 + (1-y_true)*np.log(1-self.y_pred))
        return np.mean(loss)
    def backward(self):
        n= self.y_true.size 
        return ((self.y_pred - self.y_true)/(self.y_pred * (1-self.y_pred)*n))
    def parameters(self):
        return []
    def gradients(self):
        return []