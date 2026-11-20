import numpy as np
from core.Module import Module
class GeLU(Module):
    def __init__(self):
        self.input = None 
    def forward(self,x):
        self.input = x 
        c = np.sqrt(2/np.pi)
        inner = c * (x + 0.044715*(x**3))
        return 0.5 * x * (1 +np.tanh(inner))
    def backward(self,grad):
        x = self.input 
        c =np.sqrt(2/np.pi)
        inner = c * (x + 0.044715*(x**3))
        tanh_in = np.tanh(inner)
        inner_der = c *(1+ 3*0.044715*(x**2))
        gelu_der = (0.5 * (1+tanh_in) + 0.5 * x * (1-tanh_in**2)* inner_der)
        return gelu_der * grad


    def parameters(self):
        return []
    def gradients(self):
        return []