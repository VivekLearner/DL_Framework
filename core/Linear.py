import numpy as np
from core.Module import Module
class Linear(Module):
    def __init__(self,input_size,output_size,initializer=None):
        super().__init__()
        if initializer is not None:
            self.weights = initializer.initialize(
                (input_size, output_size)
            )
        else:
            self.weights = np.random.randn(
                input_size,
                output_size
            )
        self.bias = np.zeros(output_size)
        self.input_size = input_size
        self.output_size = output_size
        
        self.input = None 
        self.weight_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)
    def forward(self,x):
        self.input = x
        z = x @ self.weights + self.bias
        return z
    def backward(self,grad):
        self.weight_grad =  self.input.T @ grad 
        self.bias_grad = np.sum(grad,axis=0)
        return grad @ self.weights.T
    def parameters(self):
        return [self.weights,self.bias]
    def gradients(self):
        return [
            self.weight_grad,
            self.bias_grad
        ]
    def regularizable_parameters(self):
        return [self.weights]
    def add_regularization_gradient(self, reg_grad):
        self.weight_grad += reg_grad
