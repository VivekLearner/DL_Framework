import numpy as np

from core.Module import Module
class BatchNorm(Module):
    def __init__(self,num_features,eps=1e-5,momentum=0.9):
        super().__init__()
        self.num_features = num_features
        self.eps = eps 
        self.momentum = momentum

        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.x = None 
        self.x_hat = None 
        self.mean = None 
        self.var = None 
        self.dgamma = np.zeros(num_features)
        self.dbeta = np.ones(num_features)
        self.training=True
    def forward(self,x):
        if self.training:
            self.x = x 
            self.mean = np.mean(x,axis=0)
            self.var = np.var(x,axis=0)
            x_centered = x - self.mean 
            self.x_hat = (x_centered/np.sqrt(self.var +self.eps))
            self.running_mean = (self.momentum*self.running_mean + (1-self.momentum)*self.mean) 
            self.running_var = (self.momentum*self.running_var + (1-self.momentum)*self.var) 
        else:
            self.x_hat = (x-self.running_mean)/np.sqrt(self.running_var+self.eps)
        return self.gamma * self.x_hat + self.beta
    def backward(self,grad):
        n=self.x.shape[0]
        self.dgamma = np.sum(grad*self.x_hat,axis=0)
        self.dbeta = np.sum(grad,axis=0)
        x_centered = self.x - self.mean 
        std_inv = 1 / np.sqrt(self.var + self.eps)
        dx_hat = grad * self.gamma 
        dvar = np.sum(dx_hat*x_centered*-0.5*std_inv**3,axis=0)
        dmean = np.sum(dx_hat*-std_inv,axis=0)
        dx = (dx_hat*std_inv) + ((dvar*2*x_centered)/n) + (dmean/n)
        return dx
    def parameters(self):
        return [
            self.gamma,
            self.beta
        ]

    def gradients(self):
        return [
            self.dgamma,
            self.dbeta
        ]
    def regularizable_parameters(self):
        return []
    def add_regularization_gradient(self, reg_grad):
        pass