import numpy as np

from optimizers.OptimizerBase import OptimizerBase
class RMSProp(OptimizerBase):
    def __init__(self,learning_rate=0.01,eps = 1e-8,beta = 0.9):
        super().__init__(learning_rate)
        self.eps = eps 
        self.beta = beta 
        self.Gs = None
    def step(self, parameters, gradients):

        if self.Gs is None:
            self.Gs = [np.zeros_like(param) for param in parameters]

        for i,(param,grad) in enumerate(zip(parameters,gradients)):
            self.Gs[i] = self.beta * self.Gs[i] + (1-self.beta)*(grad**2)
            param -= self.learning_rate*grad/(np.sqrt(self.Gs[i]+self.eps))