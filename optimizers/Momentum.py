import numpy as np

from optimizers.OptimizerBase import OptimizerBase
class Momentum(OptimizerBase):
    def __init__(self,learning_rate=0.01,beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.velocity = None 
    def step(self, parameters, gradients):

        if self.velocity is None:
            self.velocity = [np.zeros_like(param) for param in parameters]

        for i,(param,grad) in enumerate(zip(parameters,gradients)):
            self.velocity[i] = self.beta*self.velocity[i] + grad 
            param -=(self.learning_rate)*self.velocity[i]