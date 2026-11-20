import numpy as np

from optimizers.OptimizerBase import OptimizerBase
class SGD(OptimizerBase):
    def step(self, parameters, gradients):
        for param,grad in zip(parameters,gradients):
            param -=self.learning_rate * grad 
