import numpy as np 
from regularization.Regularizer import Regularizer

class L1(Regularizer):
    def __init__(self,strength=0.01):
        self.strength = strength

    def penalty(self, parameters):
        total = 0.0
        for param in parameters:
            total +=np.sum(np.abs(param))
        return self.strength * total
    def gradients(self, parameters):
        return [self.strength *np.sign(param) for param in parameters]
        
