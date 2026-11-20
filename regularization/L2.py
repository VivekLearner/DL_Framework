import numpy as np 
from regularization.Regularizer import Regularizer

class L2(Regularizer):
    def __init__(self,strength=0.01):
        self.strength = strength

    def penalty(self, parameters):
        total = 0.0
        for param in parameters:
            total +=np.sum(param**2)
        return self.strength * total
    def gradients(self, parameters):

        gradients = []

        for param in parameters:
            gradients.append(
                2 * self.strength * param
            )

        return gradients
        
