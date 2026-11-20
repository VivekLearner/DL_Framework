import numpy as np


class OptimizerBase:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, parameters, gradients):
        raise NotImplementedError

    def zero_grad(self, gradients):
        for grad in gradients:
            grad[...] = 0