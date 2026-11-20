import numpy as np

from initialization.InitializerBase import InitializerBase

class He(InitializerBase):

    def initialize(self, shape):
        fan_in = shape[0]
        std =np.sqrt(2/fan_in)
        return np.random.randn(*shape)*std