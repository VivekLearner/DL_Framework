import numpy as np

from initialization.InitializerBase import InitializerBase

class Xavier(InitializerBase):

    def initialize(self, shape):
        fan_in = shape[0]
        fan_out = shape[1]

        limit = np.sqrt(6/(fan_in+fan_out))
        return np.random.uniform(-limit,limit,size=shape)