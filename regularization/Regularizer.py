class Regularizer:

    def penalty(self, parameters):
        raise NotImplementedError

    def gradients(self, parameters):
        raise NotImplementedError