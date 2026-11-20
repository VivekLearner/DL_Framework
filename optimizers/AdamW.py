import numpy as np

from optimizers.OptimizerBase import OptimizerBase
class AdamW(OptimizerBase):
    def __init__(self,learning_rate=0.01,eps = 1e-8,beta1 = 0.9,beta2=0.99,wd = 0.01):
        super().__init__(learning_rate)
        self.eps = eps 
        self.beta1= beta1 
        self.beta2=beta2
        self.wd = wd
        self.Ms = None
        self.Vs = None 
        self.t=0
    def step(self, parameters, gradients):

        if self.Ms is None and self.Vs is None:
            self.Ms = [np.zeros_like(param) for param in parameters]
            self.Vs = [np.zeros_like(param) for param in parameters]

        self.t +=1
        for i,(param,grad) in enumerate(zip(parameters,gradients)):
            self.Ms[i] = (self.beta1*self.Ms[i]) + (1-self.beta1)*grad
            self.Vs[i] = (self.beta2*self.Vs[i]) + (1-self.beta2)*(grad**2)
            m_hat = self.Ms[i]/(1-self.beta1**self.t)
            v_hat = self.Vs[i]/(1-self.beta2**self.t)
            adam =(self.learning_rate *m_hat/(np.sqrt(v_hat+self.eps)))
            weight_decay = (self.learning_rate*self.wd*param)
            param -=(adam+weight_decay)
    