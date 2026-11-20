from schedulers.SchedulerBase import SchedulerBase
import numpy as np

class CosineAnnealing(SchedulerBase):
    def __init__(self,optimizer,T_max,eta_min=0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.epoch = 0
        self.eta_min = eta_min
        self.initial_lr = optimizer.learning_rate
    def step(self):
        self.epoch +=1 
        self.optimizer.learning_rate = ( self.eta_min + 0.5 * (self.initial_lr - self.eta_min)*(1 + np.cos((np.pi*self.epoch)/self.T_max)))
        