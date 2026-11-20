from schedulers.SchedulerBase import SchedulerBase
import numpy as np

class WarmupCosine(SchedulerBase):
    def __init__(self,optimizer,warmup_epochs,total_epochs,start_lr=0.0,eta_min=0.0):
        super().__init__(optimizer)
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.start_lr = start_lr
        self.epoch = 0
        self.eta_min = eta_min
        self.max_lr = optimizer.learning_rate
    def step(self):
        self.epoch +=1 
        if self.epoch <= self.warmup_epochs:
            progress  = self.epoch / self.warmup_epochs
            self.optimizer.learning_rate = (self.start_lr) + (self.max_lr -  self.start_lr)*progress
        else:
            progress  = (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            self.optimizer.learning_rate = (self.eta_min + 0.5 * (self.max_lr - self.eta_min)*(1 + np.cos(np.pi*progress)))
                    