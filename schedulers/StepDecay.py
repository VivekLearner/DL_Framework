from schedulers.SchedulerBase import SchedulerBase

class StepDecay(SchedulerBase):
    def __init__(self,optimizer,step_size=10,gamma=0.1):
        super().__init__(optimizer)
        self.step_size=step_size
        self.gamma = gamma
        self.epoch = 0 
        self.initial_lr = optimizer.learning_rate
    def step(self):
        self.epoch +=1 
        decay_steps = self.epoch // self.step_size

        self.optimizer.learning_rate = (self.initial_lr*(self.gamma)**decay_steps)