class SchedulerBase:

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        raise NotImplementedError