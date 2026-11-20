import numpy as np


class Dataset:

    def __init__(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]