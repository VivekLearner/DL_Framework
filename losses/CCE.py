import numpy as np
from core.Module import Module


class CCE(Module):

    def __init__(self, eps=1e-8):
        super().__init__()

        self.eps = eps
        self.y_true = None
        self.y_pred = None

    def forward(self, y_pred, y_true):

        self.y_true = y_true.astype(int)
        self.y_pred = np.clip(y_pred,self.eps,1 - self.eps)
        n = y_pred.shape[0]
        correct_probs = self.y_pred[
            np.arange(n),
            self.y_true
        ]
        loss = -np.log(correct_probs)
        return np.mean(loss)

    def backward(self):

        n = self.y_pred.shape[0]
        grad = np.zeros_like(self.y_pred)
        grad[np.arange(n),self.y_true] = -1 / self.y_pred[np.arange(n),self.y_true]
        return grad / n

    def parameters(self):
        return []

    def gradients(self):
        return []