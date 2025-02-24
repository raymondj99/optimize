import numpy as np

class Optimizer:
    def __init__(self, params: dict):
        self.params = params

    def step(self):
        pass

    def zero_grad(self):
        for k, v in self.params.items():
            v.grad = np.zeros_like(v.data)
        

class SGD(Optimizer):
    def __init__(self, params: dict, lr=1e-3, momentum=0):
        super().__init__(params)
        self.lr = lr
        self.b = {k: None for k, v in params.items()}
        self.momentum = momentum
    
    def step(self):
        for k, v in self.params.items():
            g = v.grad
            if self.b[k] is not None:
                self.b[k] = self.momentum * self.b[k] + g
            else:
                self.b[k] = g
            v.data = v.data - self.lr * self.b[k]
