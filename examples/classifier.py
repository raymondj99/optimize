from optimize.core.module import Module, BCELoss
from optimize.utils.optimizer import SGD
from optimize import Tensor, Mode, set_mode
import numpy as np

# define a linear regression module
class MyModel(Module):
    def __init__(self):
        super().__init__()
        self.register_param(w1=Tensor(np.random.randn()))
        self.register_param(w2=Tensor(np.random.randn()))
        self.register_param(b=Tensor(np.random.randn()))
    
    def forward(self, x):
        w1 = self.params['w1'].repeat(x.shape[0])
        w2 = self.params['w2'].repeat(x.shape[0])
        b = self.params['b'].repeat(x.shape[0])
        y = w1 * Tensor(x[:, 0]) + w2 * Tensor(x[:, 1]) + b
        return y

if __name__ == "__main__":
    # set to backwards automatic differentation mode
    set_mode(Mode.BACKWARD)

    # define a seed for reproducibility
    np.random.seed(42)

    # we chose a simple classification model with decision boundary being 4x1 - 3x2 > 0
    x = np.random.randn(200, 2)
    y = ((x[:, 0] - 3 * x[:, 1]) > 0).astype(float)

    # define loss function and optimizer
    model = MyModel()
    criterion = BCELoss()
    opt = SGD(model.parameters(), lr=0.1, momentum=0.9)

    # training
    for epoch in range(100):
        outputs = model(x)
        targets = Tensor(y)
        loss = criterion(targets, outputs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.data)
