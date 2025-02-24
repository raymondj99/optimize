from optimize.core.module import Module, BCELoss
from optimize.utils.optimizer import SGD
from optimize import Tensor, Mode, set_mode
import optimize.core.functional as F

from sklearn import datasets
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.register_param(w=Tensor(np.random.randn(out_features, in_features)))
        self.register_param(b=Tensor(np.random.randn(1, 1)))

    def forward(self, x):
        b = self.params['b']
        y = F.matvecmul(self.params['w'], x) + b
        return y

if __name__ == "__main__":
    # set to backwards automatic differentiation
    set_mode(Mode.BACKWARD)

    # define a seed for reproducibility
    np.random.seed(42)

    # train on sklearn Iris toy dataset
    iris = datasets.load_iris()
    x = np.array(iris.data)
    y = np.array(iris.target)

    # define a loss function and optmizer
    model = Linear(4, 1)
    criterion = BCELoss()
    opt = SGD(model.parameters(), lr=1e-2)

    # training
    for epoch in range(10):
        epoch_loss = 0

        for i in range(x.shape[0]):
            inputs = Tensor(x[epoch, :].reshape(-1, 1))
            targets = Tensor(y[epoch].reshape(-1, 1))

            outputs = model(inputs)
            loss = criterion(targets, outputs)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.data
    
        print(f"Epoch [{epoch+1}/10], Loss: {epoch_loss / x.shape[0]:.4f}")