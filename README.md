# Optimize
`Optimize` is a python automatic differentiation library. Optimize offers an accessible interface similar to PyTorch, and utilities to perform gradient-based optimization.

`Optimize` is written in pure Python with a dependency on NumPy.

To get started, clone the repository and install using pip:
```
(.venv) $ pip install .
```

## Forward Mode
To utilize forward mode auto-diff, first import the `optimize.core.tensor` module and set the mode to `FORWARD` mode, which is the default.  This will allow you to compute the forward accumulation of composite functions.

```python
>>> from optimize import Tensor, Mode, set_mode
>>> import optimize.core.functional as F
>>> set_mode(Mode.FORWARD)
>>> x = Tensor([[1., 2.], [3., 4.]])
>>> y = 2 * x + 1
>>> z = -y / (x ** 3)
>>> w = F.cos((F.exp(z) + F.exp(-z)) / 2)
>>> w
optimize.Tensor(
[[-0.80037009  0.36072269]
 [ 0.51156054  0.53194201]]
)
>>> w.grad # should be the derivatives of w w.r.t x
array([[-4.20404488e+01,  4.27363350e-01],
       [ 4.17169950e-02,  8.86701846e-03]])
```
you can also use ``seed`` parameter to specify the derivative direction.

```python
>>> x = Tensor([1., 2.], seed=[1, 0])
>>> y = F.sin(3 * x + 1)
>>> y.grad
array([-1.96093086,  0.        ])

```

## Reverse Mode
For reverse mode, the construction of the tensors are the same. The difference is that the gradient won't be accumulated until the `Tensor.backward()` method is called.

```python
>>> from optimize import Tensor, Mode, set_mode
>>> import optimize.core.functional as F
>>> set_mode(Mode.BACKWARD)
>>> x = Tensor([[1, 2], [3, 4]])
>>> y = F.cos(3 * (x ** 2) + 4 * x + 1)
>>> z = y.mean()
>>> z
optimize.Tensor(-0.48065530173082893)
>>> z.backward()
>>> z.grad
array(1.)
>>> y.grad
array([[0.25, 0.25],
       [0.25, 0.25]])
>>> x.grad
array([[-2.47339562, -3.34662255],
       [-4.09812238, -5.78780076]])
```

## Constructing Simple ML Models
With backwards auto-diff, you can utilize the backwards accumulation of gradients to construct models for machine learning applications. For ease of use, construct your own model using the `optimize.core.Module` class, which allows you to register parameters.

```python
import numpy as np
from optimize import Tensor. Mode, set_mode
from optimize.core.module import Module, BCELoss
from optimize.utils.optimizer import SGD

class LinearModel(Module):
   def __init__(self):
      super().__init__()
      self.register_param(w=Tensor(np.random.randn()))
      self.register_param(b=Tensor(np.random.randn()))

   def forward(self, x):
      w = self.params['w'].repeat(x.shape[0])
      b = self.params['b'].repeat(x.shape[0])
      y = w * x + b # x is a R1 vector
      return y
```

Then utilize the SGD optimizer and a loss function to train the model.
```python
set_mode(Mode.BACKWARD)
model = MyModel()
np.random.seed(42)
x = Tensor([1, 2, 3, 4])
y = Tensor([3, 5, 7, 9])
criterion = BCELoss()
opt = SGD(model.parameters(), lr=0.1, momentum=0.9)
for i in range(10):
   outputs = model(x)
   loss = criterion(y, outputs)
   opt.zero_grad()
   loss.backward()
   opt.step()

print(model.params['w'], model.params['b'])
# optimize.Tensor(61.66494274391248) optimize.Tensor(20.456688280268263)
```

To see a more concrete example of backwards autodiff being used to train a model on the iris dataset, see `examples/iris.py`.

```
(.venv) python3.11 examples/iris.py
Epoch [1/10], Loss: 0.1355
Epoch [2/10], Loss: 0.0161
Epoch [3/10], Loss: 0.0078
Epoch [4/10], Loss: 0.0066
Epoch [5/10], Loss: 0.0028
Epoch [6/10], Loss: 0.0023
Epoch [7/10], Loss: 0.0033
Epoch [8/10], Loss: 0.0021
Epoch [9/10], Loss: 0.0037
Epoch [10/10], Loss: 0.0018
```