# Optimize
`Optimize` is a python automatic differentiation library. Optimize offers an accessible interface similar to PyTorch, and utilities to perform gradient-based optimization.

`Optimize` is written in pure Python with a dependency on NumPy.

To get started, first install it using pip:
```
(.venv) $ pip install .
```

## Forward Mode
For forward mode, first import the `optimmize.core.tensor` module and set the mode to `FORWARD` mode, which is the default.  This will allow you to compute the forward accumulation of composite functions.

```python
>>> from optimize import Tensor
>>> import optimize.core.functional as F
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