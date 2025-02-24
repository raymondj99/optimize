from optimize.core.tensor import Tensor
from optimize.core.mode import Mode, AutoDiffMode
import numpy as np

from typing import Union

class Layer():
    '''
    Base class for all following functional classes to inherit from
    
    Note: when a functional class is called by a function, it will return its automatically call method with the corresponding arguments 
    '''
    def __init__(self):
        self.desc = 'optimize.Layer'
        self.mode = AutoDiffMode.get_mode()

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        return self.forward(*args)

    def __str__(self):
        return self.desc

    def __repr__(self):
        return self.desc

class Power(Layer):
    def __init__(self, p):
        super().__init__()
        self.desc = 'optimize.Layer.Power'
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        '''
        Perform the operation of rasing the Tensor object to the power of p
    
        Parameters
        ----------
        x : Tensor
        p : float
    
        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after power operation
    
        '''
        y_data = np.power(x.data.copy(), self.p)
        if self.mode == Mode.FORWARD:
            y_grad = self.p * np.power(x.data.copy(), self.p - 1) * x.grad
            y = Tensor(y_data, grad=y_grad)
        elif self.mode == Mode.BACKWARD:
            y = Tensor(y_data)
            y.dependency = [x]
            y.layer = self
        return y

class TensorSum(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.TensorSum'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        '''
        Perform the operation of adding two Tensor objects

        Parameters
        ----------
        x : Tensor
        y : Tensor

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after addition
        
        '''
        assert x.shape == y.shape
        s_data = x.data + y.data
        if self.mode == Mode.FORWARD:
            s_grad = x.grad + y.grad
            s = Tensor(s_data, s_grad)
        elif self.mode == Mode.BACKWARD:
            s = Tensor(s_data)
            s.dependency = [x, y]
            s.layer = self
        return s

    def backward(self, x, y, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after addition
        '''
        x_grad = g.copy()
        y_grad = g.copy()
        x.backward(x_grad)
        y.backward(y_grad)

class TensorProd(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.TensorProd'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        '''
        Perform the operation of multiplicating two Tensor objects
        
        Parameters
        ----------
        x : Tensor
        y : Tensor

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after multiplication
        
        '''
        assert x.shape == y.shape
        p_data = x.data * y.data
        if self.mode == Mode.FORWARD:
            p_grad = x.grad * y.data + x.data * y.grad
            p = Tensor(p_data, p_grad)
        elif self.mode == Mode.BACKWARD:
            p = Tensor(p_data)
            p.dependency = [x, y]
            p.layer = self
        return p

    def backward(self, x, y, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after multiplication
        '''
        x_grad = y.data * g
        y_grad = x.data * g
        x.backward(x_grad)
        y.backward(y_grad)

class TensorInv(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.TensorInv'

    def forward(self, x: Tensor) -> Tensor:
        '''
        Perform the operation of taking the inverse of a Tensor object
        
        Parameters
        ----------
        x : Tensor

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after taking its inverse
        '''
        i_data = 1. / x.data
        if self.mode == Mode.FORWARD:
            i_grad = -1. / (x.data ** 2) * x.grad
            i = Tensor(i_data, i_grad)
        elif self.mode == Mode.BACKWARD:
            i = Tensor(i_data)
            i.dependency = [x]
            i.layer = self
        return i

    def backward(self, x, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after taking the inverse
        '''
        grad = (-1. / x.data) * (1. / x.data) * g
        x.backward(grad)

class NumProd(Layer):
    def __init__(self, num):
        super().__init__()
        self.desc = 'optimize.Layer.NumProd'
        if type(num) == list:
            self.num = np.array(num)
        else:
            self.num = num

    def forward(self, x: Tensor) -> Tensor:
        '''
        Perform the operation of multiplying a Tensor object by number(s)

        Parameters
        ----------
        x : Tensor
        y : int, float, list, or np.ndarray

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after mutiplication with number(s)

        '''
        s_data = x.data * self.num
        if self.mode == Mode.FORWARD:
            s_grad = x.grad * self.num
            s = Tensor(s_data, s_grad)
        elif self.mode == Mode.BACKWARD:
            s = Tensor(s_data)
            s.dependency = [x]
            s.layer = self
        return s

    def backward(self, x, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after multiplying by number(s)
        '''
        grad = self.num * g
        x.backward(grad)

class NumSum(Layer):
    def __init__(self, num):
        super().__init__()
        self.desc = 'optimize.Layer.NumSum'
        if type(num) == list:
            self.num = np.array(num)
        else:
            self.num = num

    def forward(self, x: Tensor) -> Tensor:
        '''
        Perform the operation of adding number(s) to a Tensor object 

        Parameters
        ----------
        x : Tensor
        y : int, float, list, or np.ndarray

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after adding number(s) to it

        '''
        s_data = x.data + self.num
        if self.mode == Mode.FORWARD:
            s_grad = x.grad
            s = Tensor(s_data, s_grad)
        elif self.mode == Mode.BACKWARD:
            s = Tensor(s_data)
            s.dependency = [x]
            s.layer = self
        return s

    def backward(self, x, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after adding to number(s)
        '''
        x.backward(g)

class Exp(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.Exp'

    def forward(self, x: Tensor) -> Tensor:
        '''
        Perform the operation of computing the exponential of a Tensor object

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after computing the exponential

        '''
        s_data = np.exp(x.data)
        if self.mode == Mode.FORWARD:
            s_grad = np.exp(x.data) * x.grad
            s = Tensor(s_data, s_grad)
        elif self.mode == Mode.BACKWARD:
            s = Tensor(s_data)
            s.dependency = [x]
            s.layer = self
        return s

    def backward(self, x, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after taking the exponential
        '''
        grad = g * np.exp(x.data)
        x.backward(grad)

class Exp_Base(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.ExpBase'

    def forward(self, x: Tensor, base: float) -> Tensor:
        '''
        Perform the operation of computing the exponential with an arbitrary base of a Tensor object

        Parameters
        ----------
        x : Tensor
        base : float

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after taking the exponential with an arbitrary base value

        '''
        s_data = base ** x.data
        s_grad = x.data * np.power(base, x.data - 1) * x.grad
        s = Tensor(s_data, s_grad)
        return s

class Sin(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.Sin'

    def forward(self, x: Tensor) -> Tensor:
        '''
        Compute the sine of a Tensor object

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after taking the sine
        '''
        s_data = np.sin(x.data)
        if self.mode == Mode.FORWARD:
            s_grad = np.cos(x.data) * x.grad
            s = Tensor(s_data, s_grad)
        elif self.mode == Mode.BACKWARD:
            s = Tensor(s_data)
            s.dependency = [x]
            s.layer = self
        return s

    def backward(self, x, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after taking the sine
        '''
        grad = g * np.cos(x.data)
        x.backward(grad)

class Cos(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.Cos'

    def forward(self, x: Tensor) -> Tensor:
        '''
        Compute the cosine of a Tensor object

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after taking the cosine
        '''
        s_data = np.cos(x.data)
        if self.mode == Mode.FORWARD:
            s_grad = -np.sin(x.data) * x.grad
            s = Tensor(s_data, s_grad)
        elif self.mode == Mode.BACKWARD:
            s = Tensor(s_data)
            s.dependency = [x]
            s.layer = self
        return s

    def backward(self, x, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after taking the cosine
        '''
        grad = g * -np.sin(x.data)
        x.backward(grad)

class Tan(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.Tan'

    def forward(self, x: Tensor) -> Tensor:
        '''
        Compute the tangent of a Tensor object

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after taking the tangent
        '''
        s_data = np.tan(x.data)
        if self.mode == Mode.FORWARD:
            s_grad = 1 / (np.cos(x.data) * np.cos(x.data)) * x.grad
            s = Tensor(s_data, s_grad)
        elif self.mode == Mode.BACKWARD:
            s = Tensor(s_data)
            s.dependency = [x]
            s.layer = self
        return s

    def backward(self, x, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after taking the tangent
        '''
        grad = g * (1. / np.cos(x.data)) ** 2
        x.backward(grad)

class Abs(Layer):
    def __init__(self):
        super().__init__()
        self.desc = "optimize.Layer.Abs"

    def forward(self, x: Tensor) -> Tensor:
        '''
        Compute the absolute value of a Tensor object

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after taking the absolute value
        '''
        s_data = np.abs(x.data)
        if self.mode == Mode.FORWARD:
            s_grad = x.data / np.abs(x.data) * x.grad
            s = Tensor(s_data, s_grad)
        elif self.mode == Mode.BACKWARD:
            s = Tensor(s_data)
            s.dependency = [x]
            s.layer = self
        return s

class Log(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.LogBase'

    def forward(self, x: Tensor) -> Tensor:
        '''
        Compute the natural logarithm of a Tensor object

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after taking the natural log
        
        Raises
        ------
        ValueError : raise a ValueError if any element of x has a non-positive value
        '''
        if (x.data <= 0).any():
            raise ValueError('Cannot take the log of something less than or equal to 0.')

        s_data = np.log(x.data)
        if self.mode == Mode.FORWARD:
            s_grad = 1. / x.data * x.grad
            s = Tensor(s_data, s_grad)
        elif self.mode == Mode.BACKWARD:
            s = Tensor(s_data)
            s.dependency = [x]
            s.layer = self
        return s

    def backward(self, x, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after taking the logarithm
        '''
        grad = g * (1. / x.data)
        x.backward(grad)

class Log_Base(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.LogBase'

    def forward(self, x: Tensor, base: float) -> Tensor:
        '''
        Compute the logarithm with an arbitrary base value of a Tensor object

        Parameters
        ----------
        x : Tensor
        base : float

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after taking the log with an arbitrary base
        
        Raises
        ------
        ValueError : raise a ValueError if any element of x has a non-positive value
        '''
        if (x.data <= 0).any():
            raise ValueError('Cannot take the log of something less than or equal to 0.')
        s_data = np.log(x.data) / np.log(base)
        s_grad = (1. / (x.data * np.log(base))) * x.grad
        s = Tensor(s_data, s_grad)
        return s

class ArcSin(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.ArcSin'

    def forward(self, x: Tensor):
        '''
        Compute the arcsine of a Tensor object

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after taking the arcsine
        
        Raises
        ------
        ValueError : raise a ValueError if any element of x has a value out of range [-1, 1]
        '''
        if (x.data < -1).any() or (x.data > 1).any():
            raise ValueError('Cannot perform ArcSin on something outside the range of [-1,1].')

        s_data = np.arcsin(x.data)
        if self.mode == Mode.FORWARD:
            s_grad = (1. / np.sqrt(1 - x.data ** 2)) * x.grad
            s = Tensor(s_data, s_grad)
        elif self.mode == Mode.BACKWARD:
            s = Tensor(s_data)
            s.dependency = [x]
            s.layer = self
        return s

    def backward(self, x, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after taking the arcsine
        '''
        grad = g * (1. / np.sqrt(1 - x.data**2))
        x.backward(grad)

class ArcCos(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.ArcCos'

    def forward(self, x: Tensor):
        '''
        Compute the arccosine of a Tensor object

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after taking the arccosine
        
        Raises
        ------
        ValueError : raise a ValueError if any element of x has a value out of range [-1, 1]
        '''
        if (x.data < -1).any() or (x.data > 1).any():
            raise ValueError('Cannot perform ArcCos on something outside the range of [-1,1].')
        s_data = np.arccos(x.data)
        if self.mode == Mode.FORWARD:
            s_grad = (-1. / np.sqrt(1 - x.data ** 2)) * x.grad
            s = Tensor(s_data, s_grad)
        elif self.mode == Mode.BACKWARD:
            s = Tensor(s_data)
            s.dependency = [x]
            s.layer = self
        return s

    def backward(self, x, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after taking the arccosine
        '''
        grad = g * (-1. / np.sqrt(1 - x.data**2))
        x.backward(grad)

class ArcTan(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.ArcTan'

    def forward(self, x: Tensor):
        '''
        Compute the arctangent of a Tensor object

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after taking the arctangent
        '''
        s_data = np.arctan(x.data)
        if self.mode == Mode.FORWARD:
            s_grad = (1. / (1 + x.data ** 2)) * x.grad
            s = Tensor(s_data, s_grad)
        elif self.mode == Mode.BACKWARD:
            s = Tensor(s_data)
            s.dependency = [x]
            s.layer = self
        return s

    def backward(self, x, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after taking the arctangent
        '''
        grad = g * (1 / (1 + x.data**2))
        x.backward(grad)

class Sinh(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.Sinh'

    def forward(self, x: Tensor):
        '''
        Compute the hyperbolic sine of a Tensor object

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after taking the hyperbolic sine
        '''
        s_data = np.sinh(x.data)
        if self.mode == Mode.FORWARD:
            s_grad = np.cosh(x.data) * x.grad
            s = Tensor(s_data, s_grad)
        elif self.mode == Mode.BACKWARD:
            s = Tensor(s_data)
            s.dependency = [x]
            s.layer = self
        return s

    def backward(self, x, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after taking the hyperbolic sine
        '''
        grad = g * np.cosh(x.data)
        x.backward(grad)

class Cosh(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.Cosh'

    def forward(self, x: Tensor):
        '''
        Compute the hyperbolic cosine of a Tensor object

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after taking the hyperbolic cosine
        '''
        s_data = np.cosh(x.data)
        if self.mode == Mode.FORWARD:
            s_grad = np.sinh(x.data) * x.grad
            s = Tensor(s_data, s_grad)
        elif self.mode == Mode.BACKWARD:
            s = Tensor(s_data)
            s.dependency = [x]
            s.layer = self
        return s

    def backward(self, x, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after taking the hyperbolic cosine
        '''
        grad = g * np.sinh(x.data)
        x.backward(grad)

class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.Tanh'

    def forward(self, x: Tensor):
        '''
        Compute the hyperbolic tangent of a Tensor object

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after taking the hyperbolic tangent
        '''
        s_data = np.tanh(x.data)
        if self.mode == Mode.FORWARD:
            s_grad = (1. / np.cosh(x.data) ** 2) * x.grad
            s = Tensor(s_data, s_grad)
        elif self.mode == Mode.BACKWARD:
            s = Tensor(s_data)
            s.dependency = [x]
            s.layer = self
        return s

    def backward(self, x, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after taking the hyperbolic tangent
        '''
        grad = g * (1 - (np.tanh(x.data)**2))
        x.backward(grad)
        

class Logistic(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.Logistic'

    def forward(self, x: Tensor):
        '''
        Apply the logistic function to a Tensor object

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after the logistic function
        '''
        s_data = np.exp(x.data) / (np.exp(x.data) + 1)
        s_grad = (np.exp(x.data) / (np.exp(x.data) + 1) ** 2) * x.grad
        s = Tensor(s_data, s_grad)
        return s

class SquareRoot(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.SquareRoot'

    def forward(self, x: Tensor):
        '''
        Compute the square root of a Tensor object

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        A new Tensor object with updated values and corresponding gradients after taking the square root
        
        Raises
        ------
        ValueError : raise a ValueError if any element of x has a negative value
        '''
        if (x.data < 0).any():
            raise ValueError('Cannot take the square root of something less than 0.')
        s_data = np.sqrt(x.data)
        if self.mode == Mode.FORWARD:
            s_grad = (1. / (2 * np.sqrt(x.data))) * x.grad
            s = Tensor(s_data, s_grad)
        elif self.mode == Mode.BACKWARD:
            s = Tensor(s_data)
            s.dependency = [x]
            s.layer = self
        return s

    def backward(self, x, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after taking the square root
        '''
        grad = g * (1 / (2*np.sqrt(x.data)))
        x.backward(grad)

class Sum(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.Sum'

    def forward(self, x):
        '''
        Compute the sum of all elements of a Tensor object
        
        Parameters
        ----------
        x : Tensor
        
        Returns
        -------
        A new Tensor object with updated values and corresponding dependency after taking the sum
        '''
        s_data = np.sum(x.data)
        s = Tensor(s_data)
        s.dependency = [x]
        s.layer = self
        return s

    def backward(self, x, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after taking the sum of all elements
        '''
        grad = g * np.ones_like(x.data)
        x.backward(grad)

class Repeat(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool_reverse.Layer.Repeat'

    def forward(self, x: Tensor, times: int):
        '''
        Repeat a Tensor object for arbitrary times
        
        Parameters
        ----------
        x : Tensor
        
        Returns
        -------
        A new Tensor object with updated values and corresponding dependency after repeating for arbitrary times
        '''
        assert len(x.shape) == 0
        s_data = np.repeat(x.data, times)
        s = Tensor(s_data)
        s.dependency = [x]
        s.layer = self
        return s

    def backward(self, x, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after repeating for an arbitrary times
        '''
        x_grad = g.sum()
        x.backward(x_grad)

class Comparator(Layer):
    def __init__(self, cmp):
        '''
        Initiate a generic comparator object used for the specific comparison operator cmp

        Parameters
        ----------
        cmp : np.equal, np.not_equal, np.less, np.greater, np.less_equal, or np.greater_equal
        
        '''
        super().__init__()
        self.cmp = cmp
        self.desc = 'optimize.Layer.Comparator'

    def forward(self, x: Tensor, y: Union[float, int, np.ndarray, list, Tensor]) -> Tensor:
        '''
        Compare every element in the data of two variable correspondingly using the comparison operator self.cmp

        Parameters
        ----------
        x : Tensor
        y : float, int, np.ndarray, list, or Tensor

        Returns
        -------
        A new Tensor object with the same shape as the input and contains boolean values produced by element-wise comparisons
            
        Raises
        ------
        TypeError: if the shapes of two parameters do not match, then they cannot be compare, so raise a TypeError

        '''
        if type(y) == int or type(y) == float:
            s_data = (self.cmp(x.data, y))
            s_grad = np.nan
            return Tensor(s_data, s_grad)
        elif type(y) == list:
            y = np.array(y)
        if (y.shape != x.shape):
            raise TypeError(f'param1{type(x)} and param2{type(y)} does not have the same shape')
        else:
            if type(y) == np.ndarray:
                s_data = (self.cmp(x.data, y))
            else:
                s_data = (self.cmp(x.data, y.data))
            s_grad = np.nan
        if self.mode == Mode.FORWARD:
            return Tensor(s_data, s_grad)
        elif self.mode == Mode.BACKWARD:
            return Tensor(s_data)

    def backward(self, x, y, g):
        '''
        backward function keeps track of the calculated gradient of the Tensor object after doing comparisons, which are all set to be np.nan
        '''
        x.backward(np.nan)
        y.backward(np.nan)

class Equal(Comparator):
    def __init__(self):
        super().__init__(np.equal)
        self.desc = 'optimize.Layer.Equal'

class NotEqual(Comparator):
    def __init__(self):
        super().__init__(np.not_equal)
        self.desc = 'optimize.Layer.NotEqual'

class Less(Comparator):
    def __init__(self):
        super().__init__(np.less)
        self.desc = 'optimize.Layer.Less'

class Greater(Comparator):
    def __init__(self):
        super().__init__(np.greater)
        self.desc = 'optimize.Layer.Greater'

class LessEqual(Comparator):
    def __init__(self):
        super().__init__(np.less_equal)
        self.desc = 'optimize.Layer.LessEqual'

class GreaterEqual(Comparator):
    def __init__(self):
        super().__init__(np.greater_equal)
        self.desc = 'optimize.Layer.GreaterEqual'
