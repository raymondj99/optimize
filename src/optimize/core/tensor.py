import numpy as np
from optimize.core.mode import Mode, AutoDiffMode

class Tensor():
    def __init__(self, x=None, grad=None, seed=None):
        '''
        Construct a Tensor object to perform forward mode automatic differentation.
        
        Parameters
        ----------
        x : np.ndarray, list, int, float or np.float_, optional, default is None
            values of the variable at which to compute the derivative

        grad : np.ndarray, list, int, float or np.float_, optional, default is None
               gradient with respect to the variable

        seed : np.ndarray, list, int, float or np.float_, optional, default is None
               seed vector is used to perform directional derivative

        Returns
        -------
        A Tensor object with the corresponding value, gradient, and seed
        
        Examples
        --------
        >>> x = Tensor([[1.], [2.], [3.]])
        >>> z = x + 4
        >>> print(x)
        optimize.Tensor([[1.], [2.], [3.]])
        
        >>> print(z)
        optimize.Tensor([[5.], [6.], [7.]])
        
        >>> print(z.grad)
        [[1.], [1.], [1.]]    
        
        '''

        super().__init__()
        if x is None:
            self.data = None
        else:
            assert type(x) in [np.ndarray, list, int, float, np.float64]
            if type(x) != np.ndarray:
                x = np.array(x).astype(float)
        
        self.data = x
        self._shape = x.shape
        self.dependency = None
        self.layer = None
        self.mode = AutoDiffMode.get_mode()

        if self.mode == Mode.BACKWARD:
            grad = np.zeros_like(self.data)
        elif grad is None:
            grad = np.ones_like(self.data)
        self.grad = grad
        if seed is not None:
            self.grad = np.dot(grad, seed)

    def backward(self, g=None):
        '''
        Method that calls the layer.backward method with arguments being corresponding dependencies

        Parameters
        ----------
        g : np.ndarray, list, int, or float, optional, default is None

        '''
        if g is None:
            g = np.ones_like(self.data)
        assert g.shape == self.data.shape
        self.grad += g
        if self.dependency is not None:
            self.layer.backward(*self.dependency, g)
    
    def __repr__(self):
        '''
        Dunder method for returning the representation of the Tensor object
        '''
        return str(self)

    def __str__(self):
        '''
        Dunder method for returning a readable string representation of the Tensor object
        '''
        return 'optimize.Tensor(\n%s\n)' % str(self.data)

    def __add__(self, y):
        '''
        Dunder method for adding another variable to the Tensor object
        
        Parameters
        ----------
        y : int, float, np.ndarray, list or Tensor
        
        Examples
        --------
        Add a scalar
        >>> x = Tensor([[1.], [2.], [3.]])
        >>> y = 1
        >>> z = x + y
        >>> print(z)
        optimize.Tensor([[2.], [3.], [4.]])
        
        Add another Tensor object
        >>> x = Tensor([[1.0], [2.0], [3.0]])
        >>> y = Tensor([[1.0], [1.0], [1.0]])
        >>> z = x + y
        >>> print(z)
        optimize.Tensor([[2.], [3.], [4.]])
        '''
        from optimize.core.functional import sumup
        return sumup(self, y)

    def __radd__(self, y):
        '''
        Dunder method for adding another variable to the Tensor object from left
        
        Parameters
        ----------
        y : int, float, np.ndarray, list or Tensor
        
        Examples
        --------
        >>> x = Tensor([[1.0], [2.0], [3.0]])
        >>> y = 4
        >>> z = y + x
        >>> print(z)
        optimize.Tensor([[5.], [6.], [7.]])
        '''
        return self.__add__(y)

    def __mul__(self, y):
        '''
        Dunder method for mutiplying the Tensor object by another variable
        
        Parameters
        ----------
        y : int, float, np.ndarray, list or Tensor
        
        Examples
        --------
        Mutiply by a scalar
        >>> x = Tensor([[1.], [2.], [3.]])
        >>> y = 3
        >>> z = x * y
        >>> print(z)
        optimize.Tensor([[3.], [6.], [9.]])
        
        Multiply by another Tensor object
        >>> x = Tensor([[1.0], [2.0], [3.0]])
        >>> y = Tensor([[1.0], [2.0], [3.0]])
        >>> z = x * y
        >>> print(z)
        optimize.Tensor([[1.], [4.], [9.]])
        '''
        from optimize.core.functional import prod
        return prod(self, y)

    def __rmul__(self, y):
        '''
        Dunder method for mutiplying the Tensor object by another variable from left
        
        Parameters
        ----------
        y : int, float, np.ndarray, list or Tensor
        
        Examples
        --------
        >>> x = Tensor([[1.0], [2.0], [3.0]])
        >>> y = 4
        >>> z = y * x
        >>> print(z)
        optimize.Tensor([[4.], [8.], [12.]])
        '''
        return self.__mul__(y)

    def __truediv__(self, y):
        '''
        Dunder method for dividing the Tensor object by another variable
        
        Parameters
        ----------
        y : int, float, np.ndarray, list or Tensor
        
        Examples
        --------
        Divide by a scalar
        >>> x = Tensor([[2.], [4.], [6.]])
        >>> y = 2
        >>> z = x / y
        >>> print(z)
        optimize.Tensor([[1.], [2.], [3.]])
        
        Divide by another Tensor object
        >>> x = Tensor([[1.0], [2.0], [3.0]])
        >>> y = Tensor([[1.0], [2.0], [3.0]])
        >>> z = x / y
        >>> print(z)
        optimize.Tensor([[1.], [1.], [1.]])
        '''
        from optimize.core.functional import div
        return div(self, y)

    def __rtruediv__(self, y):
        '''
        Dunder method for dividing the Tensor object by another variable from left
        
        Parameters
        ----------
        y : int, float, np.ndarray, list or Tensor
        
        Examples
        --------
        >>> x = Tensor([[1.0], [2.0], [4.0]])
        >>> y = 4
        >>> z = y / x
        >>> print(z)
        optimize.Tensor([[4.], [2.], [1.]])
        '''
        from optimize.core.functional import div
        return div(y, self)

    def __pow__(self, y):
        '''
        Dunder method for rasing the Tensor object to the power of y
                
        Parameters
        ----------
        y : int, float, np.ndarray, list or Tensor
        
        Examples
        --------
        >>> x = Tensor([[1.0], [2.0], [4.0]])
        >>> y = 2
        >>> z = x ** y
        >>> print(z)
        optimize.Tensor([[1.], [4.], [16.]])
        '''
        from optimize.core.functional import power
        return power(self, y)

    def __rpow__(self, *args):
        raise NotImplementedError

    def __neg__(self):
        '''
        Dunder method for negating the Tensor object
        
        Examples
        --------
        >>> x = Tensor([[1.0], [2.0], [4.0]])
        >>> z = -x
        >>> print(z)
        optimize.Tensor([[-1.], [-2.], [-4.]])
        '''
        from optimize.core.functional import neg
        return neg(self)

    def __sub__(self, y):
        '''
        Dunder method for subtracting another variable from the Tensor object
        
        Parameters
        ----------
        y : int, float, np.ndarray, list or Tensor
        
        Examples
        --------
        Subtract by a scalar
        >>> x = Tensor([[1.], [2.], [3.]])
        >>> y = 1
        >>> z = x - y
        >>> print(z)
        optimize.Tensor([[0.], [1.], [2.]])
        
        Subtract by another Tensor object
        >>> x = Tensor([[1.0], [2.0], [3.0]])
        >>> y = Tensor([[1.0], [1.0], [1.0]])
        >>> z = x - y
        >>> print(z)
        optimize.Tensor([[0.], [1.], [2.]])
        '''
        from optimize.core.functional import minus
        return minus(self, y)

    def __rsub__(self, y):
        '''
        Dunder method for subtracting another variable from the Tensor object from left
        
        Parameters
        ----------
        y : int, float, np.ndarray, list or Tensor
        
        Examples
        --------
        >>> x = Tensor([[1.0], [2.0], [3.0]])
        >>> y = 4
        >>> z = y - x
        >>> print(z)
        optimize.Tensor([[3.], [2.], [1.]])
        '''
        from optimize.core.functional import minus
        return minus(y, self)

    def __eq__(self, y):
        '''
        Dunder method for performing "equality" comparison
        
        Parameters
        ----------
        y : int, float, np.ndarray, list or Tensor
        
        Examples
        --------
        >>> x = tensor([[1., 2.], [3., 4.]])
        >>> y = [[1, 2], [3, 4]]
        >>> print(x == y)
        optimize.Tensor([[True, True], [True, True]])
        '''
        from optimize.core.functional import equal
        return equal(self, y)

    def __lt__(self, y):
        '''
        Dunder method for performing "less than" comparison
        
        Parameters
        ----------
        y : int, float, np.ndarray, list or Tensor
        
        Examples
        --------
        >>> x = tensor([[1., 2.], [3., 4.]])
        >>> y = np.array([[3, 4], [1, 2]])
        >>> print(x < y)
        optimize.Tensor([[True, True], [False, False]])
        '''
        from optimize.core.functional import less
        return less(self, y)

    def __gt__(self, y):
        '''
        Dunder method for performing "greater than" comparison
        
        Parameters
        ----------
        y : int, float, np.ndarray, list or Tensor
        
        Examples
        --------
        >>> x = tensor([[1., 2.], [3., 4.]])
        >>> y = np.array([[3, 4], [1, 2]])
        >>> print(x > y)
        optimize.Tensor([[False, False], [True, True]])
        '''
        from optimize.core.functional import greater
        return greater(self, y)

    def __ne__(self, y):
        '''
        Dunder method for performing "not equal" comparison
        
        Parameters
        ----------
        y : int, float, np.ndarray, list or Tensor
        
        Examples
        --------
        >>> x = tensor([[3., 2.], [3., 4.]])
        >>> y = np.array([[3, 4], [1, 2]])
        >>> print(x != y)
        optimize.Tensor([[False, True], [True, True]])
        '''
        from optimize.core.functional import not_equal
        return not_equal(self, y)

    def __le__(self, y):
        '''
        Dunder method for performing "less or equal than" comparison
        
        Parameters
        ----------
        y : int, float, np.ndarray, list or Tensor
        
        Examples
        --------
        >>> x = tensor([[3., 2.], [3., 4.]])
        >>> y = np.array([[3, 4], [1, 2]])
        >>> print(x <= y)
        optimize.Tensor([[True, True], [False, False])
        '''
        from optimize.core.functional import less_equal
        return less_equal(self, y)

    def __ge__(self, y):
        '''
        Dunder method for performing "greater or equal than" comparison
         
        Parameters
        ----------
        y : int, float, np.ndarray, list or Tensor
        
        Examples
        --------
        >>> x = tensor([[3., 2.], [3., 4.]])
        >>> y = np.array([[3, 4], [1, 2]])
        >>> print(x >= y)
        optimize.Tensor([[True, False], [True, True])
        '''
        from optimize.core.functional import greater_equal
        return greater_equal(self, y)

    def mean(self):
        '''
        Dunder method for calculating the mean of the Tensor object
         
        Parameters
        ----------

        Examples
        --------
        >>> x = tensor([[1., 2.], [3., 4.]])
        >>> y = mean(x)
        >>> print(y)
        spladtool_reverse.Tensor(2.5)
        '''
        from optimize.core.functional import mean
        return mean(self)

    def sum(self):
        '''
        Dunder method for calculating the mean of the Tensor object
         
        Parameters
        ----------

        Examples
        --------
        >>> x = tensor([[1., 2.], [3., 4.]])
        >>> y = sum(x)
        >>> print(y)
        spladtool_reverse.Tensor(10.0)
        '''
        from optimize.core.functional import sum
        return sum(self)

    def repeat(self, times):
        '''
        Dunder method for repeating a Tensor with single value for arbitrary times
         
        Parameters
        ----------
        times: int
            
        Examples
        --------
        >>> x = tensor(2)
        >>> print(repeat(x, 4))
        spladtool_reverse.Tensor([2. 2. 2. 2.])
        '''
        from optimize.core.functional import repeat
        return repeat(self, times)

    @property
    def shape(self):
        '''
        Return the shape of the Tensor object as a property object
        '''
        return self._shape
