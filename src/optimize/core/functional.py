from optimize.core.layer import *

def power(x, p):
    return Power(p)(x)

def sumup(x: "Tensor", y: Union[int, float, "Tensor", np.ndarray, list]) -> "Tensor": 
    from optimize.core.tensor import Tensor
    if isinstance(y, Tensor):
        return TensorSum()(x, y)
    else:
        return NumSum(y)(x)

def prod(x: "Tensor", y: Union[int, float, "Tensor", np.ndarray, list]) -> "Tensor":
    from optimize.core.tensor import Tensor
    if isinstance(y, Tensor):
        return TensorProd()(x, y)
    else:
        return NumProd(y)(x)

def matvecmul(x: "Tensor", y: "Tensor") -> "Tensor":
    return MatrixVecMul()(x, y)

def inv(x: "Tensor") -> "Tensor":
    return TensorInv()(x)

def div(x: Union["Tensor", int, float, "Tensor", np.ndarray, list],
        y: Union[int, float, "Tensor", np.ndarray, list]) -> "Tensor":
    from optimize.core.tensor import Tensor
    if isinstance(y, Tensor):
        if isinstance(x, Tensor):
            return prod(x, inv(y))
        else:
            return x * inv(y)
    else:
        assert type(x) == Tensor
        return prod(x, 1. / y)

def neg(x: "Tensor") -> "Tensor":
    return prod(x, -1)

def minus(x: Union["Tensor", int, float, "Tensor", np.ndarray, list],
          y: Union[int, float, Tensor, np.ndarray, list]) -> "Tensor":
    from optimize.core.tensor import Tensor
    if isinstance(y, Tensor):
        if isinstance(x, Tensor):
            return sumup(x, -y)
        else:
            return sumup(-y, x)
    else:
        assert type(x) == Tensor
        return sumup(x, -y)

def exp(x: "Tensor") -> "Tensor":
    '''
    Compute the exponential of a Tensor object
    
    Parameters
    ----------
    x : Tensor
    
    Examples
    -------
    >>> x = tensor([[1., 2.], [3., 4.]])
    >>> z = exp(x)
    >>> print(z)
    optimize.Tensor([[ 2.71828183, 7.3890561 ], [20.08553692, 54.59815003]])
    '''
    return Exp()(x)

def exp_base(x: "Tensor", base: float) -> "Tensor":
    '''
    Compute the exponential of a Tensor object with an arbitrary base value
    
    Parameters
    ----------
    x : Tensor
    
    Examples
    -------
    >>> x = tensor([[1., 2.], [3., 4.]])
    >>> base = 10
    >>> z = exp_base(x, base)
    >>> print(z)
    optimize.Tensor([[10., 100.], [1000. 10000.]])
    '''
    return Exp_Base()(x, base)

def log(x: "Tensor") -> "Tensor":
    '''
    Compute the logarithm of a Tensor object with
    
    Parameters
    ----------
    x : Tensor
    
    Examples
    -------
    >>> x = tensor([[1., 2.], [3., 4.]])
    >>> z = log(x)
    >>> print(z)
    optimize.Tensor([[0., 0.69314718], [1.09861229, 1.38629436]])    
    '''
    return Log()(x)

def log_base(x: "Tensor", base: float) -> "Tensor":
    '''
    Compute the logarithm of a Tensor object with an arbitrary base value
    
    Parameters
    ----------
    x : Tensor
    
    Examples
    -------
    >>> x = tensor([[1., 2.], [3., 4.]])
    >>> z = log_base(x, 10)
    >>> print(z)
    optimize.Tensor([[0., 0.30103], [0.47712125, 0.60205999]])
    '''
    return Log_Base()(x, base)

def sin(x: "Tensor") -> "Tensor":
    '''
    Compute the sine of a Tensor object
    
    Parameters
    ----------
    x : Tensor
    
    Examples
    -------
    >>> x = tensor([[1., 2.], [3., 4.]])
    >>> z = sin(x)
    >>> print(z)
    optimize.Tensor([[ 0.84147098, 0.90929743], [ 0.14112001, -0.7568025 ]])
    '''
    return Sin()(x)

def cos(x: "Tensor") -> "Tensor":
    '''
    Compute the cosine of a Tensor object
    
    Parameters
    ----------
    x : Tensor
    
    Examples
    -------
    >>> x = tensor([[1., 2.], [3., 4.]])
    >>> z = cos(x)
    >>> print(z)
    optimize.Tensor([[0.54030231, -0.41614684], [-0.9899925, -0.65364362]])
    '''
    return Cos()(x)

def tan(x: "Tensor") -> "Tensor":
    '''
    Compute the tangent of a Tensor object
    
    Parameters
    ----------
    x : Tensor
    
    Examples
    -------
    >>> x = tensor([[1., 2.], [3., 4.]])
    >>> z = tan(x)
    >>> print(z)
    optimize.Tensor([[1.55740772, -2.18503986], [-0.14254654, 1.15782128]])
    '''
    return Tan()(x)

def arcsin(x: Tensor):
    '''
    Compute the arcsine of a Tensor object
    
    Parameters
    ----------
    x : Tensor
    
    Examples
    -------
    >>> x = tensor([[0.1, 0.2], [0.3, 0.4]])
    >>> z = arcsin(x)
    >>> print(z)
    optimize.Tensor([[1.47062891, 1.36943841], [1.26610367, 1.15927948]])
    '''
    return ArcSin()(x)

def arccos(x: "Tensor") -> "Tensor":
    '''
    Compute the arccosine of a Tensor object
    
    Parameters
    ----------
    x : Tensor
    
    Examples
    -------
    >>> x = tensor([[0.1, 0.2], [0.3, 0.4]])
    >>> z = arccos(x)
    >>> print(z)
    optimize.Tensor([[0.09966865, 0.19739556], [0.29145679, 0.38050638]])
    '''
    return ArcCos()(x)

def arctan(x: "Tensor") -> "Tensor":
    '''
    Compute the arctangent of a Tensor object
    
    Parameters
    ----------
    x : Tensor
    
    Examples
    -------
    >>> x = tensor([[0.1, 0.2], [0.3, 0.4]])
    >>> z = arctan(x)
    >>> print(z)
    optimize.Tensor([[0.10016742, 0.20135792], [0.30469265 0.41151685]])
    '''
    return ArcTan()(x)

def sinh(x: "Tensor") -> "Tensor":
    '''
    Compute the hyperbolic sine of a Tensor object
    
    Parameters
    ----------
    x : Tensor
    
    Examples
    -------
    >>> x = tensor([[1., 2.], [3., 4.]])
    >>> z = sinh(x)
    >>> print(z)
    optimize.Tensor([[1.17520119, 3.62686041], [10.01787493, 27.2899172 ]])
    '''
    return Sinh()(x)

def cosh(x: "Tensor") -> "Tensor":
    '''
    Compute the hyperbolic cosine of a Tensor object
    
    Parameters
    ----------
    x : Tensor
    
    Examples
    -------
    >>> x = tensor([[1., 2.], [3., 4.]])
    >>> z = cosh(x)
    >>> print(z)
    optimize.Tensor([[ 1.54308063, 3.76219569], [10.067662, 27.30823284]])
    '''
    return Cosh()(x)

def tanh(x: "Tensor") -> "Tensor":
    '''
    Compute the hyperbolic tangent of a Tensor object
    
    Parameters
    ----------
    x : Tensor
    
    Examples
    -------
    >>> x = tensor([[1., 2.], [3., 4.]])
    >>> z = tanh(x)
    >>> print(z)
    optimize.Tensor([[0.76159416, 0.96402758], [0.99505475, 0.9993293 ]])
    '''
    return Tanh()(x)

def logistic(x: "Tensor") -> "Tensor":
    '''
    Apply logistic function to a Tensor object
    
    Parameters
    ----------
    x : Tensor
    
    Examples
    -------
    >>> x = tensor([[-1., 2.], [-3., 4.]])
    >>> z = logistic(x)
    >>> print(z)
    optimize.Tensor([[0.73105858, 0.88079708], [0.95257413, 0.98201379]])
    '''
    return 1 / (1 + exp(-x))

def sqrt(x: "Tensor") -> "Tensor":
    '''
    Compute the square root of a Tensor object
    
    Parameters
    ----------
    x : Tensor
    
    Examples
    -------
    >>> x = tensor([[4., 4.], [16., 16.]])
    >>> z = sqrt(x)
    >>> print(z)
    optimize.Tensor([[12., 2.], [4., 4.]])
    '''
    return SquareRoot()(x)

def abs(x: "Tensor") -> "Tensor":
    '''
    Compute the absolute value of a Tensor object
    
    Parameters
    ----------
    x : Tensor
    
    Examples
    -------
    >>> x = tensor([[-1., 2.], [-3., 4.]])
    >>> z = abs(x)
    >>> print(z)
    optimize.Tensor([[1., 2.], [3., 4.]])
    '''
    return Abs()(x)

def sum(x: Tensor) -> Tensor:
    return Sum()(x)

def mean(x: Tensor) -> Tensor:
    return sum(x) / np.prod(x.data.shape)

def log_prob(x: Tensor) -> Tensor:
    return log(logistic(x))

def repeat(x: Tensor, times: int) -> Tensor:
    return Repeat()(x, times)

def equal(x: "Tensor", y) -> bool:
    return Equal()(x, y)

def less(x: "Tensor", y) -> bool:
    return Less()(x, y)

def not_equal(x: "Tensor", y) -> bool:
    return NotEqual()(x, y)

def greater(x: "Tensor", y) -> bool:
    return Greater()(x, y)

def less_equal(x: "Tensor", y) -> bool:
    return LessEqual()(x, y)

def greater_equal(x: "Tensor", y) -> bool:
    return GreaterEqual()(x, y)

def tensor(x: "Tensor", seed=None) -> "Tensor":
    return Tensor(x, seed)