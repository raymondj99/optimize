import optimize.core.functional as F
from optimize.core.tensor import Tensor
from argparse import ArgumentParser

f = lambda x: x - F.exp(-2.0 * F.sin(4.0 * x) * F.sin(4.0 * x))

def newton(f, x_k, tol=1.0e-8, max_it=100):
    """Newton Raphson method using spladtool_forward Autodifferentiation package"""
    x_k = Tensor(x_k)
    root = None
    for k in range(max_it):
        y = f(x_k)
        dx_k = - y.data / y.grad
        if (abs(dx_k) < tol):
            root = x_k + dx_k
            print(f"Found root {root.data} at iter {k+1}")
            break
        print(f"Iter {k+1}: Dx_k = {dx_k}")
        x_k += dx_k
    return root.data

def parse_args():
    parser = ArgumentParser(description="Newton-Raphson Method")
    parser.add_argument('-g', '--initial_guess', type=float, help="Initial guess", required=True)
    parser.add_argument('-t', '--tolerance', type=float, default=1.0e-8, help="Convergence tolerance")
    parser.add_argument('-i', '--maximum_iterations', type=int, default=100, help="Maximum iterations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    newton(f, args.initial_guess, args.tolerance, args.maximum_iterations)