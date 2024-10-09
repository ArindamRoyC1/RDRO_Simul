import numpy as np
from scipy.optimize import minimize
from Regret_Analysis_Finite import constraint_sum

# L2 Objective function
def L2obj(x, samples, lambda_):
    """
    L2 Regularized objective function.

    Parameters:
    - x: Decision vector.
    - samples: Sample data (e_0).
    - lambda_: Regularization parameter.

    Returns:
    - Value of the L2 objective function.
    """
    exp_ = np.exp(-samples.dot(x))
    return exp_.mean() + lambda_ * x.dot(x)

# L2 Optimization function
def L_2_optimize(samples, lambda_):
    """
    Optimizes the L2 objective function subject to sum of theta being 1.

    Parameters:
    - samples: Sample data (e_0).
    - lambda_: Regularization parameter.

    Returns:
    - Optimal decision vector.
    """
    n = samples.shape[1]
    x_init = np.ones(n) / n
    const = {'type': 'eq', 'fun': constraint_sum}
    bounds = [(0, None) for _ in range(n)]
    result = minimize(L2obj, x_init, args=(samples, lambda_), method='SLSQP', bounds=bounds, constraints=[const])
    return result.x
