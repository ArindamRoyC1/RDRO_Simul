import numpy as np
from scipy.optimize import minimize
from Regret_Analysis_Finite import EO_solution

#Dro Solution

# Conjugate function φ*(s) = -log(1 - s) for s < 1, applied element-wise
def phi_star_KL(s):
    # Apply a heavy penalty for s >= 1
    penalty = 1e10  # Large penalty for invalid s
    return np.where(s < 1,  -np.log(1 - s),penalty)

# Define the objective function for the DRO problem
def dro_obj(vars, xi_values, lambda_, phi_star):
    x = vars[:-2]  # Decision variable x
    alpha = vars[-2]  # Dual variable α
    beta = vars[-1]   # Dual variable β
    
    if alpha <= 0:
        return np.inf  # α must be non-negative

    # Compute h(x, ξi) = exp(-x^T ξi) for all ξi
    h_xi = np.exp(-xi_values.dot(x))  # shape: (n,)

    # Compute s = (h(x, ξi) - β) / α for all ξi
    s = (h_xi - beta) / alpha
    if max(s) >= 1:
        return np.inf  # s must be < 1 

    # Apply the conjugate function φ*(s) for all s
    E_term = np.mean(phi_star(s))

    # Full objective function
    return alpha * (E_term) + alpha * lambda_ + beta

    

# New constraint: sum of x must equal 1
def constraint_sum_to_one(vars):
    x = vars[:-2]  # Decision variable x
    return np.sum(x) - 1

# DRO solver function with positivity and sum-to-one constraints
def dro_solution(xi_values, lambda_, phi_star):
    n, d = xi_values.shape  # n = number of data points, d = dimension of each data point ξi

    # Initial guess for x, α, and β
    x_init = EO_solution(xi_values)  # Random initial guess for x, ensuring positivity
    x_init = x_init / np.sum(x_init)  # Normalize to sum to 1
    alpha_init = 1.0  # Initial guess for α

    h_xi = np.exp(-xi_values.dot(x_init))  # shape: (n,)

    # Compute s = (h(x, ξi) - β) / α for all ξi


    beta_init = max(h_xi)   # Initial guess for β

    initial_guess = np.hstack([x_init, alpha_init, beta_init])

    # Define the bounds (x_i ≥ 0 and α ≥ 0)
    bounds = [(0, None)] * d + [(0, None), (None, None)]

    # Define the constraints for α ≥ 0 and sum of x = 1
    constraints = [
        {'type': 'eq', 'fun': constraint_sum_to_one} # sum(x) = 1
    ]

    # Solve the optimization problem using 'SLSQP'
    result = minimize(dro_obj, initial_guess, args=(xi_values, lambda_, phi_star), 
                      method='SLSQP', bounds=bounds, constraints=constraints)

    x_opt = result.x[:-2]
    alpha_opt = result.x[-2]
    beta_opt = result.x[-1]
    return x_opt
