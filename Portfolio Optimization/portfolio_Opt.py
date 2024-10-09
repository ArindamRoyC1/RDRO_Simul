import numpy as np
import pandas as pd
from scipy.optimize import minimize
import random

# Fix the true distribution (Finite, Discrete)

def get_true_dist(N):
  # Fix seed before running this
  # Mean vector
  mu = np.array([0.74, 0.309, 0.548])

  # Covariance matrix
  Sigma = np.array([[0.858, -0.139, -0.242],
                    [-0.139, 0.698, -0.136],
                    [-0.242, -0.136, 0.408]])


  samples = np.random.multivariate_normal(mu, Sigma, 50)
  e_0 = samples  # true distribution
  p_0 = np.ones(len(samples))/len(samples) # true distribution

  return (e_0,p_0)

# Generate Distribution within phi divergence

def get_ds(S,eta):
  d = np.random.normal(size = S)
  d = d-np.mean(d)
  scale = sum(d**2)/(eta**2/S)
  d = d/np.sqrt(scale)
  return d

def get_Qs(values,M,eta):
  '''
  values: True distribution is uniform over these points
  M: Number of Q's to generate
  eta: phi divergence radius 

  Returns a list of Q's. Each Q is a list of probabilities, on the same set of points as values
  '''
  S = len(values)
  P = [1/S]*S
  Qs = [P + get_ds(S,eta) for i in range(M)]
  return Qs


## Get optimal solution

#given $Q$, obtain $x^*(Q)$

#  - $Q$ is characterized by $(e,p)$
#  - $e$ is the set of points in the support of $Q$ (same as base $e_0$)
#  - $p$ is the corresponding probabilities



# Define the objective function
def objective(theta, p, e):
    # Compute the objective function efficiently using vectorized operations
    exp_terms = np.exp(-np.dot(e, theta))  # e_i^T * theta for all i
    return np.dot(p, exp_terms)  # Weighted sum by p

# Constraint: sum(theta) = 1
def constraint_sum(theta):
    return np.sum(theta) - 1

# Bounds for theta (theta_i >= 0)
def create_bounds(n):
    return [(0, None) for _ in range(n)]

# Fast implementation using vectorized operations
def optima(p, e):
    n = e.shape[1]  # Dimension of theta
    theta_initial = np.ones(n) / n  # Initial guess: uniform distribution

    # Define the constraints and bounds
    constraints = {'type': 'eq', 'fun': constraint_sum}
    bounds = create_bounds(n)

    # Perform the optimization using SLSQP
    result = minimize(objective, theta_initial, args=(p, e), method='SLSQP', bounds=bounds, constraints=[constraints])

    return result.x, result.fun


#Get optimals


def get_optimals(Qs, e):
  '''
  Qs: list of Q's
  e: support of Q (assumed same for all the Q's)
  returns: list of optimal loss
  '''
  return np.array([optima(Q, e)[1] for Q in Qs])


### Evaluate Regret

#  - We have a vector of decisions xs (length k)
#  - We have a list of Qs (length l)
#  - $e$ is the finite support of the distributions
#  - Return matrix ($k\times l$) with regret of each $(x,Q)$ pair

def get_regrets(xs, Qs, e_0):
  k = len(xs)
  l = len(Qs)
  A = get_optimals(Qs, e_0)
  out =  [[(objective(x,Qs[i],e_0) - A[i]) for x in xs] for i in range(l)]
  return pd.DataFrame(out, columns = [f'x{i}' for i in range(k)])

#KL Dro Solution

# Conjugate function φ*(s) = -log(1 - s) for s < 1, applied element-wise
def phi_star(s):
    # Apply a heavy penalty for s >= 1
    penalty = 1e10  # Large penalty for invalid s
    return np.where(s < 1,  -np.log(1 - s),penalty)

# Define the objective function for the DRO problem
def objective_dro(vars, xi_values, lambda_, n):
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
    sum_term = np.sum(phi_star(s))

    # Full objective function
    return alpha * (sum_term / n) + alpha * lambda_ + beta

    

# New constraint: sum of x must equal 1
def constraint_sum_to_one(vars):
    x = vars[:-2]  # Decision variable x
    return np.sum(x) - 1

# DRO solver function with positivity and sum-to-one constraints
def dro_solution(xi_values, lambda_):
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
    result = minimize(objective_dro, initial_guess, args=(xi_values, lambda_, n), 
                      method='SLSQP', bounds=bounds, constraints=constraints)

    x_opt = result.x[:-2]
    alpha_opt = result.x[-2]
    beta_opt = result.x[-1]
    return x_opt

# EO solution

def EO_solution(xi_values):
  e_new = xi_values
  p_new = np.ones(len(e_new))/len(e_new)
  EO, _ = optima(p_new, e_new)
  return EO


# Consider a bunch of datasets



def evaluate_solutions(N, n, lambda_, e_0):
    # Initialize empty lists to store the outputs for dro_solution and EO_solution
    dro_outputs = []
    EO_outputs = []

    for _ in range(N):
        # Sample xi_values from e_0 with replacement
        indices = np.random.choice(len(e_0), n, replace=True)
        xi_values = e_0[indices]
        
        # Evaluate dro_solution and EO_solution
        dro_output = dro_solution(xi_values, lambda_)
        EO_output = EO_solution(xi_values)
        
        # Append the outputs to their respective lists
        dro_outputs.append(dro_output)
        EO_outputs.append(EO_output)

    # Convert the lists of outputs into dataframes
    #dro_df = pd.DataFrame(dro_outputs, columns=[f'dro_{i}' for i in range(len(dro_outputs[0]))])
    #EO_df = pd.DataFrame(EO_outputs, columns=[f'EO_{i}' for i in range(len(EO_outputs[0]))])
    
    return dro_outputs, EO_outputs


'''
# Single Run

### Parameters:
  - $\gamma_{\lambda}$, $\kappa_{\lambda}$:   Control relative growths of $(n, \lambda_n)$.  
  - $\gamma_{\eta}$, $\kappa_{\eta}$:   Control relative growths of $(n, \eta_n)$.  
  - N:  Number of data sets to estimate Expectation
  - M:  Number of Qs to take maximum over

### Scheme:
We have $(n, \eta_n, \lambda_n)$.   
Consider $n\in [100, 500, 1000, 1500, 2000]$.    
output: a single vector, representing max expected regret at each n.  


For each n:

  - Create M Qs within $\eta_n$ divergence of $e_0$
  - Create N datasets. For each data, evaluate $x^{EO}$, $x^{DRO}_{\lambda_n}$.
  - Create the two $M\times N$ matrix of regrets
  - Take Expectation over N, then Max over M

'''

def Single_Run(ns, G_l, K_l, G_e, K_e, N, M, e_0):
  lambdas = (G_l/(ns**K_l))**2
  etas = G_e/(ns**K_e)
  Ss = 1/np.sqrt(ns)+etas
  l = len(ns)
  outs = []
  for i in range(l):
    n = ns[i]
    print(n, flush = True)
    lambda_ = lambdas[i]
    eta = etas[i]
    s = Ss[i]

   # print('n ,lambda_, eta, s = ',np.round([n,lambda_, eta, s],6))

    Qs = get_Qs(e_0,M,eta)

    DRO_sols, EO_sols = evaluate_solutions(N, n, lambda_, e_0)

    DRO_Mat = get_regrets(DRO_sols, Qs, e_0)/(s**2)
    EO_Mat = get_regrets(EO_sols, Qs, e_0)/(s**2)

    DRO_Expectations = DRO_Mat.mean(axis = 1)
    EO_Expectations = EO_Mat.mean(axis = 1)

    # print(DRO_Expectations.shape) # should be M
    # print(EO_Expectations.shape)

    max_DRO = DRO_Expectations.max()
    max_EO = EO_Expectations.max()

    outs.append([max_DRO, max_EO])
  return pd.DataFrame(outs, columns = ['DRO', 'EO'], index = ns)


