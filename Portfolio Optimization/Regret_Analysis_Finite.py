import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Generate the true distribution
def get_true_dist(N):
    """
    Generates samples from a true multivariate normal distribution.

    Parameters:
    - N: Number of samples to generate.

    Returns:
    - e_0: Generated samples from the true distribution.
    - p_0: True distribution probabilities (uniform).
    """
    # Mean vector
    mu = np.array([0.074, 0.309, 0.548])

    # Covariance matrix
    Sigma = np.array([[0.858, -0.139, -0.242],
                      [-0.139, 0.698, -0.136],
                      [-0.242, -0.136, 0.408]])

    # Generate samples
    samples = np.random.multivariate_normal(mu, Sigma, N)
    e_0 = samples
    p_0 = np.ones(len(samples)) / len(samples)

    return e_0, p_0

# Generate perturbed vector
def get_ds(S, eta):
    """
    Generates a perturbed vector d.

    Parameters:
    - S: Size of the distribution.
    - eta: The divergence radius (variance).

    Returns:
    - d: Perturbed distribution.
    """
    d = np.random.normal(size=S)
    d -= np.mean(d)
    scale = sum(d**2) / (eta**2 / S)
    d = d / np.sqrt(scale)
    return d

# Generate perturbed distributions (Q's)
def get_Qs(values, M, eta):
    """
    Generates a list of perturbed distributions (Q's).

    Parameters:
    - values: Support points for the true distribution.
    - M: Number of Q's to generate.
    - eta: Divergence radius for generating Q's.

    Returns:
    - Qs: A list of Q's (each a perturbed distribution).
    """
    S = len(values)
    P = [1 / S] * S
    Qs = [P + get_ds(S, eta) for _ in range(M)]
    return Qs

# Objective function
def objective(theta, p, e):
    """
    Computes the objective function using vectorized operations.

    Parameters:
    - theta: Decision vector.
    - p: Probability distribution.
    - e: Support of the distribution.

    Returns:
    - Objective function value.
    """
    exp_terms = np.exp(-np.dot(e, theta))
    return np.dot(p, exp_terms)

# Create bounds for the decision variables
def create_bounds(n):
    """
    Creates bounds for the decision variables (non-negative).

    Parameters:
    - n: Number of variables.

    Returns:
    - bounds: List of bounds for each variable (0, None).
    """
    return [(0, None) for _ in range(n)]

# Constraint function for optimization
def constraint_sum(theta):
    """
    Constraint function ensuring that the sum of theta equals 1.

    Parameters:
    - theta: Decision vector.

    Returns:
    - Constraint value (should be zero).
    """
    return np.sum(theta) - 1

# Optimization for the objective function
def optima(p, e):
    """
    Optimizes the objective function subject to the constraint that sum(theta) = 1.

    Parameters:
    - p: Probability distribution.
    - e: Support of the distribution.

    Returns:
    - Optimal theta.
    - Optimal objective function value.
    """
    n = e.shape[1]
    theta_initial = np.ones(n) / n

    # Constraints and bounds
    constraints = {'type': 'eq', 'fun': constraint_sum}
    bounds = create_bounds(n)

    # Perform optimization using SLSQP
    result = minimize(objective, theta_initial, args=(p, e), method='SLSQP', bounds=bounds, constraints=[constraints])

    return result.x, result.fun

# Calculate optimal loss for each Q
def get_optimals(Qs, e):
    """
    Calculates the optimal loss for each perturbed distribution Q.

    Parameters:
    - Qs: List of perturbed distributions.
    - e: Support of the distribution.

    Returns:
    - List of optimal losses for each Q.
    """
    return np.array([optima(Q, e)[1] for Q in Qs])

# Calculate regrets
def get_regrets(xs, Qs, e_0):
    """
    Calculates the regret for each decision under each perturbed distribution.

    Parameters:
    - xs: List of decisions.
    - Qs: List of perturbed distributions.
    - e_0: Support of the true distribution.

    Returns:
    - DataFrame containing regret values for each decision under each Q.
    """
    k = len(xs)
    l = len(Qs)
    A = get_optimals(Qs, e_0)
    out = [[objective(x, Qs[i], e_0) - A[i] for x in xs] for i in range(l)]
    return pd.DataFrame(out, columns=[f'x{i}' for i in range(k)])



# EO solution

def EO_solution(xi_values):
  e_new = xi_values
  p_new = np.ones(len(e_new))/len(e_new)
  EO, _ = optima(p_new, e_new)
  return EO


# Evaluate solutions
def evaluate_solutions(N, n, lambda_, e_0, data_driven):
    """
    Evaluates data-driven and EO solutions over N datasets.

    Parameters:
    - N: Number of datasets to generate.
    - n: Sample size for each dataset.
    - lambda_: Regularization parameter.
    - e_0: True distribution.
    - data_driven: Function to evaluate data-driven solutions.

    Returns:
    - DD_outputs: List of data-driven outputs.
    - EO_outputs: List of EO solution outputs.
    """
    DD_outputs = []
    EO_outputs = []

    for _ in range(N):
        indices = np.random.choice(len(e_0), n, replace=True)
        xi_values = e_0[indices]

        DD_output = data_driven(xi_values, lambda_)
        EO_output = EO_solution(xi_values)

        DD_outputs.append(DD_output)
        EO_outputs.append(EO_output)

    return DD_outputs, EO_outputs


def Single_Run(ns, G_l, K_l, G_e, K_e, N, M, e_0, data_driven):
    """
    Performs a single run of the regret analysis simulation.

    The function evaluates regret over a range of sample sizes (n) for both data-driven 
    and empirical optimization (EO) solutions. It computes the maximum expected regret 
    for each sample size by generating perturbed distributions (Qs) and comparing 
    data-driven solutions with EO solutions.

    Parameters:
    - ns: List of integers representing the sample sizes to consider. Each value in 'ns' 
          corresponds to a specific dataset size used for regret analysis.
    - G_l: Scalar controlling the relative growth of the regularization parameter (lambda_n) 
           as a function of the sample size 'n'.
    - K_l: Scalar exponent controlling the decay rate of lambda with respect to 'n'. 
           Specifically, lambda_n is proportional to G_l / (n^K_l).
    - G_e: Scalar controlling the relative growth of the divergence radius (eta_n) 
           with respect to the sample size 'n'.
    - K_e: Scalar exponent controlling the decay rate of eta with respect to 'n'. 
           Specifically, eta_n is proportional to G_e / (n^K_e).
    - N: Integer representing the number of datasets to generate for expectation estimation.
         For each 'n', 'N' datasets are generated, and for each dataset, decisions are evaluated.
    - M: Integer representing the number of perturbed distributions (Qs) to generate for each 'n'. 
         For each dataset, M perturbed distributions are used to compute the maximum regret.
    - e_0: Array representing the support of the true distribution. This is the true distribution 
           over which the regret analysis is performed, assumed to be fixed across the simulations.
    - data_driven: Function to compute data-driven solutions. It takes in xi_values (the dataset) 
                   and lambda_ (regularization parameter) and returns a decision based on the data-driven approach.

    Process:
    - For each sample size 'n', calculate the regularization parameter 'lambda_n' and divergence 
      radius 'eta_n' based on the provided growth parameters (G_l, K_l, G_e, K_e).
    - Generate M perturbed distributions (Qs) using the divergence radius eta_n.
    - Generate N datasets from the true distribution (e_0) for each sample size 'n'. 
      For each dataset, compute both the data-driven solution and the EO solution.
    - Compute the regret matrices for the data-driven solutions and EO solutions under 
      the perturbed distributions (Qs).
    - For each solution type (data-driven and EO), compute the expected regret by averaging 
      over the N datasets and then take the maximum over the M perturbed distributions (Qs).
    - Return a DataFrame where each row corresponds to a sample size 'n' and contains the 
      maximum expected regret for both the data-driven and EO solutions.

    Returns:
    - A pandas DataFrame with columns ['DD', 'EO'], representing the maximum expected regrets 
      for the data-driven (DD) and empirical optimization (EO) solutions, respectively. 
      The index of the DataFrame corresponds to the sample sizes 'ns'.

    Example:
    -------
    ns = [100, 500, 1000, 1500, 2000]
    G_l, K_l, G_e, K_e = 1, 0.5, 0.5, 0.5
    N, M = 100, 50
    e_0 = np.array([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5], [0.3, 0.4, 0.3]])  # Example true distribution
    data_driven = my_data_driven_function  # Example data-driven function

    result = Single_Run(ns, G_l, K_l, G_e, K_e, N, M, e_0, data_driven)
    print(result)
    """
    lambdas = G_l / (ns ** K_l)
    etas = G_e / (ns ** K_e)
    Ss = 1 / np.sqrt(ns) + etas
    l = len(ns)
    outs = []

    for i in range(l):
        n = ns[i]
        print(n, flush=True)
        lambda_ = lambdas[i]
        eta = etas[i]
        s = Ss[i]

        Qs = get_Qs(e_0, M, eta)

        DD_sols, EO_sols = evaluate_solutions(N, n, lambda_, e_0, data_driven)

        DD_Mat = get_regrets(DD_sols, Qs, e_0) / (s ** 2)
        EO_Mat = get_regrets(EO_sols, Qs, e_0) / (s ** 2)

        DD_Expectations = DD_Mat.mean(axis=1)
        EO_Expectations = EO_Mat.mean(axis=1)

        max_DD = DD_Expectations.max()
        max_EO = EO_Expectations.max()

        outs.append([max_DD, max_EO])

    return pd.DataFrame(outs, columns=['DD', 'EO'], index=ns)


