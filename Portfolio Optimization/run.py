import numpy as np
import pickle
import concurrent.futures
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import Regret_Analysis_Finite as ra
import L2_regularization as l2
import DRO as dro

import argparse
# Set up argument parsing
parser = argparse.ArgumentParser(description="Set parameters for the simulation.")
parser.add_argument('--name', type=str, required=True, help="Name of the simulation")
# Parse the command-line arguments
args = parser.parse_args()

sim_params = {
    "simulation_1": [0.5, 0.3, 0.5, 0.3, 'Large lambda Large eta'],
    "simulation_2": [0.5, 0.3, 0.5, 0.5, 'Large lambda equal eta'],
    "simulation_3": [0.5, 0.3, 0.5, 0.7, 'Large lambda small eta'],
    "simulation_4": [0.5, 0.5, 0.5, 0.3, 'Equal lambda Large eta'],
    "simulation_5": [0.5, 0.5, 0.5, 0.5, 'Equal lambda Equal eta'],
    "simulation_6": [0.5, 0.5, 0.5, 0.7, 'Equal lambda Small eta'],
    "simulation_7": [0.5, 0.7, 0.5, 0.3, 'Small lambda Large eta'],
    "simulation_8": [0.5, 0.7, 0.5, 0.5, 'Small lambda equal eta'],
    "simulation_9": [0.5, 0.7, 0.5, 0.7, 'Small lambda Small eta'],
}

np.random.seed(10)
e_0, _ = ra.get_true_dist(50) # must be fixed for all simulations

name = args.name

##############################
#### Set Parameters ##########
##############################

destination = 'pickles/KL_DRO/'  # destination = 'pickles/L_2/'

N = 100 # repeats to get expectation. No. of times data is generated
M = 101 # repeats to get maximum (Number of distributions Q generated))

ns = np.array(range(1,21))*50

####
## function to get data driven solution
####

# data_driven = l2.L_2_optimize

def data_driven(samples, lambda_):
  return dro.dro_solution(samples, lambda_**2, dro.phi_star_KL)

##############################
##############################

G_l, K_l, G_e, K_e, comment = sim_params[name]

# Set File Name
name = destination + name + '.pkl'


simulation_params = {
    'comment': comment,
    'G_l': G_l,
    'K_l': K_l,
    'G_e': G_e,
    'K_e': K_e,
    'N': N,
    'M': M,
    'ns': ns.tolist()  # Convert numpy array to list for better readability in the dictionary
}



def run(i):
    np.random.seed(i)
    out = ra.Single_Run(ns, G_l, K_l, G_e, K_e, N, M, e_0, data_driven)
    return out

def run_simulations():
    # Get the number of available CPU cores
    n_cores = os.cpu_count()
    print(n_cores)
    # Set the number of  CPU cores to use
    n_cores = min(100,n_cores)
    # Create a list to hold the results from each simulation
    results = [simulation_params]

    # Create a ProcessPoolExecutor to run the simulations in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Submit the simulation function with inputs 1 to n_cores
        futures = [executor.submit(run, i) for i in range(1, n_cores + 1)]

        # Collect the results as they complete
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)

    # Save the entire list of results to a single pickle file
    with open(name, 'wb') as f:
        pickle.dump(results, f)

run_simulations()
