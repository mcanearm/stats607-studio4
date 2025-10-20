"""
Example: How to import and use modules in the Studio 7 project
"""
import os

# Limit OpenMP and BLAS/Accelerate to a single thread for reproducibility/stability
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"   # macOS Accelerate

import logging
import sys
from functools import partial
from itertools import product
import multiprocessing as mp
from multiprocessing import Pool
from pathlib import Path

from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd

from studio8.src.estimators import run_simulation, _get_estimator_name, SimulationResult
from studio8.src.simulation import generate_covariance_structure

logging.basicConfig(level=logging.INFO)


def run_simulation_in_parallel(scenario):
    # Set dimension
    regressor = scenario[0]
    p = scenario[1]
    degrees_of_freedom = scenario[2]
    aspect_ratio = scenario[3]
    rho = scenario[4]
    n_sim = scenario[5]
    sigma2 = scenario[6]
    save = scenario[7]
    # Generate covariance structure based on rho and p
    covariance = generate_covariance_structure(rho, p)
    reg_name = _get_estimator_name(regressor)

    # Determine filename for simulation output
    if save is True:
        filename = SimulationResult._construct_filepath(
            reg_name, p=p, ar=aspect_ratio
        )
        output_dir = Path("studio7/sim_outputs/")
        # Skip simulation if output already exists
        if (output_dir / filename).exists():
            logging.info(
                f"Skipping scenario - ar: {aspect_ratio}, p: {p}, n_sim: {n_sim}"
            )
            return
    
    try:
        sim_out = run_simulation(
            regressor,
            n_sim=n_sim,
            p=p,
            aspect_ratio=aspect_ratio,
            covariance=covariance,
            degrees_of_freedom=degrees_of_freedom,
            sigma2=sigma2
        )
        logging.info(
            f"Completed scenario - ar: {aspect_ratio}, p: {p}, n_sim: {n_sim}"
        )
        if save is True:
            sim_out.save()
        else:
            return {
                "N_sim": n_sim,
                "regressor": reg_name,
                "mse": np.mean(sim_out["rmse"]**2),
                "se_mse": np.std(sim_out["rmse"]**2)
            }
    except Exception as e:
            
        err_msg = f"Error in scenario - ar: {aspect_ratio}, p: {p}, n_sim: {n_sim}\nError: {e}"
        logging.error(err_msg)
    # Ensure all output is flushed to avoid deadlocks
    sys.stdout.flush()
    sys.stderr.flush()
    
    
# Example usage
if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    num_replications = 5000
    
    # Define scenario parameters
    scenario_configs = [
        (1, np.geomspace(0.1, 10.0, num=5000)),
        (50, np.geomspace(0.1, 10.0, num=500)),
        (1000, np.geomspace(0.1, 10.0, num=5))
    ]
    
    degrees_of_freedom = 5
    rho = 0
    sigma2 = 1
    
    # Generate all combinations of scenarios
    scenarios = []
    for n_sim, aspect_ratios in scenario_configs:
        for ar in aspect_ratios:
            p = int(ar * 200)
            scenarios.append((
                partial(Ridge, alpha=1e-10), 
                p,
                degrees_of_freedom, 
                ar, 
                rho, 
                n_sim,
                sigma2,
                False,
            ))    
    
    # Shuffle the scenarios for more balanced parallelization
    np.random.shuffle(scenarios)
    
    # Run simulations in parallel using 8 processes
    with Pool(processes=8) as pool:
        results = pool.map(run_simulation_in_parallel, scenarios)

    out = pd.DataFrame(results)
    out.to_csv("studio8/sim_outputs/simulation_summary.csv", index=False)