"""
Example: How to import and use modules in the Studio 7 project
"""
import os
import math

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

from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor, Ridge
import numpy as np
import pandas as pd

from studio8.src.estimators import run_simulation, _get_estimator_name, SimulationResult
from studio8.src.simulation import generate_covariance_structure, generate_data

logging.basicConfig(level=logging.INFO)


def run_simulation_in_parallel(scenario):
    # Set dimension
    regressor, p, df, ar, rho, n_sim, sigma2, save = scenario
    
    # Generate covariance structure based on rho and p
    covariance = generate_covariance_structure(rho, p)
    reg_name = _get_estimator_name(regressor)

    # Determine filename for simulation output
    if save is True:
        filename = SimulationResult._construct_filepath(reg_name, p=p, snr=1, ar=ar, df=df, rho=rho, n_sim=n_sim)
        output_dir = Path(f"studio8/sim_outputs/{n_sim}")
        # Skip simulation if output already exists
        if (output_dir / filename).exists():
            logging.info(
                f"Skipping scenario - regressor: {reg_name}, df: {df}, ar: {ar}, p: {p}, corr: {rho}, n_sim: {n_sim}"
            )
            return
    
    try:
        sim_out = run_simulation(
            regressor,
            n_sim=n_sim,
            p=p,
            aspect_ratio=ar,
            covariance=covariance,
            degrees_of_freedom=df,
        )
        logging.info(
            f"Completed scenario - regressor: {reg_name}, df: {df}, ar: {ar}, p: {p}, corr: {rho}, n_sim: {n_sim}"
        )
        if save is True:
            sim_out.save(output_dir)
        else:
            return {"mse_mean": np.mean(sim_out._df["rmse"]**2),
                    "mse_std": np.std(sim_out._df["rmse"]**2),
                    "n_sim": n_sim,
                    "ar": ar}
    except Exception as e:
        err_msg = f"Error in scenario - regressor: {reg_name}, df: {df}, ar: {ar}, p: {p}, corr: {rho}, n_sim: {n_sim} -- Error: {e}"
        logging.error(err_msg)
    # Ensure all output is flushed to avoid deadlocks
    sys.stdout.flush()
    sys.stderr.flush()
    
    
# Example usage
if __name__ == "__main__":
    num_replications = 5000
    
    # Define scenario parameters
    scenario_configs = [
        (1, np.geomspace(0.1, 10.0, num=5000)),
        (50, np.geomspace(0.1, 10.0, num=500)),
        (1000, np.geomspace(0.1, 10.0, num=5))
    ]
    
    degrees_of_freedom = 5
    rho = 0
    
    # Generate all combinations of scenarios
    scenarios = []
    sigma2 = 1  # Fixed sigma2 for simulations
    
    for n_sim, aspect_ratios in scenario_configs:
        for ar in aspect_ratios:
            p = int(ar * 200)
            scenarios.append((
                partial(Ridge, alpha=1e-9),
                p, 
                degrees_of_freedom, 
                ar, 
                rho, 
                n_sim,
                sigma2,
                False
            ))
    
    # Shuffle the scenarios for more balanced parallelization
    np.random.shuffle(scenarios)
    
    # Run simulations in parallel using 8 processes

    with Pool(8) as p:
        results = p.map(run_simulation_in_parallel, scenarios)

    all_results = pd.DataFrame(results)
    all_results.to_csv("studio8/sim_outputs/simulation_summary.csv", index=False)
