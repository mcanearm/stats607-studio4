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

from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor
import numpy as np
import pandas as pd

from studio8.src.estimators import run_simulation, _get_estimator_name, SimulationResult
from studio8.src.simulation import generate_covariance_structure

logging.basicConfig(level=logging.INFO)


def run_simulation_in_parallel(scenario):
    # Set dimension
    p = 5
    regressor = scenario[0]
    degrees_of_freedom = scenario[1]
    aspect_ratio = scenario[2]
    rho = scenario[3]
    SNR = scenario[4]
    n_sim = scenario[5]
    save = scenario[6]
    # Generate covariance structure based on rho and p
    covariance = generate_covariance_structure(rho, p)
    reg_name = _get_estimator_name(regressor)

    # Determine filename for simulation output
    if save is True:
        filename = SimulationResult._construct_filepath(
            reg_name, p, SNR, degrees_of_freedom, aspect_ratio, scenario[3]
        )
        output_dir = Path("studio7/sim_outputs/")
        # Skip simulation if output already exists
        if (output_dir / filename).exists():
            logging.info(
                f"Skipping scenario - df: {degrees_of_freedom}, ar: {aspect_ratio}, corr: {scenario[3]}, snr: {SNR}, regressor: {reg_name}"
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
            SNR=SNR,
        )
        logging.info(
            f"Completed scenario - df: {degrees_of_freedom}, ar: {aspect_ratio}, corr: {scenario[3]}, snr: {SNR}, regressor: {reg_name}"
        )
        if save is True:
            sim_out.save()
        else:
            return sim_out._df  
    except Exception as e:
        err_msg = f"Error in scenario - df: {degrees_of_freedom}, ar: {aspect_ratio}, corr: {scenario[3]}, snr: {SNR}, regressor: {reg_name}\nError: {e}"
        logging.error(err_msg)
    # Ensure all output is flushed to avoid deadlocks
    sys.stdout.flush()
    sys.stderr.flush()
    
    
# Example usage
if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method("spawn", force=True)
    
    num_replications = 5000
    
    # Define scenario parameters
    scenario_configs = [
        (1, np.geomspace(0.1, 10.0, num=5000)),
        (50, np.geomspace(0.1, 10.0, num=500)),
        (1000, np.geomspace(0.1, 10.0, num=5))
    ]
    
    degrees_of_freedom = 5
    SNR = 5
    rho = 0
    
    # Generate all combinations of scenarios
    scenarios = []
    for n_sim, aspect_ratios in scenario_configs:
        for ar in aspect_ratios:
            p = int(ar * 200)
            scenarios.append((
                CHANGE THIS, 
                degrees_of_freedom, 
                ar, 
                rho, 
                SNR,
                n_sim,
                False,
            ))    
    
    # Shuffle the scenarios for more balanced parallelization
    np.random.shuffle(scenarios)
    
    # Run simulations in parallel using 8 processes
    pool = Pool(8)
    results = pool.map(run_simulation_in_parallel, scenarios)
    pool.close()
    pool.join()