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

from studio7.src.estimators import run_simulation, _get_estimator_name, SimulationResult
from studio7.src.simulation import generate_covariance_structure

logging.basicConfig(level=logging.INFO)


def run_simulation_in_parallel(scenario):
    # Set dimension
    p = 5
    regressor = scenario[0]
    aspect_ratio = scenario[2]
    rho = scenario[3]
    # Generate covariance structure based on rho and p
    covariance = generate_covariance_structure(rho, p)
    degrees_of_freedom = scenario[1]
    SNR = scenario[4]
    reg_name = _get_estimator_name(regressor)

    # Determine filename for simulation output
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
    else:
        try:
            sim_out = run_simulation(
                regressor,
                n_sim=1500,
                p=p,
                aspect_ratio=aspect_ratio,
                covariance=covariance,
                degrees_of_freedom=degrees_of_freedom,
                SNR=SNR,
            )
            sim_out.save(output_dir)
            logging.info(
                f"Completed scenario - df: {degrees_of_freedom}, ar: {aspect_ratio}, corr: {scenario[3]}, snr: {SNR}, regressor: {reg_name}"
            )
        except Exception as e:
            err_msg = f"Error in scenario - df: {degrees_of_freedom}, ar: {aspect_ratio}, corr: {scenario[3]}, snr: {SNR}, regressor: {reg_name}\nError: {e}"
            logging.error(err_msg)
        # Ensure all output is flushed to avoid deadlocks
        sys.stdout.flush()
        sys.stderr.flush()


# Example usage
if __name__ == "__main__":
    
    num_replications = 5000
    scenarios = [
        (1, np.geomspace(0.1, 10.0, num=5000)),
        (50, np.geomspace(0.1, 10.0, num=500)),
        (1000, np.geomspace(0.1, 10.0, num=5))
    ]
    
    out_df = pd.DataFrame()
    for n_sim, aspect_ratio in scenarios:
        for ar in aspect_ratio:
            p = ar * 200
            covariance = generate_covariance_structure(0, p)
            sim_out = run_simulation(
                CHANGE THIS,
                n_sim=n_sim,
                p=p,
                aspect_ratio=ar,
                covariance=covariance,
                degrees_of_freedom=5,
                SNR=5,
            )
            out_df = pd.concat([out_df, sim_out._df], ignore_index=True)
