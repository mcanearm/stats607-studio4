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
    # Set multiprocessing start method for compatibility
    mp.set_start_method("spawn", force=True)

    # Use sklearn regressors
    n_sims = 1500
    t_df = [1, 1e1, 1e2, 1e3]    # list of degrees of freedom for t-distribution
    ar = [0.2, 0.5, 0.8]         # aspect ratios
    corr = [0, 0.5, 0.9]         # correlation values for covariance
    snr = [1, 5, 10]             # signal-to-noise ratios
    regressors = [
        LinearRegression,
        partial(QuantileRegressor, alpha=0),
        partial(HuberRegressor, max_iter=500),
    ]

    # Generate all combinations of scenarios
    scenarios = list(product(regressors, t_df, ar, corr, snr))
    # Shuffle the scenarios for more balanced parallelization
    np.random.shuffle(scenarios)

    # Run simulations in parallel using 8 processes
    p = Pool(8)
    p.map(run_simulation_in_parallel, scenarios)
    p.close()
    p.join()
