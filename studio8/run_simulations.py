"""
Example: How to import and use modules in the Studio 7 project
"""

import os
import pickle as pkl

# Limit OpenMP and BLAS/Accelerate to a single thread for reproducibility/stability
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate

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
logger = logging.getLogger(__name__)


def run_simulation_in_parallel(scenario):
    # Set dimension
    regressor = scenario[0]
    p = scenario[1]
    degrees_of_freedom = scenario[2]
    aspect_ratio = scenario[3]
    rho = scenario[4]
    SNR = scenario[5]
    n_sim = scenario[6]
    save = scenario[7]
    # Generate covariance structure based on rho and p
    covariance = generate_covariance_structure(rho, p)
    reg_name = _get_estimator_name(regressor)

    # Determine filename for simulation output
    if save is True:
        filename = SimulationResult._construct_filepath(
            reg_name, p, SNR, degrees_of_freedom, aspect_ratio, scenario[3]
        )
        output_dir = Path("studio8/sim_outputs/")
        # Skip simulation if output already exists
        if (output_dir / filename).exists():
            logging.info(
                f"Skipping scenario - df: {degrees_of_freedom}, ar: {aspect_ratio}"
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
            return {
                "n_sim": n_sim,
                "gamma": aspect_ratio,
                "mse": np.mean(sim_out["mse"]),
                "mse_se": np.std(sim_out["mse"]),
            }
            # return sim_out._df
    except Exception as e:
        err_msg = f"Error in scenario - df: {degrees_of_freedom}, ar: {aspect_ratio}, corr: {scenario[3]}, snr: {SNR}, regressor: {reg_name}\nError: {e}"
        logging.error(err_msg)
    # Ensure all output is flushed to avoid deadlocks
    sys.stdout.flush()
    sys.stderr.flush()


# Example usage
if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    # mp.set_start_method("spawn", force=True)

    num_replications = 5000

    # Define scenario parameters
    scenario_configs = [
        (50, np.geomspace(0.1, 10.0, num=100)),
        # (1000, np.array([0.2, 0.5, 0.8, 2.0, 5.0])),
        (1, np.geomspace(0.1, 10.0, num=5000)),
    ]

    # we may not need these parameters
    degrees_of_freedom = 5
    SNR = 5
    rho = 0

    # Generate all combinations of scenarios
    for n_sim, aspect_ratios in scenario_configs:
        logger.info("Starting simulations with n_sim={n_sim}".format(n_sim=n_sim))
        scenarios=[]
        for ar in aspect_ratios:
            p = np.ceil(int(ar * 200))
            scenarios.append(
                (
                    partial(Ridge, alpha=1e-10),
                    p,
                    degrees_of_freedom,
                    ar,
                    rho,
                    SNR,
                    n_sim,
                    False,
                )
            )
        # with Pool(1) as pool:
        #     results = pool.map(run_simulation_in_parallel, scenarios)
        results = list(map(run_simulation_in_parallel, scenarios[-1:]))
        (output := pd.DataFrame(results)).to_csv(
            f"profiling_results_nsim_{n_sim}.csv", index=False
        )
        output.to_csv(f"profiling_results_nsim_{n_sim}.csv", index=False)
        with open(f"profiling_results_nsim_{n_sim}.pkl", "wb") as f:
            pkl.dump(output, f)
