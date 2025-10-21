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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_simulation_in_parallel(scenario):
    # Set dimension
    regressor, p, aspect_ratio, n_sim, save = scenario
    covariance = np.eye(p)
    reg_name = _get_estimator_name(regressor)
    append_msg = ", ".join([f"{k}: {v}" for k, v in zip(["p", "ar", "n_sim"], [p, aspect_ratio, n_sim])])
    

    # Determine filename for simulation output
    if save is True:
        filename = SimulationResult._construct_filepath(
            reg_name, p=p, aspect_ratio=aspect_ratio, n_sim=n_sim
        )
        output_dir = Path("studio8/sim_outputs/")
        # Skip simulation if output already exists
        if (output_dir / filename).exists():
            logging.info(
                f"Skipping scenario - {append_msg}"
            )
            return

    try:
        sim_out = run_simulation(
            regressor,
            n_sim=n_sim,
            p=p,
            aspect_ratio=aspect_ratio,
            covariance=covariance,
            rsq=5.0,  # rsq for sample beta generation
            noise_distribution="normal"
        )
        logging.info(
            f"Completed scenario - {append_msg}"
        )
        if save:
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
        err_msg = f"Error in scenario - {append_msg} -- {e}"
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
        # (1, np.geomspace(0.1, 10.0, num=5000)),
        # (50, np.geomspace(0.1, 10.0, num=100)),
        (1000, np.array([0.2, 0.5, 0.8, 2.0, 5.0])),
    ]

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
                    ar,
                    n_sim,
                    False,
                )
            )
        with Pool(8) as pool:
            results = list(pool.imap_unordered(run_simulation_in_parallel, scenarios))
        # results = list(map(run_simulation_in_parallel, scenarios))
        output_dir = Path("studio8/sim_outputs/")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output := pd.DataFrame(results)).to_csv(
            output_dir / f"simulation_results_fixed_nsim_{n_sim}.csv", index=False
        )
        with open(output_dir/ f"simulation_results_fixed_nsim_{n_sim}.pkl", "wb") as f:
            pkl.dump(output, f)
