"""
示例：如何在 Studio 7 项目中进行导入和使用
"""
import os

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
    p = 5
    regressor = scenario[0]
    aspect_ratio = scenario[2]
    rho = scenario[3]
    covariance = generate_covariance_structure(rho, p)
    degrees_of_freedom = scenario[1]
    SNR = scenario[4]
    reg_name = _get_estimator_name(regressor)

    filename = SimulationResult._construct_filepath(
        reg_name, p, SNR, degrees_of_freedom, aspect_ratio, scenario[3]
    )
    output_dir = Path("studio7/sim_outputs/")
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
        sys.stdout.flush()
        sys.stderr.flush()


# 使用示例
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    # 使用 sklearn 的回归器

    # 生成数据
    n_sims = 1500
    t_df = [1, 1e1, 1e2, 1e3]
    ar = [0.2, 0.5, 0.8]
    corr = [0, 0.5, 0.9]
    snr = [1, 5, 10]
    regressors = [
        LinearRegression,
        partial(QuantileRegressor, alpha=0),
        partial(HuberRegressor, max_iter=500),
    ]

    scenarios = list(product(regressors, t_df, ar, corr, snr))
    np.random.shuffle(scenarios)

    p = Pool(8)
    p.map(run_simulation_in_parallel, scenarios)
    p.close()
    p.join()
