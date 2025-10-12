"""
示例：如何在 Studio 7 项目中进行导入和使用
"""

import math
from functools import partial
from itertools import product
from pathlib import Path
import logging
import os
import pandas as pd

from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor

from studio7.src.estimators import run_simulation
from studio7.src.simulation import generate_covariance_structure, generate_data
from studio7.src.estimators import SimulationResult, _get_estimator_name

logging.basicConfig(level=logging.INFO)

# 使用示例
if __name__ == "__main__":
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

    scenarios = list(product(t_df, ar, corr, snr, regressors))
    output_dir = Path("studio7/sim_outputs/")
    results_list = []
    
    # run simulations
    for (i, scenario) in enumerate(scenarios):
        try:
            p = 5  # for now, hard code p
            _df, _ar, _corr, _snr, _regressor = scenario
            name = _get_estimator_name(_regressor)
            out_fp = SimulationResult._construct_filepath(name, p, _snr, _df, _ar, _corr)
            if (output_dir / out_fp).exists():
                logging.info(f"Skipping scenario {i+1} - already exists: {out_fp}")
                continue

            cov_mat = generate_covariance_structure(_corr, p)
            sim_result  = run_simulation(
                _regressor,
                n_sim=n_sims,
                p=p,
                aspect_ratio=_ar,
                covariance=cov_mat,
                degrees_of_freedom=_df,
                SNR=_snr,
            )
            logging.info(f"Completed {i+1} of {len(scenarios)} scenarios")
            print(sim_result)
            sim_result.save(output_dir)
            results_list.append(sim_result)
        except Exception as e:
            err_msg = f"Error in scenario {i+1} - df: {_df}, ar: {_ar}, corr: {_corr}, snr: {_snr}, regressor: {_regressor}\nError: {e}"
            logging.error(err_msg)
            continue