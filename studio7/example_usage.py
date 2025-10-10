"""
示例：如何在 Studio 7 项目中进行导入和使用
"""
import math
from itertools import product
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor
from studio7.src.simulation import generate_data, generate_covariance_structure
from studio7.src.estimators import run_simulation
from studio7.src.plot_function import plot_figure

# 使用示例
if __name__ == "__main__":
    # 使用 sklearn 的回归器

    # 生成数据
    n_sims = 1000
    t_df = [1, 2, 3, 20, math.inf]
    ar = [0.2, 0.5, 0.8]
    corr = [0, 0.5, 0.9]
    snr = [1, 5, 10]
    regressors = [LinearRegression, QuantileRegressor, partial(HuberRegressor, max_iter=500)]

    scenarios = product(t_df, ar, corr, snr, regressors)

    for scenario in scenarios:
        p = 5  # for now, hard code p
        _df, _ar, _corr, _snr, _regressor = scenario
        cov_mat = generate_covariance_structure(_corr, p)
        sim_result = run_simulation(
            _regressor,
            n_sim=n_sims,
            p=p,
            aspect_ratio=_ar,
            covariance=cov_mat,
            degrees_of_freedom=_df,
            SNR=_snr,
        )
        print(sim_result)
        sim_result.save("./sim_outputs/")

