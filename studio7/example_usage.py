"""
示例：如何在 Studio 7 项目中进行导入和使用
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.estimators as est
import src.plot_function as plot
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor
from src import SimulationResult, generate_data, plot_figure, run_simulation
from src.estimators import SimulationResult, generate_data, run_simulation
from src.plot_function import plot_figure

# 使用示例
if __name__ == "__main__":
    # 使用 sklearn 的回归器

    # 生成数据
    n_sims = 1000
    t_df = [1, 2, 3, 20, math.inf]
    ar = [0.2, 0.5, 0.8]
    corr = [0, 0.5, 0.9]
    snr = [1, 5, 10]
    regressors = [LinearRegression, QuantileRegressor, HuberRegressor]

    # 运行仿真
    result = run_simulation(LinearRegression, n_sim=10)
    print(f"Simulation result: {result}")

    # 如果你需要使用其他常用的库

    print("所有导入成功！")
