"""
示例：如何在 Studio 7 项目中进行导入和使用
"""

# 方法 1: 从 src 包导入所有内容
from src import generate_data, SimulationResult, run_simulation, plot_figure

# 方法 2: 分别从各个模块导入
from src.estimators import generate_data, SimulationResult, run_simulation
from src.plot_function import plot_figure

# 方法 3: 导入整个模块
import src.estimators as est
import src.plot_function as plot

# 方法 4: 如果你在 studio7 目录下工作
# from src import *  # 导入所有公开的函数和类

# 使用示例
if __name__ == "__main__":
    # 使用 sklearn 的回归器
    from sklearn.linear_model import LinearRegression, QuantileRegressor, HuberRegressor
    
    # 生成数据
    X, y = generate_data()
    print(f"Generated data: X shape {X.shape}, y shape {y.shape}")
    
    # 运行仿真
    result = run_simulation(LinearRegression, n_sim=10)
    print(f"Simulation result: {result}")
    
    # 如果你需要使用其他常用的库
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    print("所有导入成功！")
