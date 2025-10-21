import pandas as pd
import numpy as np
from studio8.src.plot_function import expected_value_beta_rmse
from matplotlib import pyplot as plt
import matplotlib

mse_values = pd.concat([
    pd.read_csv("./simulation_results_nsim_1.csv"),
    pd.read_csv("./simulation_results_nsim_50.csv"),
    pd.read_csv("./simulation_results_nsim_1000.csv"),
])

gamma1 = np.arange(0.1, 0.99, 0.01)
gamma2 = np.arange(1.01, 10.01, 0.01)

n_sim_1 = mse_values[(mse_values["n_sim"] == 1) & (mse_values["mse"] < 1)][["mse", "gamma"]]

plt.plot(gamma1, expected_value_beta_rmse(gamma1), color="black")
plt.plot(gamma2, expected_value_beta_rmse(gamma2), color="black")
plt.plot(n_sim_1["gamma"], n_sim_1["mse"], 'o')
plt.show()

expected_value_beta_rmse(gamma)
mse_values
