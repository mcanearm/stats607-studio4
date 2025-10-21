import pandas as pd
import numpy as np
from studio8.src.plot_function import expected_value_beta_rmse
from matplotlib import pyplot as plt

mse_values = pd.concat([
    pd.read_csv("./simulation_results_nsim_1.csv"),
    pd.read_csv("./simulation_results_nsim_50.csv"),
    pd.read_csv("./simulation_results_nsim_1000.csv"),
])

mse_values["p"] = mse_values["gamma"] * 200

gamma1 = np.arange(0.1, 0.99, 0.01)
gamma2 = np.arange(1.1, 10.01, 0.01)

n_sim_1 = mse_values[(mse_values["n_sim"] == 1)][["mse", "gamma", "p"]]
n_sim_50 = mse_values[(mse_values["n_sim"] == 50)][["mse", "gamma", "p"]]
n_sim_1000 = mse_values[(mse_values["n_sim"] == 1000)][["mse", "gamma", "p"]]

plt.plot(n_sim_1["gamma"], n_sim_1["p"]*n_sim_1["mse"], 'o', alpha=0.05)
plt.plot(n_sim_50["gamma"], n_sim_50["p"]*n_sim_50["mse"], 'o', alpha=0.5)
plt.plot(n_sim_1000["gamma"], n_sim_1000["p"]*n_sim_1000["mse"], 'o', alpha=1.0)
plt.plot(gamma1, expected_value_beta_rmse(gamma1), color="black")
plt.plot(gamma2, expected_value_beta_rmse(gamma2), color="black")
plt.ylim(0, 10)

plt.savefig("studio8/figures/mse_vs_aspect_ratio.png")
plt.show()
