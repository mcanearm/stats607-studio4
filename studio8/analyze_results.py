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

fig, ax = plt.subplots(figsize=(8, 6), ncols=1, nrows=3, sharex=True, sharey=True)
for a, ds in zip(ax, [n_sim_1, n_sim_50, n_sim_1000]):
    a.plot(ds["gamma"], ds["p"]*ds["mse"], 'o', alpha=0.05)
    a.plot(gamma1, expected_value_beta_rmse(gamma1), color="black")
    a.plot(gamma2, expected_value_beta_rmse(gamma2), color="black")
    # set y limit on each subplot
    a.set_ylim(0, 10)

plt.savefig("studio8/figures/mse_vs_aspect_ratio.png")
plt.show()
