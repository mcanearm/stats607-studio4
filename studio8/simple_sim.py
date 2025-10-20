from studio8.src.simulation import generate_data
from studio8.src.estimators import run_simulation
from sklearn.linear_model import Ridge
from functools import partial
import numpy as np

gamma = np.arange(0.1, 10, 0.1)
n = 200
sigma = 1


X, y, beta = generate_data(p=500, N=200, covariance=np.eye(500), degrees_of_freedom=5, SNR=2, rng=None)


estimator_class = partial(Ridge, alpha=1e-10)

gamma = np.arange(0.1, 10, 0.1)
run_simulation(Ridge, n_sim=1, )