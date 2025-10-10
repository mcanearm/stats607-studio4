import pytest
import numpy as np
from studio7.src.simulation import generate_data
from studio7.src.estimators import run_simulation
from sklearn.linear_model import LinearRegression, QuantileRegressor, HuberRegressor
from functools import partial


@pytest.mark.parametrize("p", [5, 10, 20], ids=["p=5", "p=10", "p=20"])
def test_data_generation(p):
    X, y, beta = generate_data(p=p,
            aspect_ratio=0.2,
            covariance=np.identity(p),
            degrees_of_freedom=100,
            SNR=2.0)

    assert np.all(np.diagonal(np.linalg.inv((X.T @ X))) >= 0)

@pytest.mark.parametrize("regressor", [LinearRegression, partial(QuantileRegressor, alpha=0), HuberRegressor], ids=["OLS", "QR", "Huber"])
@pytest.mark.parametrize("p", [1, 5, 10, 20], ids=["p=1", "p=5", "p=10", "p=20"])
def test_simulation_run(regressor, p):
    sim_result = run_simulation(
        regressor,
        n_sim=5,
        p=p,
        aspect_ratio=0.2,
        covariance=np.identity(p),
        degrees_of_freedom=100,
        SNR=2.0,
    )
    assert sim_result
    assert sim_result.name in ["OLS", "QR", "Huber"]
    assert (sim_result.p == p).all()

    
def test_simulation_print():
    sim_result = run_simulation(
        LinearRegression,
        n_sim=5,
        p=2,
        aspect_ratio=0.2,
        covariance=np.identity(2),
        degrees_of_freedom=100,
        SNR=2.0,
    )
    print(sim_result)