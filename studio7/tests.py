import pytest
import numpy as np
from studio7.src.simulation import generate_data
from studio7.src.estimators import run_simulation
from sklearn.linear_model import LinearRegression, QuantileRegressor, HuberRegressor


@pytest.mark.parametrize("p", [5, 10, 20], ids=["p=5", "p=10", "p=20"])
def test_data_generation(p):
    X, y, beta = generate_data(p=p,
            aspect_ratio=0.2,
            covariance=np.identity(p),
            degrees_of_freedom=100,
            SNR=2.0)

    assert np.all(np.diagonal(np.linalg.inv((X.T @ X))) >= 0)

@pytest.mark.parametrize("regressor", [LinearRegression, QuantileRegressor, HuberRegressor], ids=["OLS", "QR", "Huber"])
@pytest.mark.parametrize("p", [5, 10, 20], ids=["p=5", "p=10", "p=20"])
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
    # assert sim_result.estimator_name == regressor.__name__
    # assert sim_result.n_simulations == 200
    # assert sim_result.p == p
    # assert sim_result.aspect_ratio == 0.2
    # assert sim_result.covariance.shape == (p, p)
    # assert sim_result.degrees_of_freedom == 100
    # assert sim_result.SNR == 2.0
    # assert len(sim_result.outputs) == 200
    # for output in sim_result.outputs:
    #     assert "predictions" in output
    #     assert "beta_hat" in output
    #     assert "rmse" in output
    #     assert "r2" in output
    #     assert "true_beta" in output
    #     assert "se_beta" in output
    #     assert "N" in output