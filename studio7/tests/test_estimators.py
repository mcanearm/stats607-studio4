import tempfile
import numpy as np
from pathlib import Path
import pytest
from studio7.src.estimators import SimulationResult, run_simulation
from sklearn.linear_model import LinearRegression, HuberRegressor
from functools import partial


@pytest.fixture
def sample_data():
    p = 5
    sim_result = run_simulation(
        LinearRegression,
        n_sim=5,
        p=p,
        aspect_ratio=0.2,
        covariance=np.identity(p),
        degrees_of_freedom=100,
        SNR=2.0,
    )
    return sim_result


def test_partial_regressor():
    p = 5
    sim_result = run_simulation(
        partial(HuberRegressor, max_iter=500),
        n_sim=5,
        p=p,
        aspect_ratio=0.2,
        covariance=np.identity(p),
        degrees_of_freedom=100,
        SNR=2.0,
    )
    assert sim_result


def test_sim_read_write(sample_data):
    # Prepare a DataFrame and save to a temporary CSV file
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_data.save(Path(tmpdir))
        result = SimulationResult.load(tmpdir / sample_data.filename)

        assert isinstance(result, SimulationResult)
        assert result.name == sample_data.name
        assert np.allclose(result.beta_hat[0], sample_data.beta_hat[0])


@pytest.mark.parametrize("attribute", SimulationResult.stacked_attribtes)
def test_dict_interface(attribute, sample_data):
    assert np.all(sample_data[attribute] == sample_data.__getattr__(attribute))
