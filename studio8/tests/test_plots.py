import pytest
from studio8.src.plot_function import expected_value_beta_rmse
import numpy as np

def test_expected_value_beta_rmse():
    with pytest.warns(RuntimeWarning): 
        output = expected_value_beta_rmse(np.array([0.5, 1.0, 2.0]), sigma2=1, r2=5)
        assert output.shape == (3,)

