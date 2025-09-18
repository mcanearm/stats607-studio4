import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, R_squared


np.random.seed(22)
covariates = np.random.uniform(0, 1, size=(100, 3))
X = np.concat(
    [np.ones(covariates.shape[0])[:, None], covariates ], axis=1
)
beta = np.array([0, 2, -1, 3.5])
Y = X @ beta + np.random.normal(0, 3, size=X.shape[0])


def test_bootstrap_sample():
    """
    Test that bootstrap_sample outputs a valid length of numpy array
    """
    n_boot = 123
    samples_boot = bootstrap_sample(X, Y, compute_stat=R_squared, n_bootstrap=n_boot)
    assert isinstance(samples_boot, np.ndarray)
    assert len(samples_boot) == n_boot

def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""
    
    r2 = R_squared(X, Y)
    boot_samples = bootstrap_sample(X, Y, R_squared, n_bootstrap=500)
    cis = bootstrap_ci(boot_samples, alpha=0.05)

    assert True


@pytest.mark.parametrize("X, Y", [("test", Y), (X, "test"), ("test", "test")])
def test_R_squared_type(X, Y):
    with pytest.raises(TypeError, match="X and Y must be an ndarray"):
        R_squared(X, Y)

        
def test_R_squared_shape():
    assert len(R_squared(X, Y)) == 1
   
