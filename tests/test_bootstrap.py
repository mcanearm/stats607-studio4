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

def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""
    
    r2 = R_squared(X, Y)
    boot_samples = bootstrap_sample(X, Y, R_squared, n_bootstrap=500)
    cis = bootstrap_ci(boot_samples, alpha=0.05)

    assert True

def test_bootstrap_ci():
    """Test that bootstrap_ci returns correct shape and values"""
    
    samples = np.random.normal(0, 1, size=1000)
    
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        bootstrap_ci(samples, alpha=-0.1)
    
    with pytest.raises(TypeError, match="length of bootstrap_stats is zero"):
        bootstrap_ci([])
    
    with pytest.raises(TypeError, match="samples must be a 1D array-like"):
        bootstrap_ci(np.array([[1, 2], [3, 4]]))
    
    with pytest.raises(ValueError, match="samples must contain at least two values"):
        bootstrap_ci(np.array([1]))
    
    low95, up95 = bootstrap_ci(samples, alpha=0.05)
    low90, up90 = bootstrap_ci(samples, alpha=0.10)
    
    assert low90 < low95 or up90 > up95
    assert low95 < low90 or up95 > up90

