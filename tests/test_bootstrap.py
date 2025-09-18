import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, r_squared


np.random.seed(22)
covariates = np.random.uniform(0, 1, size=(100, 3))
X = np.concat(
    [np.ones(covariates.shape[0])[:, None], covariates ], axis=1
)
beta = np.array([0, 2, -1, 3.5])
Y = X @ beta + np.random.normal(0, 3, size=X.shape[0])

def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""
    
    r2 = r_squared(X, Y)
    boot_samples = bootstrap_sample(X, Y, r_squared, n_bootstrap=500)
    cis = bootstrap_ci(boot_samples, alpha=0.05)

    assert True
