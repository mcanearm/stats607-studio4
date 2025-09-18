import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, R_squared


"""
Generate testing data. For basic tests, we can do this once, but
this could also be split into a fixture so that each test can get a 
copy without accidentally sharing state. This would be very important
for pandas dfs, dicts, or other mutable objects.
"""
np.random.seed(22)
covariates = np.random.uniform(0, 1, size=(100, 3))
X = np.concat([np.ones(covariates.shape[0])[:, None], covariates], axis=1)
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
    """
    Test that bootstrap_sample and bootstrap_ci work together. This is the
    "happy path test."
    """

    r2 = R_squared(X, Y)
    boot_samples = bootstrap_sample(X, Y, R_squared, n_bootstrap=500)
    cis = bootstrap_ci(boot_samples, alpha=0.05)

    assert True


@pytest.fixture
def samples():
    """
    Simple fixture to generate random data for use in other tests.
    """
    np.random.seed(571)
    return np.random.normal(0, 1, size=1000)



"""
atomize the next few tests to see what specifically fails.
"""
def test_boot_alpha(samples):
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        bootstrap_ci(samples, alpha=-0.1)


def test_boot_length():
    with pytest.raises(TypeError, match="length of bootstrap_stats is zero"):
        bootstrap_ci([])


def test_1d_array():
    with pytest.raises(TypeError, match="samples must be a 1D array-like"):
        bootstrap_ci(np.array([[1, 2], [3, 4]]))


def test_at_least_two_values():
    with pytest.raises(ValueError, match="samples must contain at least two values"):
        bootstrap_ci(np.array([1]))


def test_bootstrap_ci_order(samples):
    low95, up95 = bootstrap_ci(samples, alpha=0.05)
    low90, up90 = bootstrap_ci(samples, alpha=0.10)

    assert not low90 < low95 or up90 > up95


def test_R_squared_type():
    with pytest.raises(
        ValueError, match="X must be a 2D array and y must be a 1D array."
    ):
        R_squared(Y, Y)


def test_R_squared_shape():
    assert isinstance(R_squared(X, Y), float)
