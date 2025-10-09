"""
Strong linear model in regression
    Y = X beta + eps, where eps~ N(0, sigma^2 I)
    Under the null where beta_1 = ... = beta_p = 0,
    the R-squared coefficient has a known distribution
    (if you have an intercept beta_0), 
        R^2 ~ Beta(p/2, (n-p-1)/2)
"""

import numpy as np


def bootstrap_sample(X, y, compute_stat, n_bootstrap=1000):
    """
    Generate bootstrap distribution of a statistic

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)
    compute_stat : callable
        Function that computes a statistic (float) from data (X, y)
    n_bootstrap : int, default 1000
        Number of bootstrap samples to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap

    ....
    """
    bootstrap_stats = np.empty(n_bootstrap)
    
    for b in range(n_bootstrap):
        idx = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        Xtilde = X[idx]
        ytilde = y[idx]
        bootstrap_stats[b] = compute_stat(Xtilde, ytilde)

    return bootstrap_stats

def bootstrap_ci(bootstrap_stats, alpha=0.05):
    """
    Calculate confidence interval from the bootstrap samples

    Parameters
    ----------
    bootstrap_stats : array-like
        Array of bootstrap statistics
    alpha : float, default 0.05
        Significance level (e.g. 0.05 gives 95% CI)

    Returns
    -------
    tuple (float, float)
        1d array of length giving (lower_bound, upper_bound) of the CI

    ....
    """
    try:
        if not (0 < alpha and alpha < 1):
            raise ValueError("alpha must be between 0 and 1")
        elif len(bootstrap_stats) == 0:
            raise TypeError("length of bootstrap_stats is zero")
        elif len(bootstrap_stats.shape) != 1:
            raise TypeError("samples must be a 1D array-like")
        elif len(bootstrap_stats) == 1:
            raise ValueError("samples must contain at least two values")
    except Exception as e:
        raise e

    alpha_bounds = np.array([alpha / 2, 1 - alpha / 2])
    return tuple(np.quantile(bootstrap_stats, np.array(alpha_bounds)))


def R_squared(X, y):
    """
    Calculate R-squared from multiple linear regression.

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)

    Returns
    -------
    float
        R-squared value (between 0 and 1) from OLS
    
    Raises
    ------
    ValueError
        If X.shape[0] != len(y)
        If X.ndim != 2
    """
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    dim_msg = "X must be a 2D array and y must be a 1D array."
    if X.ndim != 2:
        raise ValueError(dim_msg)
    if y.ndim != 1:
        raise ValueError(dim_msg)
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of rows of X must match length of y.")

    # Compute OLS via least squares
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta

    # Compute Sum of Squares Total and Sum of Squares Error
    y_mean = np.mean(y)
    sst = np.sum((y - y_mean) ** 2)
    sse = np.sum((y - y_hat) ** 2)

    # Handle degenerate case where y is constant
    if sst == 0:
        # If the model predicts perfectly, return 1.0; else 0.0
        return float(sse == 0)

    r2 = 1.0 - sse / sst
    # Numerical guard: clamp to [0,1] due to floating-point noise
    return float(np.clip(r2, 0.0, 1.0))