import numpy as np

"""
Strong linear model in regression
    Y = X beta + eps, where eps~ N(0, sigma^2 I)
    Under the null where beta_1 = ... = beta_p = 0,
    the R-squared coefficient has a known distribution
    (if you have an intercept beta_0), 
        R^2 ~ Beta(p/2, (n-p-1)/2)
"""


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
    raise NotImplementedError

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
    tuple 
        (lower_bound, upper_bound) of the CI
    
    ....
    """
    raise NotImplementedError

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
    y = np.asarray(y).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
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