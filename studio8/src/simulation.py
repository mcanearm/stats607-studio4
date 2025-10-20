import numpy as np
import math
import logging

logging.basicConfig(level=logging.INFO)


def generate_covariance_structure(rho, p):
    """
    Generates a covariance matrix with rho correlation between variables.

    Arguments
    ----------
    rho : float
        Correlation between variables.
    p : int
        Dimension of the data.

    Returns
    -------
    covariance : np.ndarray
        pxp covariance matrix.
    """
    covariance = np.full((p, p), rho)
    np.fill_diagonal(covariance, 1)
    return covariance


def generate_covariates(p, aspect_ratio, covariance, rng=None):
    """
    Simulates the X matrix of covariates from multivariate normal with mean 0 and given covariance.

    Arguments
    ----------
    p : int
        Dimension of the data.
    aspect_ratio : float
        Ratio p/n.
    covariance : np.ndarray
        Covariance matrix of the multivariate normal data.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    X : np.ndarray
        n x p matrix of covariates.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = math.ceil(p / aspect_ratio)
    X = rng.multivariate_normal(np.zeros(p), covariance, n)
    return X


def generate_data(
    p=5, rsq=0.5, aspect_ratio=0.2, covariance=None, degrees_of_freedom=5, SNR=0.5, 
    rng=None, sample_beta=False, noise_distribution="t", sigma2=None
):
    """
    Simulates the data (X, Y). Samples X from multivariate normal with mean 0 and given covariance.
    Y = Xβ + ε, where ε ~ t_df(0, σ²).

    Arguments
    ----------
    p : int
        Dimension of the data.
    aspect_ratio : float
        Ratio p/n.
    covariance : np.ndarray, optional
        Covariance matrix of the multivariate normal data.
    degrees_of_freedom : int
        Degrees of freedom of t-distribution.
    SNR : float
        Signal-to-noise ratio.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    (X, Y, beta) : tuple of np.ndarray
        Covariate matrix, response vector, and true coefficients.
    """
    if rng is None:
        rng = np.random.default_rng()

    if covariance is None:
        covariance = np.eye(p)

    if sample_beta:
        beta = rng.multivariate_normal(np.zeros(p), np.identity(p))
    else:
        beta = np.ones(p) * np.sqrt(rsq/p)
    
    # beta = beta / np.linalg.norm(beta)  # optional normalization

    X = generate_covariates(p, aspect_ratio, covariance, rng=rng)
    n = math.ceil(p / aspect_ratio)

    if sigma2 is None:
        signal_var = np.var(X @ beta)
        sigma2 = signal_var / SNR

    sigma = math.sqrt(sigma2)

    if noise_distribution == "standard normal":
        error =  rng.normal(0, 1, n)
    elif noise_distribution == "normal":
        error = sigma * rng.normal(0, 1, n)
    elif noise_distribution == "t":
        error = sigma * rng.standard_t(degrees_of_freedom, n)
    
    Y = X @ beta + error

    return (X, Y, beta)
