import numpy as np
import math
import logging

logging.basicConfig(level=logging.INFO)


def generate_covariance_structure(rho, p):
    """
    Generates a covariance matrix with rho correlation between variables

    Arguments:
    __________________________________________
    rho - type float
        - correlation between variables

    p - type int
      - dimension of the data

    Returns:
    ___________________________________________
    covariance - pxp numpy darray
    """

    covariance = np.full((p, p), rho)
    np.fill_diagonal(covariance, 1)
    return covariance


def make_positive_definite(X, min_eig=1e-6):
    """
    Force a symmetric matrix X to be positive definite by adjusting its eigenvalues.

    Parameters
    ----------
    X : np.ndarray
        A square (n x n) symmetric matrix.
    min_eig : float, optional
        Minimum eigenvalue threshold. Eigenvalues smaller than this are lifted.

    Returns
    -------
    X_pd : np.ndarray
        A symmetric positive definite version of X.
    """
    # Ensure symmetry (important for numerical stability)
    X_sym = (X + X.T) / 2

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(X_sym)

    # Lift eigenvalues below the threshold
    eigvals[eigvals < min_eig] = min_eig

    # Reconstruct the positive definite matrix
    X_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Enforce symmetry again for good measure
    X_pd = (X_pd + X_pd.T) / 2

    return X_pd


def generate_covariates(p, aspect_ratio, covariance):
	"""
	Simulates the X, i.e. matrix of covariates from multivariate normal with mean 0 and given covariance

	Arguments:
	__________________________________________
	p - type int
	  - dimension of the data

	aspect_ratio - type float
			 - p/n

	covariance - type numpy darray
		   - the covariance matrix of the multivariate normal data

	Returns:
	___________________________________________
	X - nxp numpy darray
	"""

	n = math.ceil(p/aspect_ratio)

	X = np.random.multivariate_normal(np.zeros(p), covariance, n)
	return X


def generate_data(p, aspect_ratio, covariance, degrees_of_freedom, SNR):
    """
    Simulates the data (X,Y). Samples X from multivariate normal with mean 0 and given covariance. Y = X beta + Error

    Arguments:
    _____________________________________________
    p - type int
      - dimension of the data

    aspect ratio - type float
             - p/n

    covariance - type numpy darray
           - the covariance matrix of the multivariate normal data

    degrees_of_freedom - type int
              - degrees of freedom of t-distribution, measure of tail-heaviness


    SNR - type float
        - the signal strength (beta'X'X beta)/sigma^2

    Return:
     ______________________________________________
    (Y,X) tuple of numpy darrays
    """

    beta = np.random.multivariate_normal(np.zeros(p), np.identity(p))
    # beta = beta / np.linalg.norm(beta)

    n = math.ceil(p / aspect_ratio)

    X = generate_covariates(p, aspect_ratio, covariance)

    SNR_numerator = beta.T @ X.T @ X @ beta

    sigma2 = SNR_numerator / SNR
    sigma2 = (beta.T @ X.T @ X @ beta) / (n * SNR)
    sigma = math.sqrt(sigma2)

    error = sigma * np.random.standard_t(degrees_of_freedom, n)

    Y = X @ beta + error

    # X,y is a more common order for tuples, so use that here
    return (X, Y, beta)
