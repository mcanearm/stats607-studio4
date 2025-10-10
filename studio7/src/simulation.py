import numpy as np
import math
import logging

logging.basicConfig(level=logging.INFO)


np.random.seed(0)


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

    n = math.ceil(p / aspect_ratio)

    X = np.random.multivariate_normal(np.zeros(p), covariance, n)
    return X


def generate_data(
    p=5, aspect_ratio=0.2, covariance=np.eye(5), degrees_of_freedom=5, SNR=0.5
):
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
