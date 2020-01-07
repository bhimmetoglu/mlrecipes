"""
Burak Himmetoglu, 2019

Loss functions
"""

import numpy as np

EPS = 1e-5 # Clipping factor for probabilities

def mse(Y, yHat):
    return np.mean( (Y-yHat)**2)

def binary_log_loss(Y, P):
    """
    Compute negative log loss
    """
    N = len(Y)
    # Clip values very close to 1 or 0
    P = np.clip(P, EPS, 1 - EPS)

    # Negative log likelihood function
    mask0 = (Y == 0) # label = 0 observations
    mask1 = (Y == 1) # label = 1 observations

    nll = -(np.log(P[mask1]).sum() + np.log(1-P[mask0]).sum())

    return nll/N
