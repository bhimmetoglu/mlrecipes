"""
Burak Himmetoglu, 2019

Common utilities for regression
"""
import numpy as np

# -- Predict
def predict(X, beta0, beta):
    return beta0 + X @ beta

# -- Initiate coefficients
def init_coef(X, y):
    beta0Hat = np.mean(y)
    betaHat = np.dot(y, X) / np.sum(X*X, axis=0)
    return beta0Hat, betaHat