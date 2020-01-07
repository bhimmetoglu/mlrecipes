"""
Burak Himmetoglu, 2019

Ridge Regression Implementation with gradient descent optimization
"""

import numpy as np 
from mlbook.utils.losses import mse
from mlbook.linear_regression.common import predict

# -- Gradients
def _grads(X, Y, beta0, beta, lam):
    N = len(X)
    dbeta0 = -np.mean(Y - X @ beta - beta0)
    dbeta = -np.dot(Y - X @ beta - beta0, X)/N + lam * beta
    return dbeta0, dbeta

# -- Loss function
def _loss(X, y, beta0, beta, lam):
    yHat = predict(X, beta0, beta)
    return 0.5 * (mse(y, yHat) + lam * beta @ beta)

# -- Update learning rate by backtracing line search
def _backtrace_line(X, y, beta0, beta, lam, c = 0.5, rho = 0.5):
    # initiate 
    lr = 1.
    dbeta0, dbeta = _grads(X, y, beta0, beta, lam)
    grad2 = dbeta @ dbeta + dbeta0 * dbeta0
    loss0 = _loss(X, y, beta0, beta, lam)
    # Find step size
    foundStep = False
    while not foundStep:
        beta_step = beta - lr * dbeta
        beta0_step = beta0 - lr * dbeta0
        phi = _loss(X, y, beta0_step, beta_step, lam)
        l = loss0 - c * lr * grad2
        # Shrink lr to satisfy sufficient decrease condition
        if phi > l:
            lr /= rho 
        else:
            foundStep = True
    
    # Return new beta0 & beta
    return beta0_step, beta_step

# -- Fit gradient descent
def fit(X, y, lam, lr, tol, maxIter, initBeta0 = None, initBeta = None):
    # Initiate
    _,p = X.shape
    if initBeta is None:
        initBeta = np.random.randn(p)
    if initBeta0 is None:
        initBeta0 = 0.0

    beta0Hat, betaHat = initBeta0, initBeta
    curLoss = _loss(X, y, beta0Hat, betaHat, lam)
    if np.abs(curLoss) < tol:
        return beta0Hat, betaHat

    # Iterate until convergence
    for it in range(maxIter):
        # Update coefficients
        dbeta0, dbeta = _grads(X, y, beta0Hat, betaHat, lam)
        beta0Hat -= lr * dbeta0
        betaHat -= lr * dbeta
        # New loss
        newLoss = _loss(X, y, beta0Hat, betaHat, lam)
        
        fit_summary = {'loss':curLoss,
                       'convIter':it,
                       'beta0Hat':beta0Hat,
                       'betaHat':betaHat}
        # Check convergence
        if np.abs(newLoss - curLoss) < tol:
            return fit_summary
        else:
            curLoss = newLoss
    print('No convergence in {:d} iterations'.format(it+1))

    return fit_summary

# -- Fit with backtrace line search (learning rate found inside loop)
def fit_bl(X, y, lam, tol, maxIter, initBeta0 = None, initBeta = None):
    # Initiate
    _,p = X.shape
    if initBeta is None:
        initBeta = np.random.randn(p)
    if initBeta0 is None:
        initBeta0 = 0.0

    beta0Hat, betaHat = initBeta0, initBeta
    curLoss = _loss(X, y, beta0Hat, betaHat, lam)
    if np.abs(curLoss) < tol:
        return beta0Hat, betaHat

    # Iterate until convergence
    for it in range(maxIter):
        # Update coefficients
        beta0Hat, betaHat = _backtrace_line(X,y,beta0Hat, betaHat, lam)

        # New loss
        newLoss = _loss(X, y, beta0Hat, betaHat, lam)
        
        fit_summary = {'loss':curLoss,
                       'convIter':it,
                       'beta0Hat':beta0Hat,
                       'betaHat':betaHat}
        # Check convergence
        if np.abs(newLoss - curLoss) < tol:
            return fit_summary
        else:
            curLoss = newLoss
    print('No convergence in {:d} iterations'.format(it+1))

    return fit_summary