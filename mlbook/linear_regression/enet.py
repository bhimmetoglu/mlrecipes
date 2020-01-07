"""
Burak Himmetoglu, 2019

ElasticNet Implementation with coordinate descent optimization
"""
import random
import numpy as np 
from mlbook.utils.losses import mse
from mlbook.linear_regression.common import predict

# -- Penalty term
def _penalty(beta, lam, al):
    p = 0.5 * (1-al) * np.sum(beta**2) + al * np.sum(np.abs(beta))
    return lam * p

# -- Loss function
def _loss(X, y, beta0, beta, lam, al):
    yHat = predict(X, beta0, beta)
    return 0.5 * mse(y, yHat) + _penalty(beta, lam, al)

# -- Update coefficients by coordinate descent
def _update_coef(c, X, y, beta0Hat, betaHat, s2, lam, al):
    _, p = X.shape
    yHat = predict(X, beta0Hat, betaHat)
    if c < p:
        yTilde = yHat - X[:,c] * betaHat[c]
        zc = np.mean((y - yTilde) * X[:,c])
        
        if np.abs(zc) > lam * al:
            if zc > 0:
                betaHat[c] = zc - lam * al
            else:
                betaHat[c] = zc + lam * al
            
            betaHat[c] /= (s2[c] + lam * (1 - al))
        else:
            betaHat[c] = 0.
    elif c == p:
        # Update bias term
        beta0Hat = np.mean(y - X @ betaHat)

    return beta0Hat, betaHat

# -- Fit by coordinate descent
def fit(X, y, lam, al, tol, maxIter, initBeta0 = None, initBeta = None):
    _, p = X.shape
    if initBeta is None:
        initBeta = np.random.randn(p)
    if initBeta0 is None:
        initBeta0 = 0.0

    beta0Hat, betaHat = initBeta0, initBeta
    curLoss = _loss(X, y, beta0Hat, betaHat, lam, al)
    if np.abs(curLoss) < tol:
        return beta0Hat, betaHat

    s2 = np.mean(X*X, axis=0)
    # Iterate over full coord. descent cycles
    for it in range(maxIter):
        # Loop over randomized coordinates
        randp = [x for x in range(p+1)]
        random.shuffle(randp)
        for c in randp:
            beta0Hat, betaHat = _update_coef(c, X, y, beta0Hat, betaHat, s2, lam, al)
            
            # New loss
            newLoss = _loss(X, y, beta0Hat, betaHat, lam, al)

            fit_summary = {'loss':curLoss,
                           'convIter':it,
                           'beta0Hat':beta0Hat,
                           'betaHat':betaHat}

        # Check convergence after one full cycle
        if np.abs(newLoss - curLoss) < tol:
            return fit_summary
        else:
            curLoss = newLoss
    
    print('No convergence in {:d} iterations'.format(it+1))
    return fit_summary
