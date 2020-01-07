"""
Burak Himmetoglu, 2019

Testers for elasticnet regression
"""

from sklearn.datasets import make_regression
from mlbook.utils.losses import mse 
from mlbook.linear_regression.enet import *
from mlbook.utils.scalers import standardize
from mlbook.linear_regression.common import init_coef

if __name__ == "__main__":
    ns, nf = 500, 5
    X, Y = make_regression(n_samples = ns, n_features=nf, random_state=1)
    X = standardize(X)
    
    # ElasticNet
    beta0_i, beta_i = init_coef(X, Y) # From univariate fit
    fit_summary = fit(X, Y, lam=0.01, al=0.5, tol=1e-6, maxIter=500,
                      initBeta0 = beta0_i, initBeta = beta_i)
    print(fit_summary)
    Yhat = predict(X, fit_summary['beta0Hat'], fit_summary['betaHat'])
    mse_ = mse(Y, Yhat) 
    print("ElasticNet coord. descent: MSE = {:.4f}".format(mse_))

