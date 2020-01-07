"""
Burak Himmetoglu, 2019

Testers for ridge regression
"""

from sklearn.datasets import make_regression
from mlbook.utils.losses import mse 
from mlbook.linear_regression.ridge import *
from mlbook.utils.scalers import standardize
from mlbook.linear_regression.common import init_coef

if __name__ == "__main__":
    ns, nf = 500, 5
    X, Y = make_regression(n_samples = ns, n_features=nf, random_state=1)
    X = standardize(X)

    # Grad descent
    beta0_i, beta_i = init_coef(X, Y) # From univariate fit
    fit_summary = fit(X, Y, lam=0.1/(2*ns), lr=0.1, tol=1e-4, maxIter=200,
                      initBeta0 = beta0_i, initBeta = beta_i)
    print(fit_summary)
    Yhat = predict(X, fit_summary['beta0Hat'], fit_summary['betaHat'])
    mse_ = mse(Y, Yhat) 
    print("Ridge Grad. descent: MSE = {:.4f}".format(mse_))

    # Backtrace line search
    fit_summary2 = fit_bl(X, Y, lam = 0.1/(2*ns), tol=1e-4, maxIter=200,
                          initBeta0 = beta0_i, initBeta = beta_i)
    print(fit_summary2)