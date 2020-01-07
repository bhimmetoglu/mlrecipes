"""
Burak Himmetoglu, 2019

Testers for logistic regression
"""

from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from mlbook.logistic.L2 import prediction, fit_grad_desc
from mlbook.utils.scalers import standardize

if __name__ == "__main__":
    X, Y = make_classification(n_samples = 200, n_features=50, random_state=1)
    X = standardize(X)
    
    # Grad descent
    W, b, newLoss = fit_grad_desc(X,Y, verbosity=1, lam=0.1)
    print(W)
    Yhat = prediction(X, W, b)
    auc = roc_auc_score(Y, Yhat) # Implement this yourself too
    print("L2 Grad. descent: AUC = {:.4f}".format(auc))