"""
Burak Himmetoglu, 2019

Testers for logistic regression (elasticnet)
"""

from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from mlbook.logistic.elnet import prediction, fit_elnet
from mlbook.utils.scalers import standardize

if __name__ == "__main__":
    X, Y = make_classification(n_samples = 400, n_features=50, random_state=1, flip_y=0.05)
    X = standardize(X)
    
    # Coord descent
    W, b, newLoss = fit_elnet(X,Y, verbosity=1, lam=0.1, alpha=0.1)
    print(W)

    Yhat = prediction(X, W, b)
    auc = roc_auc_score(Y, Yhat) # Implement this yourself too
    print("ElasticNet Coord. descent: AUC = {:.4f}".format(auc))