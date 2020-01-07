"""
Burak Himmetoglu, 2019

Data scalers
"""

def standardize(X):
    """ Standardize data set by mean and standard deviation """
    mu = X.mean(axis=0, keepdims=True)
    s = X.std(axis=0, keepdims=True)
    return (X-mu)/s