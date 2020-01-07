"""
Burak Himmetoglu, 2019

Splitters
"""

import random

# -- Train-test splitter 
def train_test(n, testRatio = 0.20, seed = 1):
    """ Train/test split (returns list of indices) 
        Returns indices of train,test samples
    """
    random.seed(seed)
    nts = int(n * testRatio)
    idx = random.sample(range(n), n)
    idx_ts, idx_tr = idx[:nts], idx[nts:]

    return idx_ts, idx_tr

# -- K-folds generator
def k_fold_gen(n, k, seed=1):
    """ K-folds split generator (yields lists of indices) """
    random.seed(seed)
    idx = random.sample(range(n), n)
    nf = n // k
    for i in range(k):
        if i == k-1:
            idx_vld, idx_tr = idx[i*nf:], idx[:i*nf]
        else:
            idx_vld, idx_tr = idx[i*nf:(i+1)*nf], idx[:i*nf]+idx[(i+1)*nf:]
        yield idx_vld, idx_tr
