"""
Burak Himmetoglu, 2019

Activation functions
"""

import numpy as np

def sigmoid(z):
    """ Numerically stable sigmoid function """
    return np.where(z >= 0, 
                    1 / (1 + np.exp(-z)), 
                    np.exp(z) / (1 + np.exp(z)))