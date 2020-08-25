"""
=================Docstring=================
Just an illustration that docstrings actually work out pretty nicely

This is just printed to terminal when the program is run.

What we will choose to write here when the project is finished is beyond me at the moment.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as metric
import sys

print(__doc__)

def frankefunc(x,y):
    term1 = 0.75*np.exp(-((9*x-2)**2)/4 - ((9*x-2)**2)/4)
    term2 = 0.75*np.exp(-((9*x+1)**2)/49 - (9*y+1)/10)
    term3 = 0.5*np.exp(-((9*x-7)**2)/4 - ((9*y-3)**2)/4)
    term4 = 0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 - term4

def frankefunc_noise(x,y,noise):
    term1 = 0.75*np.exp(-((9*x-2)**2)/4 - ((9*x-2)**2)/4)
    term2 = 0.75*np.exp(-((9*x+1)**2)/49 - (9*y+1)/10)
    term3 = 0.5*np.exp(-((9*x-7)**2)/4 - ((9*y-3)**2)/4)
    term4 = 0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 - term4 + noise*np.random.randn(np.shape(x))

def DesignMatrixCreator_2dpol(p,x,y):
    """
    Creates a design matrix for 2D polynomial

    Input:
    N: Number of data points
    p: Degree of polynomial
    x: Array of x-values

    Output:
    X: Design matrix
    """

    if len(x) != len(y):
        sys.exit(0)

    N = len(x)
    num_of_terms = int(((p+1)*(p+2))/2)

    X = np.zeros([N,num_of_terms])
    X[:,0] = 1
    column = 1
    for i in range(1,p+1):
        for j in range(i+1):
            X[:,column] = (x**j)*(y**(i-j))
            column += 1

    del(column)
    print(X)
    return X

#Create N random pairs of x & y values
N = 100
xy = np.random.rand(N,2)
x = xy[:,0]; y = xy[:,1]

p = 2 #Order of polynomial

noise = 0.1
X = DesignMatrixCreator_2dpol(p,x,y)

"""
for i in range(1,p+1):
    X = np.zeros([len(x),i+1])

    for j in range(i+1):
        X[:,j] = (x**j)*(y**(i-j))

    z = frankefunc_noise(x,y,noise)
    m = np.linalg.lstsq(X,z)
    print(m)
"""
