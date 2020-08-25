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
from sklearn.preprocessing import StandardScaler
import sys

#print(__doc__)

"""
Defining different functions to be used
"""

def R2(y,ytilde):
    """
    Takes an array of data points and corresponding predicted values. Calculates the R2 score

    Input:
    y: Array of actual datapoints
    ytilde: Array of preducted datapoints

    Output:
    R2_score: Self explanatory
    """
    if len(y) != len(ytilde):
        sys.exit(0)

    n = len(y)
    sample_mean = 0
    for i in range(n):
        sample_mean += y[i]
    sample_mean = sample_mean/n

    sum_above = 0
    sum_below = 0
    for i in range(n):
        sum_above += (y[i] - ytilde[i])**2
        sum_below += (y[i] - sample_mean)**2
    R2_score = 1 - (sum_above/sum_below)
    return R2_score

def MSE(y,ytilde):
    """
    Takes an array of data points and corresponding predicted values. Calculates the mean squared error

    Input:
    y: Array of actual datapoints
    ytilde: Array of predicted datapoints

    Output:
    Mean_Squared_Error: self-explanatory
    """
    if len(y) != len(ytilde):
        sys.exit(0)
    sum = 0
    n = len(y)
    for i in range(n):
        sum += (y[i]-ytilde[i])**2
    Mean_Squared_Error = sum/n
    return Mean_Squared_Error

def frankefunc_analytic(x,y):
    """
    Just your regular Franke function

    Input:
    x,y: position (x,y)

    Output:
    Franke Function value at position (x,y)
    """

    if len(x) != len(y):
        sys.exit(0)

    term1 = 0.75*np.exp(-((9*x-2)**2)/4 - ((9*x-2)**2)/4)
    term2 = 0.75*np.exp(-((9*x+1)**2)/49 - (9*y+1)/10)
    term3 = 0.5*np.exp(-((9*x-7)**2)/4 - ((9*y-3)**2)/4)
    term4 = 0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 - term4

def frankefunc_noise(x,y,noise):
    """
    FrankeFunc

    Input:
    x,y: position (x,y)
    noise: Factor of noise

    Output:
    Franke Function value at position (x,y) with added noise
    """
    if len(x) != len(y):
        sys.exit(0)

    N = len(x)
    term1 = 0.75*np.exp(-((9*x-2)**2)/4 - ((9*x-2)**2)/4)
    term2 = 0.75*np.exp(-((9*x+1)**2)/49 - (9*y+1)/10)
    term3 = 0.5*np.exp(-((9*x-7)**2)/4 - ((9*y-3)**2)/4)
    term4 = 0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 - term4 + noise*np.random.randn(N)

def DesignMatrixCreator_2dpol(p,x,y):
    """
    Creates a design matrix for 2 variable polynomial (x,y)

    Input:
    p: Degree of polynomial
    x, y: Array of x-values & y-values

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

    return X

"""
Part a)
"""

N = 1000
p = 5 #Order of polynomial
noise = 0.1 #Factor of noise in data

xy = np.random.rand(N,2)
x = xy[:,0]; y = xy[:,1]


X = DesignMatrixCreator_2dpol(p,x,y)
z = frankefunc_noise(x,y,noise)

X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)

"""
Need to understand how scaling works(!!!!)
"""
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train

z_tilde_train = X_train @ beta
z_tilde_test = X_test @ beta

MSE_train = metric.mean_squared_error(z_train,z_tilde_train)
R2_train = metric.r2_score(z_train,z_tilde_train)

MSE_test = metric.mean_squared_error(z_test,z_tilde_test)
R2_test = metric.r2_score(z_test,z_tilde_test)

R2_train2 = R2(z_test,z_tilde_test)

print(R2_test)
print(R2_train2)


"""
Part b)
"""
