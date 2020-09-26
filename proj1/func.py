import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as metric
from sklearn.preprocessing import StandardScaler
import sys
import scipy.stats as st

def CI_normal(mean,var,alpha=0.95):
    """
    Calculates the confidence interval for a normally distributed set of values.

    Input:
    mean: Mean value of normal distribution
    var: Variance of normal distribution
    alpha: (Explain) + (Set to (BLANK))

    Output:
    l: lower confidence boundary
    u: upper confidence boundary
    """

    sigma = np.sqrt(var)
    Z = st.norm.ppf(1-alpha/2)
    l = mean - Z*sigma
    u = mean + Z*sigma
    return l,u


def bias(fi,exp_ytilde):
    """
    Calculates the bias-value assosciated with the mean squared error

    Input:
    fi: Actual function value at given points. (Eventually data points)
    exp_ytilde: Expectation values of the

    Output:
    Bias: Calculated Bias
    """

    sum = 0
    n = len(fi)
    for i in range(n):
        sum += (f[i]-exp_ytilde)**2
    bias = sum/n
    return bias


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


def ME(y,ytilde):
    """
    Takes an array of data points and corresponding predicted values. Calculates the mean (absolute) error

    Input:
    y: Array of actual datapoints
    ytilde: Array of predicted datapoints

    Output:
    Mean_Error: self-explanatory
    """
    if len(y) != len(ytilde):
        sys.exit(0)
    sum = 0
    n = len(y)
    for i in range(n):
        sum += np.abs((y[i]-ytilde[i]))
    Mean_Error = sum/n
    return Mean_Error


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

    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

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
