import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import sys
import scipy.stats as st

from sklearn.model_selection import train_test_split, cross_val_score
import sklearn.metrics as metric
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.linear_model import Lasso, Ridge, SGDRegressor

def DesignMatrixCreator_1dpol(p,x):
    """
    Creates a design matrix for 1 variable polynomial x

    INPUT:
    p: Degree of polynomial
    x : Array of x-values

    OUTPUT:
    X: Design matrix
    """

    if len(x.shape) > 1:
        x = np.ravel(x)

    N = len(x)
    X = np.zeros([N,p])
    X[:,0] = 1
    column = 0
    for i in range(p):
        X[:,i] = (x**i)
        column += 1
    return X

def OLS(XTrain, XTest, yTrain, yTest):
    """
    Performs an Ordinary Least Squares (OLS) regression on a data set

    INPUT:
    XTrain: The training design matrix of the problem
    XTest: The testing design matrix of the problem
    yTrain: The data point sets corresponding to the respective points in the training design matrix
    yTest: The data point sets corresponding to the respective points in the test design matrix

    OUTPUT:
    ytildeTest: Approximated values corresponding to the test data
    ytildeTrain: Approximated values corresponding to the train data
    Beta_OLS_optimal: The optimal coefficient values for the best-fit polynomial in OLS
    """

    Beta_OLS_optimal = np.linalg.pinv(XTrain.T @ XTrain) @ XTrain.T @ yTrain

    ytildeTrain = XTrain @ Beta_OLS_optimal
    ytildeTest = XTest @ Beta_OLS_optimal

    return ytildeTest, ytildeTrain, Beta_OLS_optimal

def learning_schedule(t):
    """
    Learning schedule that gradually reduces learning rate
    Input:
    t = epochs*m+i
    """
    t0,t1 = 5,50
    return t0/(t+t1)

def SGD(X,y,n,M,epochs):
    """
    Stochastic Gradient Descent
    Input:
    X = design matrix
    n = number of datapoints
    M = Minibatch size
    epochs = number of iterations over minibatches
    """
    m = int(n/M)
    theta = np.random.randn(2,1) # random initialization
    for epoch in range(epochs):
        for j in range(m):
            k = np.random.randint(m)#index to pick random bin
            Xk = X[k:k+1]
            yk = y[k:k+1]
            gradient = 2*Xk.T@(Xk@theta-yk)#Derivative of cost function
            gamma = learning_schedule(epoch * m + j)
            theta = theta - gamma * gradient
    return(theta)

    """Kode er i stor grad basert på Geron sin tekstbok. """

















#
