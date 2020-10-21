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



def SGD(X,n,M,epochs):
    """
    Stochastic Gradient Descent
    Input:
    X = design matrix
    n = number of datapoints
    M = Minibatch size
    epochs = number of iterations over minibatches
    """
    m = int(n/M)
    for i in range(1,epochs+1):
        for j in range(m):
            k = np.random.randint(m)
            gradient = 2/n*X.T@(X@beta-y)#Derivative of cost function


def learning_schedule(t):
    """
    Learning schedule that gradually reduces learning rate
    Input:
    t = epochs*m+i
    """
    return t0/(t+t1)
