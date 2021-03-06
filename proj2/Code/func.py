import numpy as np
#from imageio import imread
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

def DesignMatrixCreator_2dpol(p,x,y):
    """
    Creates a design matrix for 2 variable polynomial (x,y)

    INPUT:
    p: Degree of polynomial
    x, y: Array of x-values & y-values

    OUTPUT:
    X: Design matrix
    """

    #if len(x) != len(y):
        #sys.exit(0)

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

def frankefunc_noise(x,y,noise):
    """
    The Franke function evaluated at a given point (x,y) including noise.

    INPUT:
    x,y: position (x,y)
    noise: Factor of noise

    OUTPUT:
    Franke Function value at position (x,y) with added noise
    """
    if len(x) != len(y):
        sys.exit(0)

    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    term1 = 0.75*np.exp(-((9*x-2)**2)/4 - ((9*y-2)**2)/4)
    term2 = 0.75*np.exp(-((9*x+1)**2)/49 - (9*y+1)/10)
    term3 = 0.5*np.exp(-((9*x-7)**2)/4 - ((9*y-3)**2)/4)
    term4 = 0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 - term4 + noise*np.random.randn(N)

def scale(xtrain, xtest):
    """
    Scales the data using the StandardScaler from SKLearn.

    INPUT:
    xtrain, xtest: Unscaled design matrices (We scale using the first input)

    OUTPUT:
    xtrain_scaled, xtest_scaled: Scaled design matrices
    """

    scaler = StandardScaler()
    scaler.fit(xtrain)
    xtrain_scaled = scaler.transform(xtrain); xtrain_scaled[:,0] = 1
    xtest_scaled = scaler.transform(xtest); xtest_scaled[:,0] = 1

    return xtrain_scaled, xtest_scaled

def learning_schedule(t,t0=5,t1=50):
    """
    Learning schedule that gradually reduces learning rate

    INPUT:
    t = epochs*m+i

    OUTPUT:
    Learning schedule
    """
    return t0/(t+t1)

def SGD(X,y,n,M,epochs,costfunc='OLS',lamb=0,gamma=0,classes=2):
    """
    Stochastic Gradient Descent
    Input:
    X = design matrix
    n = number of datapoints
    M = Minibatch size
    epochs = number of iterations over minibatches

    returns:
    "Optimal" parameters
    """

    """
    This code is largely based on the code in Gerons textbook, ch. 4
    """

    m = int(n/M)
    if costfunc == 'OLS':
        theta = np.random.randn(X.shape[1]) #Random initialization
        for epoch in range(epochs):
            for j in range(m):
                k = np.random.randint(n-M) #Index to pick random bin
                X_k = X[k:k+M]
                y_k = y[k:k+M]
                gradient = (2/n)*X_k.T@((X_k@theta)-y_k) #Derivative of (MSE) cost function
                gamma = learning_schedule(epoch * m + j)
                theta = theta - gamma * gradient
        return theta
    elif costfunc == 'Ridge':
        theta = np.random.randn(X.shape[1]) #Random initialization
        for epoch in range(epochs):
            for j in range(m):
                k = np.random.randint(n-M)
                X_k = X[k:k+M]
                y_k = y[k:k+M]
                gradient = (2/n)*X_k.T@((X_k@theta)-y_k) + 2*lamb*theta
                #gamma = learning_schedule(epoch * m + j)
                theta = theta - gamma * gradient
        return theta
    elif costfunc == 'Logistic':
        theta = np.random.randn(X.shape[1],classes) #Random initialization
        for epoch in range(epochs):
            print(epoch)
            gradient = np.zeros(classes)
            for j in range(n):
                likely = X[j]@theta
                for clas in range(classes):
                    likelyhood = likely[clas]/np.sum(likely)
                    gradient[clas] += X[j,clas]*((y[j]==clas)-likelyhood)
            theta = theta - gamma * gradient
        return theta




def logistic_sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(z):
    exp_term = np.exp(z-np.max(z))
    return exp_term/np.sum(np.exp(z-np.max(z)), axis=1, keepdims=True)

def activation_func(x,w,b,method):
    """
    input:
    1. x = function input
    2. w = wheights
    3. b = bias
    """
    if method == 'sigmoid':
        z = x @ w + b
        return logistic_sigmoid(z)
    if method == 'softmax':
        z = x @ w + b
        return softmax(z)
    else:
        return None

def FFNN(x,N_n,N_l,*args):
    """
    input:
    x = function input
    N_n = nodes in layer
    N_l = Number of layers
    Optional input args:
    1. w = wheights
    2. b = bias
    """
    if len(args) == 2:
        w = args[0]
        b = args[1]

    if (args) == ():
        w = np.random.rand(len(x),N_l)
        b = np.random.rand(N_l)

    for i in range(N_l):
        Z = activation_func(x,w,b,'sigmoid')

    predict = np.argmax(Z,axis = 1)
    y = 1
    return y

"""
Logistic Regression cost function (log loss)
"""
def log_reg_cost(X,Y,weights,beta,learning_rate):
    m = X.shape[0]
    p_hat = activation_func(X,weights,beta,'softmax') #probabilities
    J = -np.sum(Y*np.log(p_hat)) #cost
    dJ = -(X.T @ (-p_hat.T + Y.T).T)/m + learning_rate * weights #gradients

    return J,dJ

def onehotvec(Y):
    """Convert data to one hot format"""
    classes = np.arange(np.max(Y)+1)
    hotvec = np.zeros((len(Y),len(classes)))
    for i in range(len(Y)):
        hotvec[i,Y[i]] = 1
    return hotvec

def accuracy_score(y,t):
    n = len(t)
    sum = 0
    for i in range(n):
        sum += y[i] == t[i]
    return sum/n

def MSE(y, y_tilde):
    n = len(y)
    mse = np.sum((y-y_tilde)**2)
    return mse/n


#FFNN([1,2,3],4,1)











#
