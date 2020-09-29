import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import sys
import scipy.stats as st

from sklearn.model_selection import train_test_split
import sklearn.metrics as metric
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


def CI_normal(mean,var,alpha):
    """
    Calculates the confidence interval for a normally distributed set of values.

    INPUT:
    mean: Mean value of normal distribution
    var: Variance of normal distribution
    alpha: significance level (confidence level = 1 - alpha)

    OUTPUT:
    l: lower confidence boundary
    u: upper confidence boundary
    """

    sigma = np.sqrt(var)
    Z = st.norm.ppf(1-alpha/2)
    l = mean - Z*sigma
    u = mean + Z*sigma
    return l,u

def variance_estimator(p,y,ytilde):
    """
    Estimation of unkown variance

    Input:
    p: polynomial degree
    y: Array of actual datapoints
    ytilde: Array of predicted datapoints

    Returns:
    Estimated variance
    """
    if len(y) != len(ytilde):
        sys.exit(0)
    N = len(y)
    var_hat = 1/(N-p-1)*np.sum((y-ytilde)**2)#estimate variance of z
    return var_hat


def bias(fi,exp_ytilde):
    """
    Calculates the bias-value assosciated with the mean squared error (NOTE: Returns Bias^2)

    INPUT:
    fi: Actual function value at given points. (Eventually data points)
    exp_ytilde: Expectation values of the

    OUTPUT:
    Bias: Calculated Bias
    """

    sum = 0
    n = len(fi)
    for i in range(n):
        sum += (fi[i]-exp_ytilde)**2
    bias = sum/n
    return bias


def R2(y,ytilde):
    """
    Takes an array of data points and corresponding predicted values. Calculates the R2 score

    INPUT:
    y: Array of actual datapoints
    ytilde: Array of predicted datapoints

    OUTPUT:
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

    INPUT:
    y: Array of actual datapoints
    ytilde: Array of predicted datapoints

    OUTPUT:
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

    INPUT:
    y: Array of actual datapoints
    ytilde: Array of predicted datapoints

    OUTPUT:
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
    The Franke function evaluated at a given point (x,y)

    INPUT:
    x,y: position (x,y)

    OUTPUT:
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
    The Franke function evaluated at a given point (x,y) including noise.

    INPUT:
    x,y: position (x,y)
    noise: Factor of noise

    OUTPUT:
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

    INPUT:
    p: Degree of polynomial
    x, y: Array of x-values & y-values

    OUTPUT:
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

def bootstrap(X_train,X_test,y_train,y_test,n):
    """
    Does the bootstrap resampling technique, and returns predicted values for test and train data.

    INPUT:
    X_train: Design matrix corresponding to training values
    X_test: Design matrix corresponding to test values
    y_train: Function values at points corresponding to train values for the design matrix
    y_test: Function values at points corresponding to test values for the design matrix
    n: Number of bootstrap iterations

    OUTPUT:
    ytilde_train: Matrix with rows corresponding to predicted values for training data.
    ytilde_test: Matrix with rows corresponding to predicted values for test data
    """

    ytilde_train = np.zeros([y_train.shape[0],n])
    ytilde_test = np.zeros([y_test.shape[0],n])

    for i in range(n):
        x,y = resample(X_train,y_train) #Defaults at bootstrap resampling


        beta = np.linalg.pinv(x.T @ x) @ x.T @ y
        ytilde_train[:,i] = X_train @ beta
        ytilde_test[:,i] = X_test @ beta

    return ytilde_train,ytilde_test

def OLS(X,y,testsize=0.2):
    """
    Performs an Ordinary Least Squares (OLS) regression on a data set

    INPUT:
    X: The design matrix of the problem
    y: The data point sets corresponding to the respective points in the design matrix
    Testsize: The amount of the full data set should be set aside for testing

    OUTPUT:
    ytildeTest: Approximated values corresponding to the test data
    ytildeTrain: Approximated values corresponding to the train data
    yTest: The "true" data values of the test data
    yTrain: The "true" data values of the train data
    XTestScaled: The scaled design matrix corresponding to the test values
    XTrainScaled: The scaled design matrix corresponding to the train values
    Beta_OLS_optimal: The optimal coefficient values for the best-fit polynomial in OLS
    """

    XTrain, XTest, yTrain, yTest = train_test_split(X,y,test_size=testsize)
    scaler = StandardScaler()
    scaler.fit(XTrain)
    XTrainScaled = scaler.transform(XTrain); XTrainScaled[:,0] = 1
    XTestScaled = scaler.transform(XTest); XTestScaled[:,0] = 1

    Beta_OLS_optimal = np.linalg.pinv(XTrainScaled.T @ XTrainScaled) @ XTrainScaled.T @ yTrain

    ytildeTrain = XTrainScaled @ Beta_OLS_optimal
    ytildeTest = XTestScaled @ Beta_OLS_optimal

    return ytildeTest, ytildeTrain, yTest, yTrain, XTestScaled, XTrainScaled, Beta_OLS_optimal

def Ridge(X,y,lamb,testsize=0.2):
    """
    Performs Ridge regression on a data set and finds the optimal hyperparameter

    INPUT:
    X: The design matrix of the problem
    y the data point sets corresponding to the respective points in the design matrix
    lamb: an array of potential lambda values we are optimizing for
    Testsize: (Optional) The amount of the full data set should be set aside for testing

    OUTPUT:
    ytildeTest: Approximated values corresponding to the test data
    ytildeTrain: Approximated values corresponding to the train data
    yTest: The "true" data values of the test data
    yTrain: The "true" data values of the train data
    XTestScaled: The scaled design matrix corresponding to the test values
    XTrainScaled: The scaled design matrix corresponding to the train values
    Beta_Ridge_Optimal: The optimal coefficient values for the best-fit polynomial using Ridge
    optimalLambda: The optimal value for lambda (where the MSE value is the lowest)
    MSE_lamb: An array with the same length as lambda, contains
    """

    XTrain, XTest, yTrain, yTest = train_test_split(X,y,test_size=testsize)
    scaler = StandardScaler()
    scaler.fit(XTrain)
    XTrainScaled = scaler.transform(XTrain); XTrainScaled[:,0] = 1
    XTestScaled = scaler.transform(XTest); XTestScaled[:,0] = 1

    Beta_Ridge = np.zeros((len(lamb),X.shape[1])); MSE_lamb = np.zeros(len(lamb))
    XTraining, XValidate, yTraining, yValidate = train_test_split(XTrainScaled,yTrain,test_size=testsize)

    for i,lambval in enumerate(lamb):
        Beta_Ridge[i,:] = np.linalg.pinv(XTraining.T @ XTraining + lambval * np.identity((XTraining.T @ XTraining).shape[0])) @ XTraining.T @ yTraining

        ytildeValidate = XValidate @ Beta_Ridge[i]

        MSE_lamb[i] = MSE(yValidate,ytildeValidate)

    optimalLambda = lamb[np.argmin(MSE_lamb)]
    Beta_Ridge_Optimal = Beta_Ridge[np.argmin(MSE_lamb)]

    ytildeTrain = XTrainScaled @ Beta_Ridge_Optimal
    ytildeTest = XTestScaled @ Beta_Ridge_Optimal

    return ytildeTest, ytildeTrain, yTest, yTrain, XTestScaled, XTrainScaled, Beta_Ridge_Optimal, optimalLambda, MSE_lamb


def cross_validation(X_train,y_train,K):
    """
    Description

    INPUT:
    K: the amount of folds we will cross-validate

    OUTPUT:
    idk
    """


    pass

if __name__ == '__main__':
    """
    idk
    """
    pass
