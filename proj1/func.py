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
from sklearn.linear_model import Lasso, Ridge
import sklearn.linear_model as skl

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

    Z = st.t.ppf(1-alpha/2,var.shape[0]-1)
    l = mean - Z*sigma
    u = mean + Z*sigma
    return l,u

def variance_estimator(p,y,ytilde):
    """
    Estimation of unkown variance

    INPUT:
    p: polynomial degree
    y: Array of actual datapoints
    ytilde: Array of predicted datapoints

    OUTPUT:
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
        print('wrong')
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

    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

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

def Ridge(XTrain, XTest, yTrain, yTest,lamb,validate_testsize=0.2):
    """
    Performs Ridge regression on a data set and finds the optimal hyperparameter

    INPUT:
    XTrain: The scaled design matrix corresponding to the train values
    XTest: The scaled design matrix corresponding to the test values
    yTrain: The "true" data values of the train data
    yTest: The "true" data values of the test data
    lamb: an array of potential lambda values we are optimizing for
    validate_testsize: (Optional) The amount of the full data set should be set aside for testing

    OUTPUT:
    ytildeTest: Approximated values corresponding to the test data
    ytildeTrain: Approximated values corresponding to the train data
    Beta_Ridge_Optimal: The optimal coefficient values for the best-fit polynomial using Ridge
    optimalLambda: The optimal value for lambda (where the MSE value is the lowest)
    MSE_lamb: An array with the same length as lambda, contains
    """

    Beta_Ridge = np.zeros((len(lamb),XTrain.shape[1])); MSE_lamb = np.zeros(len(lamb))

    XTraining, XValidate, yTraining, yValidate = train_test_split(XTrain,yTrain,test_size=validate_testsize)

    for i,lambval in enumerate(lamb):
        Beta_Ridge[i,:] = np.linalg.pinv(XTraining.T @ XTraining + lambval * np.identity((XTraining.T @ XTraining).shape[0])) @ XTraining.T @ yTraining
        ytildeValidate = XValidate @ Beta_Ridge[i]

        MSE_lamb[i] = MSE(yValidate,ytildeValidate)

    optimalLambda = lamb[np.argmin(MSE_lamb)]
    Beta_Ridge_Optimal = Beta_Ridge[np.argmin(MSE_lamb)]

    ytildeTrain = XTrain @ Beta_Ridge_Optimal
    ytildeTest = XTest @ Beta_Ridge_Optimal



    """""""""""""""""""""""""""
    fjern etter ferdig testet!!!!!!!
    """""""""""""""""""""""""""
    I = np.identity((XTraining.T @ XTraining).shape[0])
    MSEPredict = np.zeros(len(lamb))
    MSEPredictSKL = np.zeros(len(lamb))
    MSETrain = np.zeros(len(lamb))
    for i,lambval in enumerate(lamb):
        lmb = lambval
        # add ridge
        clf_ridge = skl.Ridge(alpha=lmb).fit(XTraining, yTraining)
        yridge = clf_ridge.predict(XValidate)
        Ridgebeta = np.linalg.inv(XTraining.T @ XTraining+lmb*I) @ XTraining.T @ yTraining
        # and then make the prediction
        ytildeRidge = XTraining @ Ridgebeta
        ypredictRidge = XValidate @ Ridgebeta
        MSEPredict[i] = MSE(yValidate,ypredictRidge)
        MSEPredictSKL[i] = MSE(yValidate,yridge)
        MSETrain[i] = MSE(yTraining,ytildeRidge)

    return ytildeTest, ytildeTrain, Beta_Ridge_Optimal, optimalLambda, MSE_lamb, MSEPredict

def scale(xtrain, xtest):
    """
    Scales the data using the StandardScaler from SKLearn.

    INPUT:
    xtrain, xtest: Unscaled design matrices

    OUTPUT:
    xtrain_scaled, xtest_scaled: Scaled design matrices
    """

    scaler = StandardScaler()
    scaler.fit(xtrain)
    xtrain_scaled = scaler.transform(xtrain); xtrain_scaled[:,0] = 1
    xtest_scaled = scaler.transform(xtest); xtest_scaled[:,0] = 1

    return xtrain_scaled, xtest_scaled

def Func_Bootstrap(X_train,X_test,y_train,y_test,n, method):
    """
    Does the bootstrap resampling technique, and returns predicted values for test and train data.

    INPUT:
    X_train: Design matrix corresponding to training values
    X_test: Design matrix corresponding to test values
    y_train: Function values at points corresponding to train values for the design matrix
    y_test: Function values at points corresponding to test values for the design matrix
    n: Number of bootstrap iterations
    method: method of linear regression to be used ('OLS', 'Ridge', 'Lasso')

    OUTPUT:
    mse: The mean value of all of the calculated Mean Squared Errors of all bootstraps
    Bias: The mean value of all of the calculated Biases of all bootstraps
    variance: The mean value of all of the calculated Variances of all bootstraps
    """

    ytilde_test = np.empty((y_test.shape[0], n))

    for i in range(n):
        rand_idx = np.random.randint(0,len(X_train),len(X_train))
        X = X_train[rand_idx]
        Y = y_train[rand_idx]
        if method == 'OLS':
            ytilde_test[:,i] = OLS(X, X_test, Y, y_test)[0]

        elif method == 'Ridge':
            lambdavals = np.logspace(-3,5,200)
            ytilde_test[:,i] = Ridge(X, X_test, Y, y_test,lambdavals)[0]

        elif method == 'Lasso':
            lambdavals = np.logspace(-10,5,100)

            MSE_test_array = np.zeros(len(lambdavals))

            Y_tilde_test_array = np.zeros([len(lambdavals),X_test.shape[0]])

            for j,lamb in enumerate(lambdavals):
                clf = Lasso(alpha=lamb).fit(X,Y)

                Y_tilde_test_array[j] = clf.predict(X_test)
                MSE_test_array[j] = MSE(y_test,Y_tilde_test_array[i])

            ytilde_test[:,i] = Y_tilde_test_array[np.argmin(MSE_test_array)]

        else:
            print("You didn't specify a supported regression-method!")
            sys.exit(0)

    y_test = y_test[:,np.newaxis]


    mse = np.mean(np.mean((y_test-ytilde_test)**2, axis=1, keepdims=True))
    Bias = np.mean((y_test-np.mean(ytilde_test, axis=1, keepdims=True))**2)
    variance = np.mean(np.var(ytilde_test, axis=1, keepdims=True))

    return mse, Bias, variance

def func_cross_validation(polydeg,X,y,K, method):
    """
    Performs

    INPUT:
    polydeg: the degree of the polynomial
    X: the Design Matrix
    y: The corresponding data points to the Design Matrix.
    K: the amount of folds we will cross-validate
    method: method of linear regression to be used ('OLS', 'Ridge', 'Lasso')

    OUTPUT:
    MSE_mean: the mean of all the calculated MSE values for every fold.
    """

    beta_len = int((2+polydeg)*(1+polydeg)/2)
    X_split = np.array(np.array_split(X,K))
    Y_split = np.array(np.array_split(y,K))

    mse = np.zeros(K)

    for i in range(K): #Runs through every single fold
        X_test = X_split[i]
        Y_test = Y_split[i]
        X_train = np.concatenate((X_split[:i], X_split[(i+1):]))
        Y_train = np.concatenate((Y_split[:i], Y_split[(i+1):])).ravel()
        X_train = X_train.reshape(-1, beta_len)
        X_train, X_test = scale(X_train, X_test)
        if method == 'OLS':
            ypred = OLS(X_train, X_test, Y_train, Y_test)[0]

        elif method == 'Ridge':
            lambdavals = np.logspace(-10,5,100)
            ypred = Ridge(X_train,X_test,Y_train,Y_test,lambdavals)[0]

        elif method == 'Lasso':
            lambdavals = np.logspace(-10,5,100)

            MSE_test_array = np.zeros(len(lambdavals))
            Y_tilde_test_array = np.zeros([len(lambdavals),X_test.shape[0]])

            for j,lamb in enumerate(lambdavals):
                clf = Lasso(alpha=lamb).fit(X_train,Y_train)

                Y_tilde_test_array[j] = clf.predict(X_test)
                MSE_test_array[j] = MSE(Y_test,Y_tilde_test_array[j])

            ypred = Y_tilde_test_array[np.argmin(MSE_test_array)]

        else:
            print("You didn't specify a supported regression-method!")
            sys.exit(0)

        mse[i] = MSE(Y_test, ypred)

    return np.mean(mse)

if __name__ == '__main__':
    """
    Felt like adding a name-main section just in case it would prove useful.

    EDIT: It proved useful in debugging these functions. However the code used for debugging has been removed.
    """

    print("\nYou successfully decided to run the function file instead of the actual file..\n")
    print("Nice going.")
