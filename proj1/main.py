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
import scipy.stats as st
from func import * #Really nasty syntax

#print(__doc__)

"""
Defining different functions to be used
"""

print("Which task do you want to run?")
exercise = input("Press any letter between a & g: ")

"""
Part a)
"""

if exercise == "a":
    N = 1000
    p = int(input("Enter the degree of polynomial you want to approximate: ")) #Order of polynomial
    noise = 0.1 #Factor of noise in data

    xy = np.random.rand(N,2)
    x = xy[:,0]; y = xy[:,1]


    X = DesignMatrixCreator_2dpol(p,x,y)
    z = frankefunc_noise(x,y,noise)

    X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train); X_train_scaled[:,0] = 1
    X_test_scaled = scaler.transform(X_test); X_train_scaled[:,0] = 1


    beta = np.linalg.pinv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ z_train

    z_tilde_train = X_train_scaled @ beta
    z_tilde_test = X_test_scaled @ beta

    MSE_train_scikit = metric.mean_squared_error(z_train,z_tilde_train)
    R2_train_scikit = metric.r2_score(z_train,z_tilde_train)

    MSE_test_scikit = metric.mean_squared_error(z_test,z_tilde_test)
    R2_test_scikit = metric.r2_score(z_test,z_tilde_test)

    MSE_train = MSE(z_train,z_tilde_train)
    R2_train = R2(z_train,z_tilde_train)

    MSE_test = MSE(z_test,z_tilde_test)
    R2_test = R2(z_test,z_tilde_test)

    print("\n-------------------------R2-Score-----------------------------------\n")
    print("The R2 score for the training data is %e using SKLearn" % R2_train_scikit)
    print("The R2 score for the training data is %e using own defined function" % R2_train)
    print(" ")
    print("The R2 score for the test data is %e using SKLearn" % R2_test_scikit)
    print("The R2 score for the test data is %e using own defined function" % R2_test)
    print("\n-------------------------MSE-Score----------------------------------\n")
    print("The MSE score for the training data is %e using SKLearn" % MSE_train_scikit)
    print("The MSE score for the training data is %e using own defined function" % MSE_train)
    print(" ")
    print("The MSE score for the test data is %e using SKLearn" % MSE_test_scikit)
    print("The MSE score for the test data is %e using own defined function" % MSE_test)

"""
Part b)
"""

if exercise == "b":
    print("type 'a' for complexity vs error plot, type 'b' for bootstrap bias-variance analysis.")
    specifics = input("Type here: ")
    if specifics == "a":
        MaxPoly = 20
        N = 200
        noise = 0.3
        testsize = 0.2

        xy = np.random.rand(N,2)
        x = xy[:,0]; y = xy[:,1]
        z = frankefunc_noise(x,y,noise)

        MSE_train_array = np.zeros(MaxPoly)
        MSE_test_array = np.zeros(MaxPoly)

        for polydeg in range(1,MaxPoly+1):
            X = DesignMatrixCreator_2dpol(polydeg,x,y)
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=testsize)

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train);X_train_scaled[:,0] = 1
            X_test_scaled = scaler.transform(X_test);X_test_scaled[:,0] = 1

            beta_optimal = np.linalg.pinv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ z_train

            z_tilde_train = X_train_scaled @ beta_optimal
            z_tilde_test = X_test_scaled @ beta_optimal

            MSE_train = MSE(z_train,z_tilde_train)
            MSE_test = MSE(z_test,z_tilde_test)

            MSE_train_array[polydeg-1] = (MSE_train)
            MSE_test_array[polydeg-1] = (MSE_test)

        polydeg_array = np.arange(1,MaxPoly+1)
        plt.plot(polydeg_array,MSE_train_array,label="Train")
        plt.plot(polydeg_array,MSE_test_array,label="Test")
        plt.xlabel("'Complexity' of model (Polynomial degree)",fontsize="large")
        plt.ylabel("Mean Squared Error (MSE)",fontsize="large")
        plt.title("N = %i, test size = %.1f%%, noise = %.2f"% (N,testsize*100,noise),fontsize="x-large")
        plt.legend(); plt.grid(); plt.semilogy()
        plt.show()
    elif specifics == "b":
        pass

"""
Part c)
"""

if exercise == "c":
    print("You've come a long way to start with this exercise, haven't you..?")
