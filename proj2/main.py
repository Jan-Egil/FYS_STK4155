
from func import *#importing everything from func.py, including external packages

print("\n\nWhich task do you want to run?")
exercise = 'a'#input("Press any letter between a & g: ")

"""
Part a)
"""

if exercise == "a":
    p = 2
    n = 100#number of datapoints

    x = 2 * np.random.rand(n, 1)
    y = 4 + 3 * x + np.random.randn(n, 1)

    X = DesignMatrixCreator_1dpol(p,x) #Create design matrix

    """theta from own OLS"""
    y_xx,y_gg,theta_ols = OLS(X,X,y,y)
    print('Theta from OLS: ', theta_ols,'\n')
    """sklearns SGD"""
    sgd_reg = SGDRegressor(max_iter=1000, penalty=None, eta0=0.1)
    sgd_reg.fit(x, y.ravel())
    print('Theta from sklearn SGD: ',sgd_reg.intercept_,' ', sgd_reg.coef_,'\n')

    """Own SGD scheme"""
    M = 1#Minibatch size
    #m = int(n/M)
    epochs = 50
    Tolerance = 1e-10

    theta_own_SGD = SGD(X,y,n,M,epochs)

    print('Theta from own SGD: ',theta_own_SGD)






















#
