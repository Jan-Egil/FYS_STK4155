import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as metric

x = np.random.rand(100,1)
y = 5*x**2+2+0.5*np.random.randn(100,1)

parametrization = np.polyfit(x[:,0],y[:,0],2)

def poly2(p):
    x_poly = np.linspace(0,1,100)
    return p[0]*x_poly**2 + p[1]*x_poly + p[2]

x_poly = np.linspace(0,1,100)

X = np.zeros([len(x),3])
X[:,0] = 1
X[:,1] = x[:,0]
X[:,2] = x[:,0]**2
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

ytilde_train = X_train @ beta

MSE_trainval = metric.mean_squared_error(y_train,ytilde_train)
R2_trainval = metric.r2_score(y_train,ytilde_train)

ytilde_test = X_test @ beta

MSE_testval = metric.mean_squared_error(y_test,ytilde_test)
R2_testval = metric.r2_score(y_test,ytilde_test)

print("MSE Train:")
print(MSE_trainval)
print("R2 Train:")
print(R2_trainval)

print("---------------------")

print("MSE Test:")
print(MSE_testval)
print("R2 Test:")
print(R2_testval)

plt.plot(x,y,"o")
plt.plot(x_poly,poly2(parametrization),"--")
plt.show()
