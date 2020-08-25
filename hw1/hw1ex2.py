import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

x = np.random.rand(100,1)
y = 5*x**2+2+0.1*np.random.randn(100,1)

parametrization = np.polyfit(x[:,0],y[:,0],2)
print(parametrization)

def poly2(p):
    x_poly = np.linspace(0,1,100)
    return p[0]*x_poly**2 + p[1]*x_poly + p[2]

x_poly = np.linspace(0,1,100)

X = np.zeros([len(x),3])
X[:,0] = 1
X[:,1] = x[:,0]
X[:,2] = x[:,0]**2
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
print(beta)

plt.plot(x,y,"o")
plt.plot(x_poly,poly2(parametrization),"--")
plt.show()
