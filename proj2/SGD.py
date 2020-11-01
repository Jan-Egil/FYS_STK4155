from func import * #importing everything from func.py, including external packages.

#This is the solution to part a of project 2

N = 200 #Number of data points
polydeg = 2#int(input("Enter the degree of polynomial you want to approximate: ")) #Order of polynomial
noise = 0.2 #Factor of noise in data

xy = np.random.rand(N,2) #Create random function parameters
x = xy[:,0]; y = xy[:,1]


X = DesignMatrixCreator_2dpol(polydeg,x,y) #Create design matrix

z = frankefunc_noise(x,y,noise) #

X_train, X_test, zTrain, zTest = train_test_split(X,z,test_size=0.2) #Split data into training and testing set
X_train, X_test = scale(X_train, X_test) #Properly scale the data

#p = 2
#n = 100#number of datapoints

#x = 2 * np.random.rand(n, 1)
#y = 4 + 3 * x + np.random.randn(n, 1)
#X = DesignMatrixCreator_1dpol(p,x) #Create design matrix

"""theta from own OLS"""

z_tilde_test, z_tilde_train, theta_ols = OLS(X_train, X_test, zTrain, zTest)
#y_xx,y_gg,theta_ols = OLS(X,X,y,y)
print('Theta from OLS: ', theta_ols,'\n')

"""sklearns SGD"""

sgd_reg = SGDRegressor(max_iter=1000, penalty=None, eta0=0.1)
#sgd_reg.fit(x, y.ravel())
sgd_reg.fit(X_train, zTrain.ravel())
print('Theta from sklearn SGD: ', sgd_reg.coef_,'\n')

"""Own SGD scheme"""
M = 2#Minibatch size
epochs = 50
Tolerance = 1e-10
print(X_train.shape[1])
theta_own_SGD = SGD(X_train,zTrain,N,M,epochs)
print('Theta from own SGD: ',theta_own_SGD)
