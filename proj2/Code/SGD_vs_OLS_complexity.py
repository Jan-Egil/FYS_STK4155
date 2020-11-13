from func import * #importing everything from func.py, including external packages.

#This is the solution to part a) of project 2

n_poly = 40
MSE_OLS_array = np.zeros(n_poly)
MSE_SGD_array = np.zeros(n_poly)
MSE_SGD_sklearn_array = np.zeros(n_poly)
polydegs = np.arange(1,n_poly+1)

for polydeg in polydegs:
    print(polydeg)
    N = 2000 #Number of data points
    noise = 0.2 #Factor of noise in data

    xy = np.random.rand(N,2) #Create random function parameters
    x = xy[:,0]; y = xy[:,1]


    X = DesignMatrixCreator_2dpol(polydeg,x,y) #Create design matrix

    z = frankefunc_noise(x,y,noise) #Corresponding Franke Function val w/ noise

    X_train, X_test, zTrain, zTest = train_test_split(X,z,test_size=0.2) #Split data into training and testing set
    X_train, X_test = scale(X_train, X_test) #Properly scale the data

    """
    theta from own OLS
    """

    z_tilde_test, z_tilde_train, theta_ols = OLS(X_train, X_test, zTrain, zTest)
    #y_xx,y_gg,theta_ols = OLS(X,X,y,y)
    #print('\nTheta from OLS: ', theta_ols,'\n')

    """
    sklearns SGD
    """

    sgd_reg = SGDRegressor(max_iter=200, penalty='l2', eta0=0.1,loss="squared_loss")
    #sgd_reg.fit(x, y.ravel())
    skl_sgd_model = sgd_reg.fit(X_train, zTrain)
    #print('Theta from sklearn SGD: ', a.coef_,'\n')

    """
    Own SGD scheme
    """

    M = 2 #Minibatch size
    epochs = 500

    theta_own_SGD = SGD(X_train,zTrain,N,M,epochs)

    #print('Theta from own SGD: ',theta_own_SGD)

    MSE_OLS = metric.mean_squared_error(zTest,z_tilde_test)
    MSE_SGD_own = metric.mean_squared_error(zTest,X_test@theta_own_SGD)
    MSE_SGD_SKLearn = metric.mean_squared_error(zTest,skl_sgd_model.predict(X_test))

    MSE_OLS_array[polydeg-1] = MSE_OLS
    MSE_SGD_array[polydeg-1] = MSE_SGD_own
    MSE_SGD_sklearn_array[polydeg-1] = MSE_SGD_SKLearn


plt.plot(polydegs,MSE_OLS_array,label="OLS")
plt.plot(polydegs,MSE_SGD_array,label="SGD")
#plt.plot(polydegs,MSE_SGD_sklearn_array,label="SKLearn")
plt.grid(); plt.legend(); plt.semilogy()
plt.xlabel("Complexity of model",fontsize="large")
plt.ylabel("MSE",fontsize="large")
plt.title("MSE against 'complexity of model'",fontsize="x-large")
plt.show()
