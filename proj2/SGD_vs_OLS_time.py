from func import * #importing everything from func.py, including external packages.
import time

#This is the solution to part a) of project 2

t = time.process_time()
#do some stuff
elapsed_time = time.process_time() - t



n_poly = 20
MSE_OLS_array = np.zeros(n_poly)
t_OLS_array = np.zeros(n_poly)
MSE_SGD_array = np.zeros(n_poly)
t_SGD_array = np.zeros(n_poly)

polydegs = np.arange(1,n_poly+1)

for polydeg in polydegs:
    print(polydeg)
    N = 200 #Number of data points
    noise = 1 #Factor of noise in data

    xy = np.random.rand(N,2) #Create random function parameters
    x = xy[:,0]; y = xy[:,1]


    X = DesignMatrixCreator_2dpol(polydeg,x,y) #Create design matrix

    z = frankefunc_noise(x,y,noise) #Corresponding Franke Function val w/ noise

    X_train, X_test, zTrain, zTest = train_test_split(X,z,test_size=0.2) #Split data into training and testing set
    X_train, X_test = scale(X_train, X_test) #Properly scale the data

    """
    theta from own OLS
    """
    t0 = time.time_ns()
    z_tilde_test = OLS(X_train, X_test, zTrain, zTest)[0]
    t_OLS_array[polydeg-1] = (time.time_ns() - t0)/(1e9)
    print(t_OLS_array[polydeg-1])
    """
    Own SGD scheme
    """

    M = 2 #Minibatch size
    epochs = 10*X.shape[1]
    Tolerance = 1e-10
    t1 = time.process_time()
    theta_own_SGD = SGD(X_train,zTrain,N,M,epochs)
    t_SGD_array[polydeg-1] = time.process_time() - t1

    MSE_OLS = metric.mean_squared_error(zTest,z_tilde_test)
    MSE_SGD_own = metric.mean_squared_error(zTest,X_test@theta_own_SGD)

    MSE_OLS_array[polydeg-1] = MSE_OLS
    MSE_SGD_array[polydeg-1] = MSE_SGD_own

plt.figure()
plt.plot(t_OLS_array,MSE_OLS_array,label="OLS")
plt.figure()
plt.plot(t_SGD_array,MSE_SGD_array,label="SGD")
plt.grid(); plt.legend(); plt.semilogy(); plt.semilogx()
plt.xlabel("Time spent calculating (s)")
plt.ylabel("MSE")
plt.show()
