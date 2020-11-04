from func import *
import time

t = time.process_time()
#do some stuff
elapsed_time = time.process_time() - t

lambvals = np.logspace(-10,-6,10)
learning_rates = np.logspace(-10,-6,10)

polydeg = 10 #Degree of polynomial fit
N = 200 #Number of data points
noise = 1 #Noise in data

xy = np.random.rand(N,2) #Create random function parameters
x = xy[:,0]; y = xy[:,1]

X = DesignMatrixCreator_2dpol(polydeg,x,y) #Create design matrix
z = frankefunc_noise(x,y,noise) #Corresponding Franke Function val w/ noise

X_train, X_test, zTrain, zTest = train_test_split(X,z,test_size=0.2) #Split data into training and testing set
X_train, X_test = scale(X_train, X_test) #Properly scale the data

matrixplot = np.zeros([lambvals.shape[0],learning_rates.shape[0]])

M = 2 #Minibatch size
epochs = 10*X.shape[1]

for lambindex,lambval in enumerate(lambvals):
    print(lambindex)
    for learnindex,learning_rate in enumerate(learning_rates):
        #print(learnindex)
        theta_own_SGD_Ridge = SGD(X_train,zTrain,N,M,epochs,costfunc="Ridge",lamb=lambval,gamma=learning_rate)
        MSE_val = metric.mean_squared_error(zTest,X_test@theta_own_SGD_Ridge)
        matrixplot[lambindex,learnindex] = MSE_val


plt.matshow(matrixplot)
plt.show()
