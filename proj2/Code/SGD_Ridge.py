from func import *

lambvals = np.logspace(-8,-1,8)
learning_rates = np.logspace(-3,0,4)

polydeg = 15 #Degree of polynomial fit
N = 2000 #Number of data points
noise = 0.2 #Noise in data

xy = np.random.rand(N,2) #Create random function parameters
x = xy[:,0]; y = xy[:,1]

X = DesignMatrixCreator_2dpol(polydeg,x,y) #Create design matrix
z = frankefunc_noise(x,y,noise) #Corresponding Franke Function val w/ noise

X_train, X_test, zTrain, zTest = train_test_split(X,z,test_size=0.2) #Split data into training and testing set
X_train, X_test = scale(X_train, X_test) #Properly scale the data

matrixplot = np.zeros([lambvals.shape[0],learning_rates.shape[0]])

M = 2 #Minibatch size
epochs = 200

for lambindex,lambval in enumerate(lambvals):
    print(lambindex)
    for learnindex,learning_rate in enumerate(learning_rates):
        #print(learnindex)
        theta_own_SGD_Ridge = SGD(X_train,zTrain,N,M,epochs,costfunc="Ridge",lamb=lambval,gamma=learning_rate)
        MSE_val = metric.mean_squared_error(zTest,X_test@theta_own_SGD_Ridge)
        matrixplot[lambindex,learnindex] = MSE_val

plt.matshow(matrixplot,cmap='gray',vmax=1)
plt.colorbar()
plt.xlabel("Learning Rates",fontsize="x-large")
plt.ylabel("$\lambda$",fontsize="x-large")
plt.title("MSE using SGD Ridge for different learning rates\nand different hyperparameter $\lambda$\n",fontsize="x-large")
plt.yticks(np.arange(lambvals.shape[0]),lambvals)
plt.xticks(np.arange(learning_rates.shape[0]),learning_rates,rotation=90)
plt.show()
