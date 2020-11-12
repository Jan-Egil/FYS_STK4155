from sklearn.model_selection import train_test_split
from func import *
import matplotlib.pyplot as plt
from NeuralNetwork import FFNeuralNetwork
np.random.seed(5513)

N = 100

noise = 0.1 #Factor of noise in data

xy = np.random.rand(N,2) #Create random function parameters
x = xy[:,0]; y = xy[:,1]

z = frankefunc_noise(x,y,noise) #Create the data

def D2_desmat(x, y):
    """
    Creating the inputs for the neural network
    """
    n = len(x)
    matrix=np.zeros((n,2))
    for i in range(n):
        matrix[i,0] = x[i]
        matrix[i,1] = y[i]
    return matrix

X = D2_desmat(x,y)

# Choosing hyperparameters
hidden_neurons = 125
hidden_layers = 2
epochs = 10000
batch_size = 10
confusion_matrix = np.zeros([5,11])
gamma = 0.01
lmbd = 0

X_train, X_test, z_train, z_test = train_test_split(X, z, train_size=0.8)

#mse = np.zeros(5)

for i in range(1):
    for j in range(1):
        FFNN = FFNeuralNetwork(X_train, z_train, hidden_neurons, hidden_layers, epochs, batch_size, gamma, lmbd, out_func='Sigmoid')
        FFNN.train() # training the Nettwork
        z_pred = FFNN.predict(X_test) # predicting the test data
        z_predict = FFNN.predict(X_train[:2])
        print(z_train[:2], frankefunc_noise(X_train[:2,0], X_train[:2,1], 0))
        print(z_predict)

        print(MSE(z_train[:2], z_predict))
        confusion_matrix[i,j] = MSE(z_test, z_pred)
        #mse[i] = MSE(z_test, z_pred)
        #print(mse[i])
        print(MSE(z_test, z_pred))

'''FFNN = FFNeuralNetwork(X_train, z_train, hidden_neurons, hidden_layers, epochs, batch_size, gamma=0.001, lmbd=0.1, out_func='Leaky_RELU')
z_prev = FFNN.predict(X_train[:2])
FFNN.train()
z_pred = FFNN.predict(X_train[:2])
#z_predict = FFNN.predict(X_test)
print(z_train[:2], frankefunc_noise(X_train[:2,0], X_train[:2,1], 0))
print(z_pred)
print(z_prev)

print(MSE(z_train[:2], z_pred))'''

plt.matshow(confusion_matrix, cmap='gray', vmax=0.1)
plt.colorbar()
plt.show()

'''plt.title('MSE vs. # of hidden layers', fontsize='x-large')
plt.plot(hidden_layers, mse, 'r-')
plt.ylabel('MSE', fontsize='large')
plt.xlabel('hidden layers', fontsize='large')
plt.show()'''
