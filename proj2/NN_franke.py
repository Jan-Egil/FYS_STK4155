import numpy as np
from sklearn.model_selection import train_test_split
from NeuralNetwork import FFNeuralNetwork
from func import *
import matplotlib.pyplot as plt

N = 1000

noise = 0.1 #Factor of noise in data

xy = np.random.rand(N,2) #Create random function parameters
x = xy[:,0]; y = xy[:,1]

z = frankefunc_noise(x,y,noise)

def D2_desmat(x, y):
    n = len(x)
    matrix=np.zeros((n,2))
    for i in range(n):
        matrix[i,0] = x[i]
        matrix[i,1] = y[i]
    return matrix

X = D2_desmat(x,y)

hidden_neurons = 15
hidden_layers = 4
epochs = np.array([1000])
batch_size = 100
gamma = 0.01
lmbd = 0

X_train, X_test, z_train, z_test = train_test_split(X, z, train_size=0.8)

for i in range(len(epochs)):
    FFNN = FFNeuralNetwork(X_train, z_train, hidden_neurons, hidden_layers, epochs[i], batch_size, gamma, lmbd, activation_func='Sigmoid')
    z_prev = FFNN.predict(X_test)
    FFNN.train()
    z_pred = FFNN.predict(X_test)

    mse = MSE(z_test, z_pred)

plt.plot(epochs, mse)
plt.xscale('log')
plt.show()
print(mse)
