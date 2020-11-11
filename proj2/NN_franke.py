import numpy as np
from NeuralNetwork import FFNeuralNetwork
from func import *

N = 10

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

print(x)
print(D2_desmat(x, y))
