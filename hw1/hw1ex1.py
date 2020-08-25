import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#file = open('EoS.csv','r')

EoS = pd.read_csv('EoS.csv',names=["Density","Energy"])

rho = EoS['Density']

n = len(rho) #Number of datapoints
p = 4 #To define polynomial of degree p-1
X = np.zeros([n,p])

X[:,0] = 1
X[:,1] = rho
X[:,2] = rho**2
X[:,3] = rho**3
print(X)
