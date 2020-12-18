from Neural_Network_Classes import Neural_Network_eigval
import autograd.numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

init_array = np.random.rand(6)

hidden_neurons = [25,25]
num_iter = 20
lmb = 0.01

ranmatrix = np.random.randn(init_array.shape[0],init_array.shape[0])
symmatrix = 0.5*(ranmatrix.T + ranmatrix)
print(symmatrix)

solver = Neural_Network_eigval(init_array,symmatrix,hidden_neurons,num_iter,lmb)
solver.train()
