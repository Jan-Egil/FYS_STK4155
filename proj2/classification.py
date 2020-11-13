import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from NeuralNetwork import FFNeuralNetwork

digits = datasets.load_digits()

images = digits.data; targets = digits.target

img_train, img_test, targets_train, targets_test = train_test_split(images, targets, train_size=0.8)

n = len(targets_train)
outputs = 10
y_train = np.zeros((n,outputs))

for i in range(n):
    y_train[i, targets_train[i]] = 1

def accuracy_score_np(y, t):
    return np.sum(y == t)/len(y)

epochs = 100
batch_size = 100
gamma = np.logspace(-6,-1,6)
lmbd = np.logspace(-6,-1,6)
lmbd = np.append([0],lmbd)
hidden_neurons = 50
hidden_layers = 3

confusion_matrix = np.zeros((len(gamma), len(lmbd)))

for i, gama in enumerate(gamma):
    for j, lmbda in enumerate(lmbd):
        FFNN = FFNeuralNetwork(img_train, y_train, hidden_neurons, hidden_layers, epochs, batch_size, gamma=gama, lmbd=lmbda, out_func='SoftMax', n_outputs=outputs)
        targets_prev = FFNN.classification(img_test)
        FFNN.train()
        targets_predict = FFNN.classification(img_test)
        print('learning rate: ', gama)
        print('Lambda: ', lmbda)
        print('Accuracy before train: ', accuracy_score_np(targets_prev, targets_test))
        print("Accuracy score on test set: ", accuracy_score_np(targets_predict, targets_test))
        confusion_matrix[i,j] = accuracy_score_np(targets_predict, targets_test)
        print('\n')

plt.matshow(confusion_matrix, cmap='gray', vmax=1, vmin=0.1)
plt.xlabel('$\lambda$', fontsize='large')
plt.ylabel('Learning rate, $\gamma$', fontsize='large')
plt.colorbar()
plt.show()

gamma = 0.1
lmbd = 0.001

FFNN = FFNeuralNetwork(img_train, y_train, hidden_neurons, hidden_layers, epochs, batch_size, gamma=gamma, lmbd=lmbd, out_func='SoftMax', n_outputs=outputs)
FFNN.train()
targets_predict = FFNN.classification(img_test)
confusion_matrix_nr = np.zeros((10,10))
print(targets_predict[40:60]) #Example set from predicted and test
print(targets_test[40:60])
print(np.sum(targets_predict[40:60] == targets_test[40:60]))
print("Accuracy score on test set: ", accuracy_score_np(targets_predict, targets_test))

for i in range(len(targets_predict)):
    confusion_matrix_nr[targets_predict[i], targets_test[i]] += 1

plt.matshow(confusion_matrix_nr, cmap='gray')
plt.xlabel('correct digits', fontsize='large')
plt.ylabel('predicted digits', fontsize='large')
plt.colorbar()
plt.show()
