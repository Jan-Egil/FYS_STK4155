import tensorflow as tf
from tensorflow import keras
from autograd import jacobian,hessian,grad
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def trial_func(x,t,MOD):
    return (1-t)*np.sin(np.pi*x)+x*(1-x)*t*MOD

def cost_func(true,pred):

    return None

def keras_NN_PDE(neurons_array,activ_funcs,eta,lmb):
    model = keras.Sequential()
    model.add(keras.Input(shape=(2,)))
    for indx in range(neurons.shape[0]):
        model.add(keras.layers.Dense(neurons[indx], activation=activ_func[indx], kernel_regularizer=keras.regularizers.l2(lmb)))
    model.add(keras.layers.Dense(1))
    opt = keras.optimizers.SGD(lr=eta)
    model.compile(optimizer=opt, loss='mse')
    return model

neurons = np.array([20,10,5])
activ_func = ['sigmoid']*neurons.shape[0]
lmb = np.logspace(-5,1,7)
eta = np.logspace(-5,1,7)
points = np.random.uniform(0,1,[100,2])


model = keras_NN_PDE(neurons,activ_func,1e-3,1e-2)
a = model.predict(points)
print(a)
