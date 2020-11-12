import numpy as np

class FFNeuralNetwork:
    def __init__(self, X, Y, hidden_neurons, hidden_layers, epochs, batch_size, gamma, lmbd, out_func):
        self.X_data_full = X
        self.Y_data_full = Y

        self.inputs = X.shape[0]
        self.features = X.shape[1]
        self.hidden_neurons = int(hidden_neurons)
        self.hidden_layers = int(hidden_layers)
        self.n_outputs = 1

        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.iterations = self.inputs // self.batch_size
        self.gamma = gamma
        self.lmbd = lmbd
        self.out_func = out_func
        self.activ_func = 'Sigmoid'

        self.create_bias_and_weights()

    def create_bias_and_weights(self):
        self.weights_i = np.random.randn(self.features, self.hidden_neurons)*2/np.sqrt(self.features+self.hidden_neurons)
        self.bias_i = np.zeros(self.hidden_neurons) + 0.01

        self.weights_h = np.random.randn(self.hidden_layers-1, self.hidden_neurons, self.hidden_neurons)*2/np.sqrt(2*self.hidden_neurons)
        self.bias_h = np.zeros((self.hidden_layers-1, self.hidden_neurons)) + 0.01

        self.weights_o = np.random.randn(self.hidden_neurons, self.n_outputs)*2/np.sqrt(self.hidden_neurons+self.n_outputs)
        self.bias_o = np.zeros(self.n_outputs) + 0.01

    def activation_function(self, z):
        z = z.copy()
        results = np.zeros(z.shape)

        if self.activ_func == 'Sigmoid':
            return(1/(1+np.exp(-z)))
        if self.activ_func == 'RELU':
            for j in range(len(z)):
                for i in range(len(z[0])):
                    results[j,i] = max(0.0, z[j,i])
            return results
        if self.activ_func == 'Leaky_RELU':
            self.alpha = 0.01
            for j in range(len(z)):
                for i in range(len(z[0])):
                    if z[j,i] < 0.0:
                        results[j,i] = self.alpha*z[j,i]
                    else:
                        results[j,i] = z[j,i]
            return results

    def derivatives(self,z):
        results = np.zeros(z.shape)
        if self.activ_func == 'Sigmoid':
            return self.activation_function(z)*(1-self.activation_function(z))
        if self.activ_func == 'RELU':
            for j in range(len(z)):
                for i in range(len(z[0])):
                    if z[j,i] > 0:
                        results[j,i] = 1
            return results
        if self.activ_func == 'Leaky_RELU':
            self.alpha = 0.01
            for j in range(len(z)):
                for i in range(len(z[0])):
                    if z[j,i] < 0.0:
                        results[j,i] = self.alpha
                    else:
                        results[j,i] = 1
            return results

    def output_function(self, z):
        results = z.copy()
        if self.out_func == 'Sigmoid':
            return(1/(1+np.exp(-z)))
        if self.out_func == 'RELU':
            for j in range(len(z)):
                for i in range(len(z[0])):
                    results[j,i] = max(0.0, z[j,i])
            return results
        if self.out_func == 'Leaky_RELU':
            self.alpha = 0.01
            for j in range(len(z)):
                for i in range(len(z[0])):
                    if z[j,i] < 0.0:
                        results[j,i] = self.alpha*z[j,i]
            return z
        if self.out_func == 'SoftMax':
            exp_term = np.exp(z)
            return exp_term/np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward(self):
        self.a_h = np.zeros((self.hidden_layers, self.input_train, self.hidden_neurons))
        self.z_h = np.zeros((self.hidden_layers-1, self.input_train, self.hidden_neurons))

        self.z_i = self.X_data@self.weights_i + self.bias_i
        self.a_h[0] = self.activation_function(self.z_i)

        for i in range(self.hidden_layers-1):
            self.z_h[i] = self.a_h[i]@self.weights_h[i] + self.bias_h[i]
            self.a_h[i+1] = self.activation_function(self.z_h[i])

        self.z_o = self.a_h[-1]@self.weights_o + self.bias_o

        self.outputs = self.output_function(self.z_o).ravel()

    def feed_forward_out(self, X):
        z_i = X@self.weights_i + self.bias_i
        a_h = self.activation_function(z_i)

        for i in range(self.hidden_layers-1):
            z_h = a_h@self.weights_h[i] + self.bias_h[i]
            a_h = self.activation_function(z_h)

        z_o = a_h@self.weights_o + self.bias_o

        outputs = self.output_function(z_o).ravel()

        return outputs

    def back_propagation(self):
        error_o = np.expand_dims(2/self.n_outputs*(self.outputs-self.Y_data), axis=1)
        error_h = error_o@self.weights_o.T * self.derivatives(self.z_h[-1])

        self.weight_gradient_o = self.a_h[-1].T@error_o
        self.bias_gradient_o = np.sum(error_o, axis=0)

        self.weight_gradient_h = self.a_h[-2].T@error_h
        self.bias_gradient_h = np.sum(error_h, axis=0)

        if self.lmbd > 0.0:
            self.weight_gradient_o += self.lmbd*self.weights_o
            self.weight_gradient_h += self.lmbd*self.weights_h[-1]

        self.weights_o -= self.gamma*self.weight_gradient_o
        self.bias_o -= self.gamma*self.bias_gradient_o
        self.weights_h[-1] -= self.gamma*self.weight_gradient_h
        self.bias_h[-1] -= self.gamma*self.bias_gradient_h

        for j in range(self.hidden_layers-2, 0, -1):
            error_new = error_h@self.weights_h[j].T*self.derivatives(self.z_h[j-1])#*self.a_h[j]*(1-self.a_h[j])

            self.weight_gradient_h = self.a_h[j-1].T@error_new
            self.bias_gradient_h = np.sum(error_h, axis=0)

            if self.lmbd > 0.0:
                self.weight_gradient_h += self.lmbd*self.weights_h[j-1]

            self.weights_h[j-1] -= self.gamma*self.weight_gradient_h
            self.bias_h[j-1] -= self.gamma*self.bias_gradient_h
            error_h = error_new

        error_i = error_h@self.weights_h[0].T*self.derivatives(self.z_i)

        self.weight_gradient_i = self.X_data.T@error_i
        self.bias_gradient_i = np.sum(error_i, axis=0)

        if self.lmbd > 0.0:
            self.weight_gradient_i += self.lmbd*self.weights_i

        self.weights_i -= self.gamma*self.weight_gradient_i
        self.bias_i -= self.gamma*self.bias_gradient_i

    def predict(self,X):
        y = self.feed_forward_out(X)
        return y

    def train(self):
        indeces = np.arange(self.inputs)

        for i in range(self.epochs):#(self.epochs):#self.epochs
            for j in range(self.iterations):#(self.iterations):#self.iterations
                random_indeces = np.random.choice(indeces, size=self.batch_size, replace=False)

                self.X_data = self.X_data_full[random_indeces]
                self.Y_data = self.Y_data_full[random_indeces]

                self.input_train = self.X_data.shape[0]

                self.feed_forward()
                self.back_propagation()


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from func import *
    import matplotlib.pyplot as plt
    np.random.seed(5513)

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

    hidden_neurons = 125
    hidden_layers = 2
    epochs = 10000
    batch_size = 40
    confusion_matrix = np.zeros([6,11])
    gamma = 0.0001
    lmbd = 0

    X_train, X_test, z_train, z_test = train_test_split(X, z, train_size=0.8)

    for i in range(1):
        for j in range(1):
            FFNN = FFNeuralNetwork(X_train, z_train, hidden_neurons, hidden_layers, epochs, batch_size, gamma, lmbd, out_func='Sigmoid')
            z_prev = FFNN.predict(X_train[:2])
            FFNN.train()
            z_pred = FFNN.predict(X_test)
            z_predict = FFNN.predict(X_train[:2])
            print(z_train[:2], frankefunc_noise(X_train[:2,0], X_train[:2,1], 0))
            print(z_predict)
            print(z_prev)

            print(MSE(z_train[:2], z_predict))
            print(MSE(z_test, z_pred))
            confusion_matrix[i,j] = MSE(z_test, z_pred)

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
