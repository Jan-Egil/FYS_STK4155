import numpy as np

def sigmoid(z):
    return(1/(1+np.exp(-z)))

class NeuralNetwork:
    def __init__(self, X, Y, hidden_neurons, hidden_layers, epochs, batch_size, gamma, lmbd):
        self.X_data_full = X
        self.Y_data_full = Y

        self.inputs = X.shape[0]
        self.features = X.shape[1]
        self.hidden_neurons = hidden_neurons
        self.hidden_layers = hidden_layers
        self.n_outputs = 1

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.inputs // self.batch_size
        self.gamma = gamma
        self.lmbd = lmbd

        self.create_bias_and_weights()

    def create_bias_and_weights(self):
        self.weights_i = np.random.randn(self.features, self.hidden_neurons)
        self.bias_i = np.zeros(self.hidden_neurons) + 0.01

        self.weights_h = np.random.randn(self.hidden_layers-1, self.hidden_neurons, self.hidden_neurons)
        self.bias_h = np.zeros((self.hidden_layers-1, self.hidden_neurons)) + 0.01

        self.weights_o = np.random.randn(self.hidden_neurons, self.n_outputs)
        self.bias_o = np.zeros(self.n_outputs) + 0.01

    def feed_forward(self):
        self.a_h = np.zeros((self.hidden_layers, self.input_train, self.hidden_neurons))

        self.z_i = self.X_data@self.weights_i + self.bias_i
        self.a_h[0] = sigmoid(self.z_i)

        for i in range(self.hidden_layers-1):
            self.z_h = self.a_h[i]@self.weights_h[i] + self.bias_h[i]
            self.a_h[i+1] = sigmoid(self.z_h)

        self.z_o = self.a_h[-1]@self.weights_o + self.bias_o
        #exp_term = np.exp(self.z_o)
        #self.outputs = exp_term/np.sum(np.exp(self.z_o), axis=1, keepdims=True)
        self.outputs = sigmoid(self.z_o).ravel()

    def feed_forward_out(self, X):
        z_i = X@self.weights_i + self.bias_i
        a_h = sigmoid(z_i)

        for i in range(self.hidden_layers-1):
            z_h = a_h@self.weights_h[i] + self.bias_h[i]
            a_h = sigmoid(z_h)

        z_o = a_h@self.weights_o + self.bias_o
        #exp_term = np.exp(z_o)
        #outputs = exp_term/np.sum(np.exp(z_o), axis=1, keepdims=True)
        outputs = sigmoid(z_o).ravel()

        return outputs

    def back_propagation(self):
        error_o = self.outputs-self.Y_data
        error_o = np.expand_dims(error_o, axis=1)
        error_h = error_o@self.weights_o.T * self.a_h[-1] *(1-self.a_h[-1])
        #print(error_o, error_h)

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
            error_h = error_h@self.weights_h[j].T*self.a_h[j]*(1-self.a_h[j])

            self.weight_gradient_h = self.a_h[j-1].T@error_h
            self.bias_gradient_h = np.sum(error_h, axis=0)

            if self.lmbd > 0.0:
                self.weight_gradient_h += self.lmbd*self.weights_h[j-1]

            self.weights_h[j-1] -= self.gamma*self.weight_gradient_h
            self.bias_h[j-1] -= self.gamma*self.bias_gradient_h

        error_i = error_h@self.weights_h[0].T*self.a_h[0]*(1-self.a_h[0])

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

        self.X_data = self.X_data_full
        self.Y_data = self.Y_data_full

        self.input_train = self.X_data.shape[0]

        self.feed_forward()
        self.back_propagation()

        '''for i in range(self.epochs):
            for j in range(self.iterations):
                random_indeces = np.random.choice(indeces, size=self.batch_size, replace=False)
                self.X_data = self.X_data_full[random_indeces]
                self.Y_data = self.Y_data_full[random_indeces]

                self.input_train = self.X_data.shape[0]

                self.feed_forward()
                self.back_propagation()'''

if __name__ == '__main__':
    #testing the neural network on sin^2(x)
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    def y(x):
        return np.sin(x)**2

    x = np.linspace(0,1,1000)
    def X(x, p):
        n = len(x)
        X = np.zeros((n, p+1))
        for i in range(len(x)):
            for j in range(p+1):
                X[i,j] = x[i]**j
        return X




    #plt.show()
    X_vals = X(x, 2)
    y_vals = y(x)
    #print(X_vals)
    #print(y_vals)

    # one-liner from scikit-learn library
    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(X_vals, y_vals, train_size=train_size,
                                                        test_size=test_size)

    FFNN = NeuralNetwork(X_train, Y_train, hidden_neurons=25, hidden_layers=3, epochs=20, batch_size=100, gamma=0.01, lmbd=0.1)
    Y_before = FFNN.predict(X_vals)
    FFNN.train()
    Y_vals = FFNN.predict(X_vals)

    error = Y_vals-y_vals

    plt.plot(x,y_vals, 'g-', label='true functions')
    plt.plot(x, Y_vals, 'r-', label='after train')
    plt.plot(x, Y_before, 'b-', label='before train')
    plt.legend()
    plt.show()
