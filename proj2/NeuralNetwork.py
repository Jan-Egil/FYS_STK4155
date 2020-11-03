import numpy as np

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

    def sigmoid(self, z):
        return(1/(1+np.exp(-z)))

    def feed_forward(self):
        self.a_h = np.zeros((self.hidden_layers, self.input_train, self.hidden_neurons))

        self.z_i = self.X_data@self.weights_i + self.bias_i
        self.a_h[0] = self.sigmoid(self.z_i)

        for i in range(self.hidden_layers-1):
            self.z_h = self.a_h[i]@self.weights_h[i] + self.bias_h[i]
            self.a_h[i+1] = self.sigmoid(self.z_h)

        self.z_o = self.a_h[-1]@self.weights_o + self.bias_o
        #exp_term = np.exp(self.z_o)
        #self.outputs = exp_term/np.sum(np.exp(self.z_o), axis=1, keepdims=True)
        self.outputs = self.sigmoid(self.z_o)

    def feed_forward_out(self, X):
        z_i = X@self.weights_i + self.bias_i
        a_h = self.sigmoid(z_i)

        for i in range(self.hidden_layers-1):
            z_h = a_h@self.weights_h[i] + self.bias_h[i]
            a_h = self.sigmoid(z_h)

        z_o = a_h@self.weights_o + self.bias_o
        #exp_term = np.exp(z_o)
        #outputs = exp_term/np.sum(np.exp(z_o), axis=1, keepdims=True)
        outputs = self.sigmoid(z_o).ravel()

        return outputs

    def back_propagation(self):
        error_o = self.outputs-self.Y_data
        #error_o = np.expand_dims(error_o, axis=1)
        #print(self.outputs.shape, self.Y_data.shape, error_o.shape)
        error_h = error_o@self.weights_o.T * self.a_h[-1] *(1-self.a_h[-1])
        #print(error_o.shape, error_h.shape)
        #print("after error")

        self.weight_gradient_o = self.a_h[-1].T@error_o
        self.bias_gradient_o = np.sum(error_o, axis=0)

        self.weight_gradient_h = self.a_h[-2].T@error_h
        self.bias_gradient_h = np.sum(error_h, axis=0)

        if self.lmbd > 0.0:
            self.weight_gradient_o += self.lmbd*self.weights_o
            self.weight_gradient_h += self.lmbd*self.weights_h[-1]

        #print(self.weight_gradient_o.shape, self.weights_o.shape)
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

        for i in range(self.epochs):
            for j in range(self.iterations):
                random_indeces = np.random.choice(indeces, size=self.batch_size, replace=False)

                self.X_data = self.X_data_full[random_indeces]
                self.Y_data = self.Y_data_full[random_indeces]

                self.input_train = self.X_data.shape[0]

                self.feed_forward()
                self.back_propagation()

if __name__ == '__main__':
    #testing the neural network on sin^2(x)
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    def mse(y, y_tilde):
        n = len(y)
        mse = np.sum((y-y_tilde)**2)
        return mse/n

    def y(x):
        return np.sin(x**2)

    def D1_desmat(x):
        n = len(x)
        matrix=np.zeros((n,1))
        for i in range(n):
            matrix[i,0] = x[i]
        return matrix

    x = np.linspace(0,1,1000)
    X_1D_vals = D1_desmat(x)
    Y_1d_vals = y(x).reshape(1000,1)

    train_size1d = 0.8
    test_size1d = 1 - train_size1d
    X_train1d, X_test1d, Y_train1d, Y_test1d = train_test_split(X_1D_vals, Y_1d_vals, train_size=train_size1d,
                                                        test_size=test_size1d)

    FFNN1d = NeuralNetwork(X_train1d, Y_train1d, hidden_neurons=25, hidden_layers=4, epochs=1000, batch_size=100, gamma=0.01, lmbd=0.0)
    X_pred1d = np.sort(X_1D_vals, axis=0)
    Y_before1d = FFNN1d.predict(X_pred1d)
    FFNN1d.train()
    Y_vals1d = FFNN1d.predict(X_pred1d)

    plt.plot(X_pred1d,np.sort(Y_1d_vals,axis=0), 'g-', label='true functions')
    plt.plot(X_pred1d, Y_vals1d, 'r-', label='after train')
    plt.plot(X_pred1d, Y_before1d, 'b-', label='before train')
    plt.legend()
    plt.show()
    print('mse: ', mse(Y_vals1d, np.sort(Y_1d_vals,axis=0).ravel()))

    print(" ")
    print("2d case: ")

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np

    def z(x,y):
        return np.sin(x**2) * np.cos(y**2)

    def designMatrix(x,y):
        x = x.ravel()
        y = y.ravel()
        matrix = np.zeros((len(x), 2))
        for i in range(len(x)):
            matrix[i,0] = x[i]
            matrix[i,1] = y[i]
        return matrix

    # Make data.
    X = np.array([0.9, 0.5])
    Y = np.array([0.0, 0.5])
    X, Y = np.meshgrid(X, Y)
    Z = z(X,Y)
    des_mat = designMatrix(X,Y)
    print(des_mat)

    X_vals = designMatrix(X,Y)
    y_vals = z(X,Y).reshape((4,1))

    # one-liner from scikit-learn library
    print(X_vals.shape, y_vals.shape)

    FFNN = NeuralNetwork(X_vals, y_vals, hidden_neurons=30, hidden_layers=4, epochs=1000, batch_size=1, gamma=0.05, lmbd=0.0)
    #X_pred = np.sort(X_vals, axis=0)
    Y_before = FFNN.predict(X_vals)
    FFNN.train()
    Y_vals = FFNN.predict(X_vals)

    print("one point fit")
    print(Y_before)
    print(Y_vals)
    print(y_vals.ravel())
    print(mse(Y_vals, y_vals.ravel()))

    print("multiple points")

    #1 2d point:

    #2D:

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(0,1,0.1)
    Y = np.arange(0,1,0.1)
    X, Y = np.meshgrid(X, Y)
    Z = z(X,Y)
    des_mat = designMatrix(X,Y)
    print(des_mat)


    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


    #plt.show()
    X_vals = designMatrix(X,Y)
    y_vals = z(X,Y).reshape((100,1))

    # one-liner from scikit-learn library
    train_size = 0.8
    test_size = 1 - train_size
    print(X_vals.shape, y_vals.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X_vals, y_vals, train_size=train_size)
    print(X_train.shape, Y_train.shape)

    FFNN = NeuralNetwork(X_train, Y_train, hidden_neurons=35, hidden_layers=4, epochs=12500, batch_size=10, gamma=0.05, lmbd=0.0)
    #X_pred = np.sort(X_vals, axis=0)
    Y_before = FFNN.predict(X_vals[46])
    FFNN.train()
    Y_vals = FFNN.predict(X_vals[46])

    print("Results")

    print(X_vals[46], z(X_vals[46,0], X_vals[46,1]))
    print(Y_before)
    print(Y_vals)
    print(y_vals[46])
    print(mse(Y_vals, y_vals[46].ravel()))
