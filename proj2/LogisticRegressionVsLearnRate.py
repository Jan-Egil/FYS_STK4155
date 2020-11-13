from func import * #importing everything from func.py, including external packages
from sklearn import datasets




"""
Logistic Regression cost function (log loss)
"""
def log_reg_cost(X,Y,weights,beta,learning_rate):
    m = X.shape[0]
    p_hat = activation_func(X,weights,beta,'softmax')
    J = -np.sum(Y*np.log(p_hat))
    dJ = -(X.T @ (-p_hat.T + Y.T).T)/m + learning_rate * weights
    return J,dJ

def onehotvec(Y):
    classes = np.arange(np.max(Y)+1)
    hotvec = np.zeros((len(Y),len(classes)))
    for i in range(len(Y)):
        hotvec[i,Y[i]] = 1
    return hotvec

"""
Last opp håndskrevne tall fra sklearn
"""
# ensure the same random numbers appear every time
np.random.seed(123)
# download MNIST dataset
digits = datasets.load_digits()

# define inputs and labels
inputs = digits.images
labels = digits.target

# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)


labels = onehotvec(labels)
train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, test_size = 0.2)


"""
logistic regression start (Code inspired by [https://medium.com/@awjuliani/simple-softmax-in-python-tutorial-d6b4c4ed5c16])
"""
b = np.random.uniform(-0.1,0.1,Y_train.shape[1])
weights = np.random.uniform(-0.1,0.1,[X_train.shape[1],Y_train.shape[1]])

iterations = 1000
learning_rates = np.logspace(-7,0,50)
accarray = np.zeros(learning_rates.shape[0])
for k,learning_rate in enumerate(learning_rates):
    losses = []
    for i in range(0,iterations):
        cost,gradient = log_reg_cost(X_train,Y_train,weights,b,1)
        losses.append(cost)
        weights = weights - (learning_rate * gradient)

    probability_matrix = X_test@weights
    confusion_matrix = np.zeros([10,10])

    acc = 0

    for j in range(X_test.shape[0]):
        prediction = np.argmax(probability_matrix[j])
        targetval = np.argmax(Y_test[j])
        acc += prediction == targetval
        confusion_matrix[prediction,targetval] += 1


    print(cost)
    print(Y_test.shape[0])
    acc = acc/Y_test.shape[0]
    accarray[k] = acc

plt.plot(learning_rates,accarray*100)
plt.grid(); plt.legend(); plt.semilogx()
plt.xlabel("Learning rate $\gamma$",fontsize='large')
plt.ylabel("Accuracy (%)",fontsize='large')
plt.title("Accuracy as function of learning rate",fontsize='x-large')
plt.show()
