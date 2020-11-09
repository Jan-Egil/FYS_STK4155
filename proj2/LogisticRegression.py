from func import * #importing everything from func.py, including external packages

from sklearn.linear_model import LogisticRegression
from sklearn import datasets


"""X = digits["data"][:, 3:] # petal width
y = (digits["target"] == 2).astype(np.int) # 1 if Iris-Virginica, else 0

log_reg = LogisticRegression()
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica") # + more Matplotlib code to make the image look pretty
plt.show()

#This is the solution to part e) of project 2

X = DesignMatrixCreator_2dpol(polydeg,x,y) #Create design matrix

z = frankefunc_noise(x,y,noise) #

X_train, X_test, zTrain, zTest = train_test_split(X,y,test_size=0.2) #Split data into training and testing set
X_train, X_test = scale(X_train, X_test) #Properly scale the data

w, b = np.randin
w ,b = np.random.rand(N), np.random.rand(N)
"""
def somefunc(y):
    if y == 1:
        c = -np.log(p_hat)
    if y == 0:
        c = -np.log(1-p_hat)
    return 0


def plot_random_numbers():
    # choose some random images to display
    indices = np.arange(n_inputs)
    random_indices = np.random.choice(indices, size=5)
    for i, image in enumerate(digits.images[random_indices]):
        plt.subplot(1, 5, i+1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title("Label: %d" % digits.target[random_indices[i]])
    plt.show()

"""Logistic Regression cost function (log loss)"""
def log_reg_cost(X,Y,weights,beta,learning_rate):
    m = X.shape[0]
    p_hat = activation_func(X,weights,beta,'softmax')
    J = - (np.sum(-Y.T * np.log(p_hat) - (1-Y).T*np.log(1-p_hat)))/m
    dJ = -(X.T @ (-p_hat + Y.T).T)/m + learning_rate * weights#learning rate here
    return J,dJ

"""Last opp håndskrevne tall fra sklearn
   Opplastingskode fra https://compphysics.github.io/MachineLearning/doc/pub/week41/html/week41.html """
# ensure the same random numbers appear every time
np.random.seed(0)
# display images in notebook
plt.rcParams['figure.figsize'] = (12,12)
# download MNIST dataset
digits = datasets.load_digits()

# define inputs and labels
inputs = digits.images
labels = digits.target

# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)
print("X = (n_inputs, n_features) = " + str(inputs.shape))

train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, test_size = 0.2)



log_reg = LogisticRegression()
print(log_reg.fit(X_train, Y_train))

"""
log reg Inspiration from [https://medium.com/@awjuliani/simple-softmax-in-python-tutorial-d6b4c4ed5c16]
"""
b = 0#np.zeros([X_train.shape[1],Y_train.shape[0]])
weights = np.zeros([X_train.shape[1],Y_train.shape[0]])
iterations = 200
learning_rate = 1e-5
losses = []
for i in range(0,iterations):
    cost,gradient = log_reg_cost(X_train,Y_train,weights,b,1)
    losses.append(cost)
    weights = weights - (learning_rate * gradient)

print(cost)

plt.plot(losses)
plt.show()





    #
