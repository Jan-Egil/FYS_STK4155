from func import * #importing everything from func.py, including external packages
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

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
    J = -np.sum(Y*np.log(p_hat))#- (np.sum(-Y.T * np.log(p_hat) - (1-Y).T*np.log(1-p_hat)))/m

    #da = (-Y/p_hat)
    #mat = p_hat@(np.ones((1,9))) * (np.identity(9)-np.ones((9,1))@p_hat.T)
    dJ = -(X.T @ (-p_hat.T + Y.T).T)/m + learning_rate * weights#learning rate here

    return J,dJ

def onehotvec(Y):
    classes = np.arange(np.max(Y))
    hotvec = np.zeros((len(Y),len(classes)))
    for i in range(len(Y)):
        hotvec[i,Y[i]-1] = Y[i]
    return hotvec

"""Last opp håndskrevne tall fra sklearn"""
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


labels = onehotvec(labels)
train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, test_size = 0.2)


"""
log reg Inspiration from [https://medium.com/@awjuliani/simple-softmax-in-python-tutorial-d6b4c4ed5c16]
"""
b = np.random.uniform(-0.1,0.1,Y_train.shape[1])#0#np.zeros([X_train.shape[1],Y_train.shape[0]])
weights = np.random.uniform(-0.1,0.1,[X_train.shape[1],Y_train.shape[1]])

iterations = 1000
learning_rate = 1e-5
losses = []
for i in range(0,iterations):
    cost,gradient = log_reg_cost(X_train,Y_train,weights,b,1)
    losses.append(cost)
    weights = weights - (learning_rate * gradient)

print(cost)

plt.plot(losses)
plt.show()


"""Log reg fra sklearn"""
log_reg = LogisticRegression(fit_intercept=True,
                        multi_class='auto',
                        penalty='l2', #ridge regression
                        solver='saga',
                        max_iter=10000,
                        C=50)

#log_reg.fit(X_train,Y_train)
#coef = log_reg.coef_.copy()
#plt.imshow(coef[0].reshape(8,8).round(2))
#plt.show()
#sucsess_score = log_reg.score(X_test, y_test)
#print(sucsess_score)
#print(log_reg.predict(X_test[0:20]))
#print(Y_test[0:20])



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
