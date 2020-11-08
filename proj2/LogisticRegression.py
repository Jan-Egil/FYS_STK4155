from func import * #importing everything from func.py, including external packages

from sklearn.linear_model import LogisticRegression
from sklearn import datasets

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

print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
print("labels = (n_inputs) = " + str(labels.shape))


# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)
print("X = (n_inputs, n_features) = " + str(inputs.shape))

train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, test_size = 0.2)


log_reg = LogisticRegression()
log_reg.fit(X, y)

# choose some random images to display
indices = np.arange(n_inputs)
random_indices = np.random.choice(indices, size=5)

for i, image in enumerate(digits.images[random_indices]):
    plt.subplot(1, 5, i+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Label: %d" % digits.target[random_indices[i]])
plt.show()







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

"""Logistic Regression cost function (log loss)"""
def log_reg_cost(X,Y,w,b,n):
    p_hat = activation_func(X,w,b)
    J = -np.sum(Y.T*np.log(p_hat) + (1-Y.T)*np.log(1-p_hat))/n
    dCdw = (X.T @ (p_hat - Y.T).T)
    dCdb = np.sum(p_hat - Y.T)/n
    return C,dCdw,dCdb












    #
