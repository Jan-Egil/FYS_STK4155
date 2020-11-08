from func import * #importing everything from func.py, including external packages

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())

X = iris["data"][:, 3:] # petal width
y = (iris["target"] == 2).astype(np.int) # 1 if Iris-Virginica, else 0

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

def somefunc(y):
    if y == 1:
        c = -np.log(p_hat)
    if y = 0:
        c = -np.log(1-p_hat)

"""Logistic Regression cost function (log loss)"""

def log_reg_cost(X,Y,w,b,n):
    p_hat = activation_func(X,w,b)
    C = -np.sum(Y.T*np.log(p_hat) + (1-Y.T)*np.log(1-p_hat))/n
    dCdw = (X.T @ (p_hat - Y.T).T)
    dCdb = np.sum(p_hat - Y.T)/n
    return C,dCdw,dCdb












    #
