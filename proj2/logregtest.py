from func import * #importing everything from func.py, including external packages

from sklearn.linear_model import LogisticRegression
from sklearn import datasets

digits = datasets.load_digits()

images = digits.data; targets = digits.target
images_train, images_test, targets_train, targets_test = train_test_split(images,targets,test_size=0.2)

epochs = 100; M = 2; n = images_train.shape[0]
theta = SGD(images_train,targets_train,n,M,epochs,costfunc='Logistic',gamma=0.001,classes=10)

pred_test = images_test @ theta

sum = 0
for i in range(targets_test.shape[0]):
    sum += targets_test[i]==np.argmax(pred_test[i])
sum = sum/targets_test.shape[0]
print("%.2f%%" % (sum*100))


"""
plt.imshow(images_train[0].reshape(8,8),cmap=plt.cm.gray_r,interpolation='nearest')
plt.show()
"""
