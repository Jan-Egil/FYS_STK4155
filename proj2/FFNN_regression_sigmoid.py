from func import * #importing everything from func.py, including external packages

#This is the solution to part b of project 2

N = 200 #Number of data points
polydeg = 2#int(input("Enter the degree of polynomial you want to approximate: ")) #Order of polynomial
noise = 0.2 #Factor of noise in data

xy = np.random.rand(N,2) #Create random function parameters
x = xy[:,0]; y = xy[:,1]


X = DesignMatrixCreator_2dpol(polydeg,x,y) #Create design matrix

z = frankefunc_noise(x,y,noise) #

X_train, X_test, zTrain, zTest = train_test_split(X,z,test_size=0.2) #Split data into training and testing set
X_train, X_test = scale(X_train, X_test) #Properly scale the data
