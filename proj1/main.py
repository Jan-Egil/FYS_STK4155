"""
=================Docstring=================
Just an illustration that docstrings actually work out pretty nicely

This is just printed to terminal when the program is run.

What we will choose to write here when the project is finished is beyond me at the moment.
"""

from func import * #Really nasty syntax

#print(__doc__)

"""
Defining different functions to be used
"""

print("Which task do you want to run?")
exercise = input("Press any letter between a & g: ")

"""
Part a)
"""

if exercise == "a":
    N = 1000
    polydeg = int(input("Enter the degree of polynomial you want to approximate: ")) #Order of polynomial
    noise = 1 #Factor of noise in data

    xy = np.random.rand(N,2)
    x = xy[:,0]; y = xy[:,1]


    X = DesignMatrixCreator_2dpol(polydeg,x,y)
    z = frankefunc_noise(x,y,noise)

    X_train, X_test, zTrain, zTest = train_test_split(X,z,test_size=0.2)
    X_train, X_test = scale(X_train, X_test)

    z_tilde_test, z_tilde_train, beta = OLS(X_train, X_test, zTrain, zTest)

    MSE_train_scikit = metric.mean_squared_error(zTrain, z_tilde_train)
    R2_train_scikit = metric.r2_score(zTrain,z_tilde_train)

    MSE_test_scikit = metric.mean_squared_error(zTest,z_tilde_test)
    R2_test_scikit = metric.r2_score(zTest,z_tilde_test)

    MSE_train = MSE(zTrain,z_tilde_train)
    R2_train = R2(zTrain,z_tilde_train)

    MSE_test = MSE(zTest,z_tilde_test)
    R2_test = R2(zTest,z_tilde_test)

    """Confidence interval"""
    var_Z = variance_estimator(polydeg,zTrain,z_tilde_train)
    var_beta = np.diag(np.linalg.pinv(X_train.T @ X_train))*var_Z
    mean_beta = np.mean(beta)

    CI_beta_L,CI_beta_U = CI_normal(beta, var_beta, 0.05)

    print("\n-------------------------R2-Score-----------------------------------\n")
    print("The R2 score for the training data is %e using SKLearn" % R2_train_scikit)
    print("The R2 score for the training data is %e using own defined function" % R2_train)
    print(" ")
    print("The R2 score for the test data is %e using SKLearn" % R2_test_scikit)
    print("The R2 score for the test data is %e using own defined function" % R2_test)
    print("\n-------------------------MSE-Score----------------------------------\n")
    print("The MSE score for the training data is %e using SKLearn" % MSE_train_scikit)
    print("The MSE score for the training data is %e using own defined function" % MSE_train)
    print(" ")
    print("The MSE score for the test data is %e using SKLearn" % MSE_test_scikit)
    print("The MSE score for the test data is %e using own defined function" % MSE_test)
    print("\n-------------------------CI-score----------------------------------\n")
    print(CI_beta_U)
    print(beta)
    print(CI_beta_L)

"""
Part b)
"""

if exercise == "b":
    print("\ntype 'a' for complexity vs error plot, type 'b' for bootstrap bias-variance analysis.")
    specifics = input("Type here: ")
    if specifics == "a":
        MaxPoly = 20
        N = 200
        noise = 0.3
        testsize = 0.2

        xy = np.random.rand(N,2)
        x = xy[:,0]; y = xy[:,1]
        z = frankefunc_noise(x,y,noise)

        MSE_train_array = np.zeros(MaxPoly)
        MSE_test_array = np.zeros(MaxPoly)

        for polydeg in range(1,MaxPoly+1):
            X = DesignMatrixCreator_2dpol(polydeg,x,y)
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)
            X_train, X_test = scale(X_train, X_test)

            z_tilde_test, z_tilde_train, beta_optimal = OLS(X_train, X_test, z_train, z_test)

            MSE_train = MSE(z_train,z_tilde_train)
            MSE_test = MSE(z_test,z_tilde_test)

            MSE_train_array[polydeg-1] = (MSE_train)
            MSE_test_array[polydeg-1] = (MSE_test)

        polydeg_array = np.arange(1,MaxPoly+1)
        plt.plot(polydeg_array,MSE_train_array,label="Train")
        plt.plot(polydeg_array,MSE_test_array,label="Test")
        plt.xlabel("'Complexity' of model (Polynomial degree)",fontsize="large")
        plt.ylabel("Mean Squared Error (MSE)",fontsize="large")
        plt.title("N = %i, test size = %.1f%%, noise = %.2f"% (N,testsize*100,noise),fontsize="x-large")
        plt.legend(); plt.grid(); plt.semilogy()
        plt.show()

    elif specifics == "b":
        print("\nHow many bootstrap runs do you wish to do?")
        n = int(input("Type here: "))
        MaxPoly = 20
        N = 200
        noise = 0.3
        testsize = 0.2

        xy = np.random.rand(N,2)
        x = xy[:,0]; y = xy[:,1]
        z = frankefunc_noise(x,y,noise)

        MSE_a = np.zeros(MaxPoly)

        bias_a = np.zeros(MaxPoly)
        var_a = np.zeros(MaxPoly)

        i = 0
        for polydeg in range(1,MaxPoly+1):
            X = DesignMatrixCreator_2dpol(polydeg,x,y)

            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=testsize)
            X_train, X_test = scale(X_train, X_test)

            MSE_a[polydeg-1], bias_a[polydeg-1], var_a[polydeg-1] = Func_Bootstrap(X_train, X_test, z_train, z_test, n, "OLS")

        polydeg_array = np.arange(1,MaxPoly+1)
        plt.plot(polydeg_array,bias_a,label="Bias")
        plt.plot(polydeg_array,var_a,label="Variance")
        plt.plot(polydeg_array,MSE_a,label="MSE")
        plt.grid(); plt.legend(); plt.semilogy()
        plt.show()

"""
Part c)
"""

if exercise == "c":
    print("You've come a long way to start with this exercise, haven't you..?")

    polydeg = int(input("Enter the degree of polynomial you want to approximate: ")) #Order of polynomial

    N = 200; noise = 1; k = 5
    xy = np.random.rand(N,2)
    x = xy[:,0]; y = xy[:,1]
    z = frankefunc_noise(x,y,noise)
    X = DesignMatrixCreator_2dpol(polydeg,x,y)

    mse_crossval = func_cross_validation(polydeg, X, y, k, "OLS")
    print(mse_crossval)

    """print(X.shape)
    print(y.shape)
    print(y)
    i = 0
    X_split = np.array(np.array_split(X,k))
    Y_split = np.array(np.array_split(y,k))
    print(X_split.shape)
    print(Y_split.shape)
    print(Y_split)
    X_test = X_split[i]
    Y_test = Y_split[i].ravel()
    print(Y_test.shape)
    print(Y_test)
    X_train = np.concatenate((X_split[:i], X_split[(i+1):]))
    Y_train = np.concatenate((Y_split[:i], Y_split[(i+1):])).ravel()
    print(X_train.shape)
    print(Y_train)
    X_train = X_train.reshape(-1,6)
    print(X_train.shape)
    #print(X_train)

    #X_train = X_split[1:5].transpose(2,0,1).reshape(80,-1)
    print(X_train.shape)"""


"""
Part d)
"""

if exercise == "d":
    PolyDeg = 5
    N = 50
    noise = 1

    xy = np.random.rand(N,2)
    x = xy[:,0]; y = xy[:,1]
    z = frankefunc_noise(x,y,noise)
    X = DesignMatrixCreator_2dpol(PolyDeg,x,y)

    lambdavals = np.logspace(-3,5,200)

    X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)
    X_train_scaled, X_test_scaled = scale(X_train, X_test)

    z_tilde_test, z_tilde_train, BetaRidge, OptLamb, MSE_lamb = Ridge(X_train,X_test,z_train,z_test,lambdavals)

    plt.plot(lambdavals,MSE_lamb)
    plt.axvline(OptLamb)
    plt.semilogx()
    plt.grid()
    plt.xlabel("Value for hyperparameter $\lambda$",fontsize="x-large")
    plt.ylabel("Mean Squared Error (MSE)",fontsize="x-large")
    plt.show()

"""
Part e)
"""

if exercise == "e":
    print("Lasso time")
    PolyDeg = 10
    N = 500
    noise = 1

    xy = np.random.rand(N,2)
    x = xy[:,0]; y = xy[:,1]
    z = frankefunc_noise(x,y,noise)
    X = DesignMatrixCreator_2dpol(PolyDeg,x,y)

    X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)


    lambdavals = np.logspace(-3,5,200)
    MSE_array = np.zeros(len(lambdavals))
    for i,lamb in enumerate(lambdavals):
        clf = Lasso(alpha=lamb).fit(X_train,z_train)
        beta_lasso = clf.coef_

        ztilde_test = X_test @ beta_lasso
        MSE_array[i] = MSE(z_test,ztilde_test)


        #print(clf.predict(X_test))
    plt.plot(lambdavals,MSE_array)
    plt.semilogx()
    plt.show()
"""
Part f)
"""

if exercise == "f":
    print("\nExercise f doesn't really exist\n")
    print("It is only a placeholder for the download of data for the next exercise\n")
    print("However: we will gladly plot the selected image file for y'all!")

    terrainvar = imread('n59_e010_1arc_v3.tif')

    plt.figure()
    plt.title("Terrain over the Oslo fjord region",fontsize="x-large")
    plt.imshow(terrainvar,cmap='gray')
    plt.xlabel("<-- West - East -->",fontsize="large")
    plt.ylabel("<-- South - North -->",fontsize="large")
    plt.show()

"""
Part g)
"""

if exercise == "g":
    print("\nWelcome to the bottom.\n")

    terrainvar = imread('n59_e010_1arc_v3.tif')

    N = 1000
    x = np.random.randint(0,terrainvar.shape[1],size=N)
    y = np.random.randint(0,terrainvar.shape[0],size=N)

    allx = np.copy(terrainvar[1])
    ally = np.copy(terrainvar[0])

    TrainingData = terrainvar[y,x]

    print("\nDo you want to perform OLS, Ridge, or Lasso regression analysis?")
    print("Type 'a' for OLS, 'b' for Ridge or 'c' for Lasso:\n")
    choice = input("Type here: ")

    if choice == 'a':
        #OLS
        PolyDeg = 10
        X = DesignMatrixCreator_2dpol(PolyDeg,x,y)
        Ximg = DesignMatrixCreator_2dpol(PolyDeg,allx,ally)
        Yimg = DesignMatrixCreator_2dpol(PolyDeg,ally,allx)

        X_train, X_test, z_train, z_test = train_test_split(X,TrainingData,test_size=0.2)
        X_train, X_test = scale(X_train, X_test)
        X_train, Ximg = scale(X_train,Ximg)
        X_train, Yimg = scale(X_train,Yimg)

        z_tilde_test, z_tilde_train, beta_optimal = OLS(X_train, X_test, z_train, z_test)
        MSE_train = MSE(z_train,z_tilde_train)
        MSE_test = MSE(z_test,z_tilde_test)

        ApproxImgX = Ximg @ beta_optimal
        ApproxImgY = Yimg @ beta_optimal

        ApproxImg = np.meshgrid(ApproxImgX,ApproxImgY)
        print(np.array(ApproxImg).shape)

        plt.figure()
        plt.title("Approximate map",fontsize="x-large")
        plt.imshow(ApproxImg[0],cmap='gray')
        plt.xlabel("<-- West - East -->",fontsize="large")
        plt.ylabel("<-- South - North --->",fontsize="large")

        plt.figure()
        plt.title("Actual map",fontsize="x-large")
        plt.imshow(terrainvar,cmap='gray')
        plt.xlabel("<-- West - East -->",fontsize="large")
        plt.ylabel("<-- South - North --->",fontsize="large")
        plt.show()

    elif choice == 'b':
        #Ridge
        pass
    elif choice == 'c':
        #Lasse
        pass
    else:
        print("\nLol you can't even follow simple instructions")
