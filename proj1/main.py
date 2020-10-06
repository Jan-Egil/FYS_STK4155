
"""
=================================================================================
==================== WELCOME TO THE LINEAR REGRESSION-INATOR ====================
=================================================================================

This program is used as the solution for project 1 in the course 'FYS-STK4155'
at the University of Oslo.

This program runs several linear regression models to fit data. First to an
analytic function called the Franke function, then to actual topographical
map data provided by NASA. We will be using Ordinary Least Squares (OLS), Ridge,
& Lasso as regression methods. We will be using bootstrap resampling and
Leave-One-Out Cross Validation (LOOCV) resampling methods to optimize our results.

Note that this program runs several function files from the file func.py, so
make sure that this file is located in the same repository as this file.

The program is split into one chunk for each of the subproblems in the
project description. They are as follows:

a) Basic OLS regression. Compare the results with the ones from SKLearns
matric-package. Calculations of confidence intervals of regression parameters.

b) Complexity vs Error plot. Introduce the Bootstrap resampling method. Use
Bootstrap to perform Bias-variance trade-off analysis. Still using OLS

c) Introduce LOOCV and optimize using this resampling technique.

d) Much the same as the previous 3 exercises, but replace OLS with Ridge

e) Much the same as in exercise d), but replace Ridge with Lasso.

f) Not really an exercise for programming. Plots the chosen topographical
map that we will use for the next exercise.

g) Much of the same as subproblems a) through e), but with real data as
given by NASA SRTM maps. Not that thorough of a analysis and more on
the qualitative results.

We hope you enjoy your stay here at the Linear Regression-inator

==================================================================================
======================= READ EVERYTHING ABOVE THIS LINE!!! =======================
==================================================================================
"""
print(__doc__)

from func import * #Really nasty syntax, importing everything from func.py, including external packages


print("\n\nWhich task do you want to run?")
exercise = input("Press any letter between a & g: ")

"""
Part a)
"""

if exercise == "a":
    N = 200
    polydeg = int(input("Enter the degree of polynomial you want to approximate: ")) #Order of polynomial
    noise = 0.2 #Factor of noise in data

    xy = np.random.rand(N,2)
    x = xy[:,0]; y = xy[:,1]


    X = DesignMatrixCreator_2dpol(polydeg,x,y)
    print(X.shape)
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
    CI_beta_L,CI_beta_U = CI_normal(beta, var_beta, 0.05)
    CIbeta_df = pd.DataFrame(np.transpose(np.array([CI_beta_L, beta, CI_beta_U])),columns=['Lower CI', 'beta', 'Upper CI'])
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
    if polydeg == 5:
        ticks = ["const","$y$","$x$","$y^2$","$xy$","$x^2$","$y^3$","$xy^2$","$x^2y$","$x^3$",
        "$y^4$","$xy^3$","$x^2y^2$","$x^3y$","$x^4$","$y^5$","$xy^4$","$x^2y^3$","$x^3y^2$","$x^4y$","$x^5$"]
        plt.scatter(np.arange(0,21),beta,marker="^",label="$\\beta_j$")
        plt.errorbar(np.arange(0,21),beta,yerr=[np.abs(beta-CI_beta_L),np.abs(beta-CI_beta_U)],capsize=5,fmt="none",label="95% Confidence intervals")
        plt.xticks((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),ticks,fontsize="large")
        plt.legend(); plt.grid()
        plt.xlabel("Corresponding factor",fontsize="x-large")
        plt.ylabel("Coefficient value",fontsize="x-large")
        plt.title("Values for parameters $\\beta$ and corresponding confidence intervals",fontsize="x-large")
        print(CIbeta_df)
        plt.show()
    else:
        print(CIbeta_df)


"""
Part b)
"""

if exercise == "b": #Complexity vs error, bootstrap & bias-variance analysis

    print("\ntype 'a' for complexity vs error plot, type 'b' for bootstrap bias-variance analysis.")
    specifics = input("Type here: ")

    if specifics == "a":
        MaxPoly = 20
        N = 200
        noise = 0.2
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

            MSE_train_array[polydeg-1] = MSE(z_train,z_tilde_train)
            MSE_test_array[polydeg-1] = MSE(z_test,z_tilde_test)

        polydeg_array = np.arange(1,MaxPoly+1)
        plt.plot(polydeg_array,MSE_train_array,label="Train")
        plt.plot(polydeg_array,MSE_test_array,label="Test")
        plt.xlabel("'Complexity' of model (Polynomial degree)",fontsize="large")
        plt.ylabel("Mean Squared Error (MSE)",fontsize="large")
        plt.title("N = %i, test size = %.1f%%, noise = %.2f\nOrdinary Least Squares"% (N,testsize*100,noise),fontsize="x-large")
        plt.legend(); plt.grid(); plt.semilogy()
        plt.show()

    elif specifics == "b":
        print("\nHow many bootstrap runs do you wish to do?")
        n = int(input("Type here: "))
        MaxPoly = 20
        N = 200
        noise = 1
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
        plt.plot(polydeg_array,bias_a,"--",label="Bias")
        plt.plot(polydeg_array,var_a,"--",label="Variance")
        plt.plot(polydeg_array,MSE_a,label="MSE")
        plt.xlabel("Complexity of model (Degree of Polynomial)",fontsize="large")
        plt.ylabel("Error score (MSE/Bias/Variance)",fontsize="large")
        plt.title("Bias-Variance tradeoff. N = %i, noise = %.2f, bootstraps=%i"%(N,noise,n),fontsize="x-large")
        plt.grid(); plt.legend(); plt.semilogy()
        plt.show()

"""
Part c)
"""

if exercise == "c": #Cross-Validation

    polydeg = int(input("Enter the degree of polynomial you want to approximate: ")) #Order of polynomial

    N = 200; noise = 1; k = 5
    xy = np.random.rand(N,2)
    x = xy[:,0]; y = xy[:,1]
    z = frankefunc_noise(x,y,noise)
    X = DesignMatrixCreator_2dpol(polydeg,x,y)

    mse_crossval = func_cross_validation(polydeg, X, y, k, "OLS")
    print(mse_crossval)


"""
Part d)
"""

if exercise == "d": #Ridge regression

    print("Type 'a' for hyperparameter fitting plot, type 'b' for Complexity vs error-plot")
    decisions = input("Please type here: ")

    if decisions == "a": #Hyperparameter Fitting
        PolyDeg = 5
        N = 200
        noise = 1

        xy = np.random.rand(N,2)
        x = xy[:,0]; y = xy[:,1]
        z = frankefunc_noise(x,y,noise)
        X = DesignMatrixCreator_2dpol(PolyDeg,x,y)

        lambdavals = np.logspace(-10,5,200)

        X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)
        X_train_scaled, X_test_scaled = scale(X_train, X_test)

        z_tilde_test, z_tilde_train, BetaRidge, OptLamb, MSE_lamb, MSE_lamb_skl = Ridge(X_train,X_test,z_train,z_test,lambdavals)

        plt.plot(lambdavals,MSE_lamb,label="Test data MSE",color = 'g')
        plt.plot(lambdavals,MSE_lamb_skl,linestyle ='--',dashes=(5, 10), label = 'SKL ridge', color = 'yellow')
        plt.axvline(OptLamb,label="$\lambda = $%e"%OptLamb,color = 'r')
        plt.semilogx();
        plt.grid()
        plt.xlabel("Value for hyperparameter $\lambda$",fontsize="large")
        plt.ylabel("Mean Squared Error (MSE)",fontsize="large")
        plt.title("Ridge Hyperparameter fit for $\lambda$",fontsize="x-large")
        plt.legend()
        plt.show()

    elif decisions == "b": #Complexity vs error
        MaxPoly = 20
        N = 200
        noise = 0.2
        testsize = 0.2

        xy = np.random.rand(N,2)
        x = xy[:,0]; y = xy[:,1]
        z = frankefunc_noise(x,y,noise)
        lambdavals = np.logspace(-10,5,100)

        MSE_train_array = np.zeros(MaxPoly)
        MSE_test_array = np.zeros(MaxPoly)

        for polydeg in range(1,MaxPoly+1):
            X = DesignMatrixCreator_2dpol(polydeg,x,y)
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=testsize)
            X_train, X_test = scale(X_train, X_test)

            z_tilde_test, z_tilde_train, Beta_Ridge, optimalLambda, MSE_lamb, MSE_lamb_skl = Ridge(X_train,X_test,z_train,z_test,lambdavals)

            MSE_train_array[polydeg-1] = MSE(z_train,z_tilde_train)
            MSE_test_array[polydeg-1] = MSE(z_test,z_tilde_test)
            print(polydeg)

        polydegs = np.arange(1,MaxPoly+1)
        plt.figure()
        plt.plot(polydegs,MSE_train_array,label="Train")
        plt.plot(polydegs,MSE_test_array,label="Test")
        plt.xlabel("'Complexity' of model (Polynomial degree)",fontsize="large")
        plt.ylabel("Mean Squared Error (MSE)",fontsize="large")
        plt.title("N = %i, test size = %.1f%%, noise = %.2f\nRidge Shrinkage Method"% (N,testsize*100,noise),fontsize="x-large")
        plt.grid(); plt.legend(); plt.semilogy()
        plt.show()

    elif decisions == "c": #Bootstrap
        pass

    elif decisions == "d": #CV
        pass

"""
Part e)
"""

if exercise == "e": #Lasso regression

    print("Type 'a' for hyperparameter fitting plot, type 'b' for Complexity vs error-plot")
    decisions = input("Please type here: ")

    if decisions == 'a':
        PolyDeg = 5
        N = 200
        noise = 1

        xy = np.random.rand(N,2)
        x = xy[:,0]; y = xy[:,1]
        z = frankefunc_noise(x,y,noise)
        X = DesignMatrixCreator_2dpol(PolyDeg,x,y)[:,1:]
        print(X)

        X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)
        X_train, X_test = scale(X_train, X_test)


        lambdavals = np.logspace(-10,3,200)
        MSE_array = np.zeros(len(lambdavals))
        for i,lamb in enumerate(lambdavals):
            clf = Lasso(alpha=lamb).fit(X_train,z_train)
            beta_lasso = clf.coef_
            #ztilde_test = X_test @ beta_lasso

            ztilde_test = clf.predict(X_test)

            MSE_array[i] = MSE(z_test,ztilde_test)


            #print(clf.predict(X_test))
        plt.plot(lambdavals,MSE_array,label="Test data MSE")
        plt.axvline(lambdavals[np.argmin(MSE_array)],label="$\lambda$ = %e"%lambdavals[np.argmin(MSE_array)])
        plt.xlabel("Value for hyperparameter $\lambda$",fontsize="large")
        plt.ylabel("Mean Squared Error (MSE)",fontsize="large")
        plt.title("Lasso Hyperparameter fit for $\lambda$",fontsize="x-large")
        plt.semilogx(); plt.grid(); plt.legend()
        plt.show()

    elif decisions == 'b':
        MaxPoly = 20
        N = 200
        noise = 0.2
        testsize = 0.2

        xy = np.random.rand(N,2)
        x = xy[:,0]; y = xy[:,1]
        z = frankefunc_noise(x,y,noise)
        lambdavals = np.logspace(-10,5,30)

        MSE_train_array = np.zeros(MaxPoly)
        MSE_test_array = np.zeros(MaxPoly)

        for polydeg in range(1,MaxPoly+1):
            X = DesignMatrixCreator_2dpol(polydeg,x,y)
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=testsize)
            X_train, X_test = scale(X_train, X_test)

            MSE_array_testlamb = np.zeros(len(lambdavals))
            MSE_array_trainlamb = np.zeros(len(lambdavals))

            for i, lamb in enumerate(lambdavals):
                clf = Lasso(alpha=lamb).fit(X_train,z_train)

                ztilde_test = clf.predict(X_test)
                ztilde_train = clf.predict(X_train)

                MSE_array_testlamb[i] = MSE(z_test,ztilde_test)
                MSE_array_trainlamb[i] = MSE(z_train,ztilde_train)

            MSE_test_array[polydeg-1] = np.min(MSE_array_testlamb)
            MSE_train_array[polydeg-1] = MSE_array_trainlamb[np.argmin(MSE_array_testlamb)]

        polydegs = np.arange(1,MaxPoly+1)

        plt.plot(polydegs,MSE_train_array, label="Train")
        plt.plot(polydegs,MSE_test_array, label="Test")
        plt.xlabel("'Complexity' of model (Polynomial degree)",fontsize="large")
        plt.ylabel("Mean Squared Error (MSE)",fontsize="large")
        plt.title("N = %i, test size = %.1f%%, noise = %.2f\nLasso Shrinkage Method"% (N,testsize*100,noise),fontsize="x-large")
        plt.grid(); plt.legend(); plt.semilogy()
        plt.show()

    elif decisions == 'c': #Bootstrap
        pass

    elif decisions == 'd': #CV
        pass

"""
Part f)
"""

if exercise == "f": #Plot map data

    print("\nExercise f doesn't really exist\n")
    print("It is only a placeholder for the download of data for the next exercise\n")
    print("However: we will gladly plot the selected image file for y'all!")

    terrainvar = imread('n59_e010_1arc_v3.tif')

    plt.figure()
    plt.title("Terrain over the Oslo fjord region",fontsize="x-large")
    plt.imshow(terrainvar,cmap='gray')
    plt.xlabel("<-- West - East -->",fontsize="x-large")
    plt.ylabel("<-- South - North -->",fontsize="x-large")
    plt.xticks([]); plt.yticks([])
    plt.show()

"""
Part g)
"""

if exercise == "g": #EVERYTHING, but with map data instead
    print("\nDo you want to get an MSE vs Complexity plot, or attempt to recreate the topographical maps?")
    print("\nType 'a' for MSE-plots, type 'b' to recreate the maps")
    decisions = input("Please type here: ")

    if decisions == "a":
        terrainvar = imread('n59_e010_1arc_v3.tif')

        N = 1000
        x = np.random.randint(0,terrainvar.shape[1],size=N)
        y = np.random.randint(0,terrainvar.shape[0],size=N)

        z = terrainvar[y,x]

        MaxPoly = 45

        MSE_testOLS_array = np.zeros(MaxPoly)
        MSE_trainOLS_array = np.zeros(MaxPoly)
        MSE_testridge_array = np.zeros(MaxPoly)
        MSE_testlasso_array = np.zeros(MaxPoly)
        polydegs = np.arange(1,MaxPoly+1)
        for polydeg in range(1,MaxPoly+1):
            X = DesignMatrixCreator_2dpol(polydeg,x,y)

            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)
            X_train, X_test = scale(X_train, X_test)

            z_tilde_test, z_tilde_train, beta_optimal = OLS(X_train, X_test, z_train, z_test)

            MSE_testOLS_array[polydeg-1] = MSE(z_test,z_tilde_test)
            MSE_trainOLS_array[polydeg-1] = MSE(z_train,z_tilde_train)
            print(polydeg)

        plt.plot(polydegs,MSE_testOLS_array,label="Test OLS")
        plt.plot(polydegs,MSE_trainOLS_array,label="Train OLS")
        plt.grid(); plt.legend(); plt.semilogy()
        plt.show()

    elif decisions == "b":
        terrainvar = imread('n59_e010_1arc_v3.tif')

        N = 1000
        x = np.random.randint(0,terrainvar.shape[1],size=N)
        y = np.random.randint(0,terrainvar.shape[0],size=N)

        TrainingData = terrainvar[y,x]

        print("\nDo you want to perform OLS, Ridge, or Lasso regression analysis?")
        print("Type 'a' for OLS, 'b' for Ridge or 'c' for Lasso:\n")
        choice = input("Type here: ")

        if choice == 'a':
            #OLS
            PolyDeg = 10
            X = DesignMatrixCreator_2dpol(PolyDeg,x,y)

            X_train, X_test, z_train, z_test = train_test_split(X,TrainingData,test_size=0.2)
            X_train, X_test = scale(X_train, X_test)

            z_tilde_test, z_tilde_train, beta_optimal = OLS(X_train, X_test, z_train, z_test)

            MSE_train = MSE(z_train,z_tilde_train)
            MSE_test = MSE(z_test,z_tilde_test)



            """
            plt.figure()
            plt.title("Approximate map",fontsize="x-large")
            plt.imshow(ApproxImg,cmap='gray')
            plt.xlabel("<-- West - East -->",fontsize="large")
            plt.ylabel("<-- South - North --->",fontsize="large")
            plt.xticks([]); plt.yticks([])
            """

            plt.figure()
            plt.title("Actual map",fontsize="x-large")
            plt.imshow(terrainvar,cmap='gray')
            plt.xlabel("<-- West - East -->",fontsize="large")
            plt.ylabel("<-- South - North --->",fontsize="large")
            plt.xticks([]);plt.yticks([])
            plt.show()

        elif choice == 'b':
            #Ridge
            pass
        elif choice == 'c':
            #Lasse
            pass
        else:
            print("\nLol you can't even follow simple instructions")

"""
Easter egg! Compare the MSEs
"""

if exercise == "h":
    print("You just activated an easter egg!")
    print("We will now present you with an MSE vs complexity plot for all")

    MaxPoly = 30
    N = 100
    noise = 2
    testsize = 0.2

    xy = np.random.rand(N,2)
    x = xy[:,0]; y = xy[:,1]
    z = frankefunc_noise(x,y,noise)

    lambdavals = np.logspace(-10,5,100)

    MSE_ols_array = np.zeros(MaxPoly)
    MSE_ridge_array = np.zeros(MaxPoly)
    MSE_lasso_array = np.zeros(MaxPoly)

    for polydeg in range(1,MaxPoly+1):
        print(polydeg)

        X = DesignMatrixCreator_2dpol(polydeg,x,y)

        X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=testsize)
        X_train, X_test = scale(X_train, X_test)

        """
        OLS
        """

        z_tilde_test_ols = OLS(X_train, X_test, z_train, z_test)[0]
        MSE_ols_array[polydeg-1] = MSE(z_test,z_tilde_test_ols)

        """
        Ridge
        """

        z_tilde_test_ridge = Ridge(X_train, X_test, z_train, z_test, lambdavals)[0]
        MSE_ridge_array[polydeg-1] = MSE(z_test,z_tilde_test_ridge)

        """
        Lasso
        """

        MSE_lasso_lamb_array = np.zeros(lambdavals.shape[0])

        for i, lamb in enumerate(lambdavals):
            clf = Lasso(alpha=lamb).fit(X_train, z_train)
            z_tilde_test = clf.predict(X_test)

            MSE_lasso_lamb_array[i] = MSE(z_test, z_tilde_test)

        MSE_lasso_array[polydeg-1] = np.min(MSE_lasso_lamb_array)

    polydegs = np.arange(1,MaxPoly+1)

    plt.plot(polydegs, MSE_ols_array, label="OLS")
    plt.plot(polydegs, MSE_ridge_array, label="Ridge")
    plt.plot(polydegs, MSE_lasso_array, label="Lasso")
    plt.xlabel("'Complexity' of model (Polynomial degree)",fontsize="large")
    plt.ylabel("Mean Squared Error (MSE)",fontsize="large")
    plt.title("N = %i, test size = %.1f%%, noise = %.2f"% (N,testsize*100,noise),fontsize="x-large")
    plt.legend(); plt.grid(); plt.semilogy()
    plt.show()
