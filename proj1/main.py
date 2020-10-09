
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
    N = 200 #Number of data points
    polydeg = int(input("Enter the degree of polynomial you want to approximate: ")) #Order of polynomial
    noise = 0.2 #Factor of noise in data

    xy = np.random.rand(N,2) #Create random function parameters
    x = xy[:,0]; y = xy[:,1]


    X = DesignMatrixCreator_2dpol(polydeg,x,y) #Create design matrix

    z = frankefunc_noise(x,y,noise) #

    X_train, X_test, zTrain, zTest = train_test_split(X,z,test_size=0.2) #Split data into training and testing set
    X_train, X_test = scale(X_train, X_test) #Properly scale the data

    z_tilde_test, z_tilde_train, beta = OLS(X_train, X_test, zTrain, zTest)

    MSE_train_scikit = metric.mean_squared_error(zTrain, z_tilde_train) #Calculate metric scores using SKLearn
    R2_train_scikit = metric.r2_score(zTrain,z_tilde_train)

    MSE_test_scikit = metric.mean_squared_error(zTest,z_tilde_test)
    R2_test_scikit = metric.r2_score(zTest,z_tilde_test)

    MSE_train = MSE(zTrain,z_tilde_train) #Calculate metric scores using self-made functions
    R2_train = R2(zTrain,z_tilde_train)

    MSE_test = MSE(zTest,z_tilde_test)
    R2_test = R2(zTest,z_tilde_test)

    """
    Confidence interval calculation
    """
    var_Z = variance_estimator(polydeg,zTrain,z_tilde_train)
    var_beta = np.diag(np.linalg.pinv(X_train.T @ X_train))*var_Z
    CI_beta_L,CI_beta_U = CI_normal(beta, var_beta, 0.05)
    CIbeta_df = pd.DataFrame(np.transpose(np.array([CI_beta_L, beta, CI_beta_U])),columns=['Lower CI', 'beta', 'Upper CI'])

    """
    Create organized output
    """
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


    if polydeg == 5: #Visualize in errorbar if juuuust right polynomial degree
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

    else: #Otherwise just print out the tabulated dataframe
        print(CIbeta_df)


"""
Part b)
"""

if exercise == "b": #Complexity vs error, bootstrap & bias-variance analysis

    print("\ntype 'a' for complexity vs error plot, type 'b' for bootstrap bias-variance analysis.")
    specifics = input("Type here: ")

    if specifics == "a": #Complexity vs Error
        MaxPoly = 20 #Run for all polynomial degrees up to this
        N = 200 #Number of data points
        noise = 1 #Factor of noise in data
        testsize = 0.2 #test size

        xy = np.random.rand(N,2) #Create both the (random) function parameters and values
        x = xy[:,0]; y = xy[:,1]
        z = frankefunc_noise(x,y,noise)

        MSE_train_array = np.zeros(MaxPoly) #Initialize empty MSE arrays to be filled
        MSE_test_array = np.zeros(MaxPoly)

        for polydeg in range(1,MaxPoly+1): #Does OLS for all polynomial degrees
            X = DesignMatrixCreator_2dpol(polydeg,x,y)
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=testsize) #Create / Split / Scale design matrices
            X_train, X_test = scale(X_train, X_test)

            z_tilde_test, z_tilde_train, beta_optimal = OLS(X_train, X_test, z_train, z_test) #Calculates OLS values

            MSE_train_array[polydeg-1] = MSE(z_train,z_tilde_train) #Fills the arrays to be plotted
            MSE_test_array[polydeg-1] = MSE(z_test,z_tilde_test)

        polydeg_array = np.arange(1,MaxPoly+1) #Plot the MSE results against each other
        plt.plot(polydeg_array,MSE_train_array,label="Train")
        plt.plot(polydeg_array,MSE_test_array,label="Test")
        plt.xlabel("'Complexity' of model (Polynomial degree)",fontsize="large")
        plt.ylabel("Mean Squared Error (MSE)",fontsize="large")
        plt.title("N = %i, test size = %.1f%%, noise = %.2f\nOrdinary Least Squares"% (N,testsize*100,noise),fontsize="x-large")
        plt.legend(); plt.grid(); plt.semilogy()
        plt.show()

    elif specifics == "b": #Bootstrap Bias-Variance Analysis
        print("\nHow many bootstrap runs do you wish to do?")
        n = int(input("Type here: ")) #Number of bootstraps
        MaxPoly = 20 #Run for all polynomial degrees up to this
        N = 200 #Number of data points
        noise = 1 #Factor of noise in data
        testsize = 0.2 #Test size

        xy = np.random.rand(N,2) #Create both the function parameters and values
        x = xy[:,0]; y = xy[:,1]
        z = frankefunc_noise(x,y,noise)

        MSE_a = np.zeros(MaxPoly) #Initialize MSE, bias and Variance arrays
        bias_a = np.zeros(MaxPoly)
        var_a = np.zeros(MaxPoly)

        for polydeg in range(1,MaxPoly+1): #Does bootstrap for all polynomial degrees
            X = DesignMatrixCreator_2dpol(polydeg,x,y)
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=testsize) #Create / Split / Scale design matrices
            X_train, X_test = scale(X_train, X_test)

            MSE_a[polydeg-1], bias_a[polydeg-1], var_a[polydeg-1] = Func_Bootstrap(X_train, X_test, z_train, z_test, n, "OLS") #Bootstrap and fill arrays

        polydeg_array = np.arange(1,MaxPoly+1) #Plot Bias, Variance and MSE together
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

    print("Press 'a' to check the MSE for cross validation for a given polynomial, varied k (folds)")
    print("Press 'b' to compare the MSE for CV with Bootstrap. Constant k (folds).")
    decisions = input("Please type here: ")

    if decisions == 'a': #Vary K, constant polynomial degree.
        polydeg = int(input("Enter the degree of polynomial you want to approximate: ")) #Order of polynomial

        N = 200; noise = 1; k = np.array([2,4,5,8,10]) #Set number of data points, noise and valid K-folds

        xy = np.random.rand(N,2)
        x = xy[:,0]; y = xy[:,1] #Create both function parameters and corresponding function values
        z = frankefunc_noise(x,y,noise)

        X = DesignMatrixCreator_2dpol(polydeg,x,y) #Create design matrix

        mse_crossval_array = np.zeros(k.shape[0])

        for i, k_val in enumerate(k): #Run through all the values for K
            mse_crossval_array[i] = func_cross_validation(polydeg, X, z, k_val, "OLS")

        crossval_data = {'# of Folds (K)': k, 'Mean Squared Error': mse_crossval_array}
        crossval_df = pd.DataFrame(data=crossval_data) #Organize data into data frame

        print(crossval_df) #Print said data frame

    elif decisions == 'b': #CV and Bootstrap comparison
        K = int(input("Enter the number of desired folds: ")) #Decide number of K-folds and n bootstraps
        n = int(input("Enter the number of desired bootstraps: "))
        N = 200; noise = 1; MaxPoly = 20; testsize = 0.2 #Set number of data points, noise, Max Poly degree and test size

        xy = np.random.rand(N,2)
        x = xy[:,0]; y = xy[:,1] #Create both function parameters and function values
        z = frankefunc_noise(x,y,noise)

        mse_crossval_array = np.zeros(MaxPoly) #Create empty arrays to be filled with interesting data
        mse_bootstrap_array = np.zeros(MaxPoly)

        for polydeg in range(1,MaxPoly+1): #Run through all degrees of polynomials
            X = DesignMatrixCreator_2dpol(polydeg,x,y)
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=testsize) #Create / split / scale design matrices
            X_train, X_test = scale(X_train, X_test)

            mse_crossval_array[polydeg-1] = func_cross_validation(polydeg, X, z, K, "OLS") #First cross-validation..
            mse_bootstrap_array[polydeg-1] = Func_Bootstrap(X_train, X_test, z_train, z_test, n, "OLS")[0] #..then bootstrap. Insert both values into arrays

        polydegs = np.arange(1,MaxPoly+1) #Plot the bootstrap and crossval results together.
        plt.plot(polydegs,mse_crossval_array,label="Cross Validation")
        plt.plot(polydegs,mse_bootstrap_array,label="Bootstrap")
        plt.xlabel("Complexity of model (Degree of Polynomial)",fontsize="large")
        plt.ylabel("Error score (MSE)",fontsize="large")
        plt.title("Cross-Validation vs. Bootstrap.\n N = %i, noise = %.2f, bootstraps=%i"%(N,noise,n),fontsize="x-large")
        plt.grid(); plt.legend(); plt.semilogy()
        plt.show()

"""
Part d)
"""

if exercise == "d": #Ridge regression

    print("Type 'a' for hyperparameter fitting plot, type 'b' for Complexity vs error-plot")
    print("Type 'c' for Bias-Variance analysis using Bootstrap. Type 'd' to compare bootstrap to Cross-Validation")
    decisions = input("Please type here: ")

    if decisions == "a": #Hyperparameter Fitting
        PolyDeg = 5 #Fix the polynomial degree
        N = 200 #Number of data points
        noise = 1 #Noise factor in data

        xy = np.random.rand(N,2)
        x = xy[:,0]; y = xy[:,1] #Create (random) function parameters and corresponding values
        z = frankefunc_noise(x,y,noise)
        X = DesignMatrixCreator_2dpol(PolyDeg,x,y) #Use parameters to create design matrix

        lambdavals = np.logspace(-10,5,200) #Create an array of lambda values.

        X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2) #Split and scale design matrix
        X_train_scaled, X_test_scaled = scale(X_train, X_test)

        z_tilde_test, z_tilde_train, BetaRidge, OptLamb, MSE_lamb, MSE_lamb_skl = Ridge(X_train,X_test,z_train,z_test,lambdavals) #Perform Ridge

        #Plot MSE vs hyperparameter, both for SKLearn and our method
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
        MaxPoly = 20 #Maximum polynomial degree
        N = 200 #Number of data points
        noise = 0.2 #Factor of noise
        testsize = 0.2 #Test size when splitting data

        xy = np.random.rand(N,2)
        x = xy[:,0]; y = xy[:,1] #Initialize random function parameters and corresponding function values
        z = frankefunc_noise(x,y,noise)

        lambdavals = np.logspace(-10,5,100) #Create an array of lambda values

        MSE_train_array = np.zeros(MaxPoly) #Create empty arrays to be filled with interesting data.
        MSE_test_array = np.zeros(MaxPoly)

        for polydeg in range(1,MaxPoly+1): #Run through all polynomial degrees
            X = DesignMatrixCreator_2dpol(polydeg,x,y)
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=testsize) #Create / Split / Scale design matrices
            X_train, X_test = scale(X_train, X_test)

            z_tilde_test, z_tilde_train, Beta_Ridge, optimalLambda, MSE_lamb, MSE_lamb_skl = Ridge(X_train,X_test,z_train,z_test,lambdavals) #Perform the Ridge method

            MSE_train_array[polydeg-1] = MSE(z_train,z_tilde_train) #Fill the arrays with interesting data
            MSE_test_array[polydeg-1] = MSE(z_test,z_tilde_test)
            print(polydeg) #Leave this in to get a feel for how long the program has come (it takes a while to run)

        polydegs = np.arange(1,MaxPoly+1) #Plot the MSE for test and training data together with polynomial degree
        plt.figure()
        plt.plot(polydegs,MSE_train_array,label="Train")
        plt.plot(polydegs,MSE_test_array,label="Test")
        plt.xlabel("'Complexity' of model (Polynomial degree)",fontsize="large")
        plt.ylabel("Mean Squared Error (MSE)",fontsize="large")
        plt.title("N = %i, test size = %.1f%%, noise = %.2f\nRidge Shrinkage Method"% (N,testsize*100,noise),fontsize="x-large")
        plt.grid(); plt.legend(); plt.semilogy()
        plt.show()

    elif decisions == "c": #Bootstrap & Bias-Variance for Ridge
        print("\nHow many bootstrap runs do you wish to do?")
        n = int(input("Type here: ")) #Number of bootstraps
        MaxPoly = 20 #Max polynomial degree
        N = 200 #Number of data points
        noise = 1 #Noise factor
        testsize = 0.2 #Test size when splitting the data

        xy = np.random.rand(N,2)
        x = xy[:,0]; y = xy[:,1] #Create (random) functional parameters and corresponding function values
        z = frankefunc_noise(x,y,noise)

        MSE_a = np.zeros(MaxPoly)
        bias_a = np.zeros(MaxPoly) #Create empty arrays to be filled with MSE, bias and variance
        var_a = np.zeros(MaxPoly)

        for polydeg in range(1,MaxPoly+1):
            X = DesignMatrixCreator_2dpol(polydeg,x,y)
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=testsize) #Create / split / scale design matrices
            X_train, X_test = scale(X_train, X_test)

            MSE_a[polydeg-1], bias_a[polydeg-1], var_a[polydeg-1] = Func_Bootstrap(X_train, X_test, z_train, z_test, n, "Ridge") #Bootstrap + fill arrays

        polydeg_array = np.arange(1,MaxPoly+1) #Plot arrays with MSE, bias and variance together.
        plt.plot(polydeg_array,bias_a,"--",label="Bias")
        plt.plot(polydeg_array,var_a,"--",label="Variance")
        plt.plot(polydeg_array,MSE_a,label="MSE")
        plt.xlabel("Complexity of model (Degree of Polynomial)",fontsize="large")
        plt.ylabel("Error score (MSE/Bias/Variance)",fontsize="large")
        plt.title("Bias-Variance tradeoff. N = %i, noise = %.2f, bootstraps=%i"%(N,noise,n),fontsize="x-large")
        plt.grid(); plt.legend(); plt.semilogy()
        plt.show()

    elif decisions == "d": #CV
        K = int(input("Enter the number of desired folds: "))
        n = int(input("Enter the number of desired bootstraps: ")) #Self explanatory
        N = 200; noise = 1; MaxPoly = 20; testsize = 0.2 #Number of data points, noise factor, max polynomial degree, test size when splitting data

        xy = np.random.rand(N,2)
        x = xy[:,0]; y = xy[:,1] #Create (random) functional parameteres and corresponding function values
        z = frankefunc_noise(x,y,noise)

        mse_crossval_array = np.zeros(MaxPoly) #Create empty arrays to be filled with MSE values
        mse_bootstrap_array = np.zeros(MaxPoly)

        for polydeg in range(1,MaxPoly+1): #Run through all polynomial degrees
            X = DesignMatrixCreator_2dpol(polydeg,x,y)
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=testsize) #Create / split / scale the design matrices
            X_train, X_test = scale(X_train, X_test)

            mse_crossval_array[polydeg-1] = func_cross_validation(polydeg, X, z, K, "Ridge") #Cross validation / Bootstrap. Fill arrays
            mse_bootstrap_array[polydeg-1] = Func_Bootstrap(X_train, X_test, z_train, z_test, n, "Ridge")[0]

        polydegs = np.arange(1,MaxPoly+1) #Plot arrays together in the same figure.
        plt.plot(polydegs,mse_crossval_array,label="Cross Validation")
        plt.plot(polydegs,mse_bootstrap_array,label="Bootstrap")
        plt.xlabel("Complexity of model (Degree of Polynomial)",fontsize="large")
        plt.ylabel("Error score (MSE)",fontsize="large")
        plt.title("Cross-Validation vs. Bootstrap.\n N = %i, noise = %.2f, bootstraps=%i"%(N,noise,n),fontsize="x-large")
        plt.grid(); plt.legend(); plt.semilogy()
        plt.show()

"""
Part e)
"""

if exercise == "e": #Lasso regression

    print("Type 'a' for hyperparameter fitting plot, type 'b' for Complexity vs error-plot")
    print("Type 'c' for Bias-Variance analysis using Bootstrap. Type 'd' to compare bootstrap to Cross-Validation")
    decisions = input("Please type here: ")

    if decisions == 'a': #Hyperparameter fit plot
        PolyDeg = 5 #Set polynomial degree
        N = 200 #Number of data points
        noise = 1 #Noise factor

        xy = np.random.rand(N,2)
        x = xy[:,0]; y = xy[:,1]  #Create random function parameters and corresponding function values
        z = frankefunc_noise(x,y,noise)

        X = DesignMatrixCreator_2dpol(PolyDeg,x,y)
        X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2) #Create / split / scale the design matrices
        X_train, X_test = scale(X_train, X_test)


        lambdavals = np.logspace(-10,3,200) #Create array of lambda values
        MSE_array = np.zeros(len(lambdavals))  #Create empty MSE array to be filled

        for i,lamb in enumerate(lambdavals): #Run through all values for lambda
            clf = Lasso(alpha=lamb).fit(X_train,z_train) #Fit model
            ztilde_test = clf.predict(X_test) #Use model to predict data
            MSE_array[i] = MSE(z_test,ztilde_test) #Fill MSE array

        #Plot MSE array
        plt.plot(lambdavals,MSE_array,label="Test data MSE")
        plt.axvline(lambdavals[np.argmin(MSE_array)],label="$\lambda$ = %e"%lambdavals[np.argmin(MSE_array)])
        plt.xlabel("Value for hyperparameter $\lambda$",fontsize="large")
        plt.ylabel("Mean Squared Error (MSE)",fontsize="large")
        plt.title("Lasso Hyperparameter fit for $\lambda$",fontsize="x-large")
        plt.semilogx(); plt.grid(); plt.legend()
        plt.show()

    elif decisions == 'b': #Complexity vs error
        MaxPoly = 40 #Max polynomial degree to run up to
        N = 200 #Number of data points
        noise = 0.2 #Noise term
        testsize = 0.2 #Test size when splitting the design matrix

        xy = np.random.rand(N,2)
        x = xy[:,0]; y = xy[:,1] #Create random function parameters and corresponding function values.
        z = frankefunc_noise(x,y,noise)

        lambdavals = np.logspace(-10,5,30) #Create a lambda array

        MSE_train_array = np.zeros(MaxPoly) #Create empty MSE arrays to be filled
        MSE_test_array = np.zeros(MaxPoly)

        for polydeg in range(1,MaxPoly+1): #Run through all polynomial degrees
            X = DesignMatrixCreator_2dpol(polydeg,x,y)
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=testsize) #Create / Split / Scale design matrices
            X_train, X_test = scale(X_train, X_test)

            MSE_array_testlamb = np.zeros(len(lambdavals)) #Create temporary MSE arrays to be filled
            MSE_array_trainlamb = np.zeros(len(lambdavals))

            for i, lamb in enumerate(lambdavals): #Run through all values of lambda
                clf = Lasso(alpha=lamb).fit(X_train,z_train)

                ztilde_test = clf.predict(X_test)
                ztilde_train = clf.predict(X_train)

                MSE_array_testlamb[i] = MSE(z_test,ztilde_test)
                MSE_array_trainlamb[i] = MSE(z_train,ztilde_train)

            MSE_test_array[polydeg-1] = np.min(MSE_array_testlamb) #Pick out the lambda corrsponding to the best MSE.
            MSE_train_array[polydeg-1] = MSE_array_trainlamb[np.argmin(MSE_array_testlamb)]

        #Plot results together
        polydegs = np.arange(1,MaxPoly+1)
        plt.plot(polydegs,MSE_train_array, label="Train")
        plt.plot(polydegs,MSE_test_array, label="Test")
        plt.xlabel("'Complexity' of model (Polynomial degree)",fontsize="large")
        plt.ylabel("Mean Squared Error (MSE)",fontsize="large")
        plt.title("N = %i, test size = %.1f%%, noise = %.2f\nLasso Shrinkage Method"% (N,testsize*100,noise),fontsize="x-large")
        plt.grid(); plt.legend(); plt.semilogy()
        plt.show()

    elif decisions == 'c': #Bootstrap
        print("\nHow many bootstrap runs do you wish to do?")
        n = int(input("Type here: ")) #Number of bootstraps
        MaxPoly = 20 #Max polynomial degree
        N = 200 #Number of data points
        noise = 1 #Noise factor
        testsize = 0.2 #Size of test data when splitting the design matrix

        xy = np.random.rand(N,2)
        x = xy[:,0]; y = xy[:,1] #Create function parameters and corresponding function values
        z = frankefunc_noise(x,y,noise)

        MSE_a = np.zeros(MaxPoly)
        bias_a = np.zeros(MaxPoly) #Create empty MSE, bias and variance arrays to be filled
        var_a = np.zeros(MaxPoly)

        for polydeg in range(1,MaxPoly+1): #Run through all polynomial degrees
            X = DesignMatrixCreator_2dpol(polydeg,x,y)
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=testsize) #Create / split / scale design matrix
            X_train, X_test = scale(X_train, X_test)

            MSE_a[polydeg-1], bias_a[polydeg-1], var_a[polydeg-1] = Func_Bootstrap(X_train, X_test, z_train, z_test, n, "Lasso") #Bootstrap lasso, then fill arrays

        polydeg_array = np.arange(1,MaxPoly+1) #Plot MSE, bias and variance together
        plt.plot(polydeg_array,bias_a,"--",label="Bias")
        plt.plot(polydeg_array,var_a,"--",label="Variance")
        plt.plot(polydeg_array,MSE_a,label="MSE")
        plt.xlabel("Complexity of model (Degree of Polynomial)",fontsize="large")
        plt.ylabel("Error score (MSE/Bias/Variance)",fontsize="large")
        plt.title("Bias-Variance tradeoff. N = %i, noise = %.2f, bootstraps=%i"%(N,noise,n),fontsize="x-large")
        plt.grid(); plt.legend(); plt.semilogy()
        plt.show()

    elif decisions == 'd': #CV
        K = int(input("Enter the number of desired folds: ")) #Number of K-folds
        n = int(input("Enter the number of desired bootstraps: ")) #Number of bootstraps
        N = 100; noise = 0.2; MaxPoly = 20; testsize = 0.2  #Number of data points, noise term, max polynomial degree, test size when splitting

        xy = np.random.rand(N,2)
        x = xy[:,0]; y = xy[:,1] #Create function parameters and corresponding function values
        z = frankefunc_noise(x,y,noise)

        mse_crossval_array = np.zeros(MaxPoly) #Create empty MSE arrays to be filled
        mse_bootstrap_array = np.zeros(MaxPoly)

        for polydeg in range(1,MaxPoly+1): #Run through all polynomial degrees
            X = DesignMatrixCreator_2dpol(polydeg,x,y)
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=testsize) #Create / split / scale design matrices
            X_train, X_test = scale(X_train, X_test)

            mse_crossval_array[polydeg-1] = func_cross_validation(polydeg, X, z, K, "Lasso") #Perform cross validation / bootstrap and fill arrays
            mse_bootstrap_array[polydeg-1] = Func_Bootstrap(X_train, X_test, z_train, z_test, n, "Lasso")[0]

        polydegs = np.arange(1,MaxPoly+1) #Plot MSE for cross_val and Bootstrap together
        plt.plot(polydegs,mse_crossval_array,label="Cross Validation")
        plt.plot(polydegs,mse_bootstrap_array,label="Bootstrap")
        plt.xlabel("Complexity of model (Degree of Polynomial)",fontsize="large")
        plt.ylabel("Error score (MSE)",fontsize="large")
        plt.title("Cross-Validation vs. Bootstrap.\n N = %i, noise = %.2f, bootstraps=%i"%(N,noise,n),fontsize="x-large")
        plt.grid(); plt.legend(); plt.semilogy()
        plt.show()

"""
Part f)
"""

if exercise == "f": #Plot map data

    print("\nExercise f doesn't really exist\n")
    print("It is only a placeholder for the download of data for the next exercise\n")
    print("However: we will gladly plot the selected image file for y'all!")

    terrainvar = imread('n59_e010_1arc_v3.tif') #Read image file

    plt.figure() #Plot image file
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

    if decisions == "a": #MSE vs Complexity for map data
        terrainvar = imread('n59_e010_1arc_v3.tif') #Read image file

        N = 1000 #Number of random points to extract
        x = np.random.randint(0,terrainvar.shape[1],size=N) #Create random indexes (corresponding to coordinates)
        y = np.random.randint(0,terrainvar.shape[0],size=N)
        z = terrainvar[y,x] #Extract corresponding value of map to the random indexes above

        MaxPoly = 45 #Max polynomial size of runs
        testsize = 0.2 #Test size to be used

        MSE_testOLS_array = np.zeros(MaxPoly)
        MSE_trainOLS_array = np.zeros(MaxPoly)
        MSE_testridge_array = np.zeros(MaxPoly) #Create empty arrays to be filled with MSE values
        MSE_testlasso_array = np.zeros(MaxPoly)

        polydegs = np.arange(1,MaxPoly+1)

        for polydeg in range(1,MaxPoly+1): #Run through all polynomial degrees
            print(polydeg) #This should be here just because simulation takes a while (gives an idea of how far the run has come)
            lambdavals = np.logspace(-10,5,100) #Initialize the lambda values

            X = DesignMatrixCreator_2dpol(polydeg,x,y)
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=testsize) #Create / split / scale design matrices
            X_train, X_test = scale(X_train, X_test)

            z_tilde_test, z_tilde_train, beta_optimal = OLS(X_train, X_test, z_train, z_test) #Run OLS

            MSE_testOLS_array[polydeg-1] = MSE(z_test,z_tilde_test)
            MSE_trainOLS_array[polydeg-1] = MSE(z_train,z_tilde_train) #Fill OLS arrays with MSE values

            z_tilde_test_ridge = Ridge(X_train, X_test, z_train, z_test, lambdavals)[0] #Run Ridge

            MSE_testridge_array[polydeg-1] = MSE(z_test,z_tilde_test_ridge) #Fill ridge array with MSE values

            MSE_temp_lasso = np.zeros(len(lambdavals))
            for i, lamb in enumerate(lambdavals): #Run Lasso
                clf = Lasso(alpha=lamb).fit(X_train,z_train)
                z_tilde_test_temp = clf.predict(X_test)
                MSE_temp_lasso[i] = MSE(z_test,z_tilde_test_temp)

            MSE_testlasso_array[polydeg-1] = np.min(MSE_temp_lasso) #Fill lasso with MSE values


        plt.figure() #Plot train and test data together for OLS
        plt.plot(polydegs,MSE_testOLS_array,label="Test OLS")
        plt.plot(polydegs,MSE_trainOLS_array,label="Train OLS")
        plt.xlabel("'Complexity' of model (Polynomial degree)",fontsize="large")
        plt.ylabel("Mean Squared Error (MSE)",fontsize="large")
        plt.title("N = %i, test size = %.1f%%,\nOLS On Topography Map"% (N,testsize*100),fontsize="x-large")
        plt.grid(); plt.legend(); plt.semilogy()

        plt.figure() #Plot MSE values for test data on all three methods together
        plt.plot(polydegs,MSE_testOLS_array,label="OLS")
        plt.plot(polydegs,MSE_testridge_array,label="Ridge")
        plt.plot(polydegs,MSE_testlasso_array,label="Lasso")
        plt.xlabel("'Complexity' of model (Polynomial degree)",fontsize="large")
        plt.ylabel("Mean Squared Error (MSE)",fontsize="large")
        plt.title("N = %i, test size = %.1f%%,\nMSE of test data on Topopgraphy map"% (N,testsize*100),fontsize="x-large")
        plt.grid(); plt.legend(); plt.semilogy()

        plt.show()

    elif decisions == "b": #Attempt at recreating the map
        terrainvar = imread('n59_e010_1arc_v3.tif') #Read initial map

        N = 10000 #Number of data points
        x = np.random.randint(0,terrainvar.shape[1],size=N) #Create random coordinates
        y = np.random.randint(0,terrainvar.shape[0],size=N)
        TrainingData = terrainvar[y,x] #Extract terrain value corresponding to random position

        print("\nDo you want to perform OLS, Ridge, or Lasso regression analysis?")
        print("Type 'a' for OLS, 'b' for Ridge or 'c' for Lasso:\n")
        choice = input("Type here: ")

        if choice == 'a': #OLS

            PolyDeg = 30 #Polynomial degree of choice

            X = DesignMatrixCreator_2dpol(PolyDeg,x,y)
            X_train, X_test, z_train, z_test = train_test_split(X,TrainingData,test_size=0.2) #Create / split / scale design matrices
            X_train, X_test = scale(X_train, X_test)

            z_tilde_test, z_tilde_train, beta_optimal = OLS(X_train, X_test, z_train, z_test) #Run OLS

            MSE_train = MSE(z_train,z_tilde_train)
            MSE_test = MSE(z_test,z_tilde_test)

            ApproxImg = np.zeros(terrainvar.shape) #Create empty shell to be filled with our model of the map

            for y_indx in range(terrainvar.shape[0]): #Attempt at recreating the map, row by row
                X_temp = DesignMatrixCreator_2dpol(PolyDeg,np.arange(terrainvar.shape[1]),y_indx*np.ones(terrainvar.shape[1])) #Create design matrix corresponding to row
                X_temp = scale(X_train, X_temp)[1] #Scale this
                print(y_indx) #Slow running process, should be left in to see how far run has come
                ApproxImg[y_indx] = X_temp @ beta_optimal #Recreate the image back, row by row.
                del X_temp #For some reason won't run without this

            plt.figure() #Plot our attempt at recreated map
            plt.title("Approximate map\nPolynomial Degree: %i , MSE value: %e"%(PolyDeg,MSE_test),fontsize="x-large")
            plt.imshow(ApproxImg,cmap='gray')
            plt.xlabel("<-- West - East -->",fontsize="large")
            plt.ylabel("<-- South - North --->",fontsize="large")
            plt.xticks([]); plt.yticks([])


            plt.figure() #Plot the actual map
            plt.title("Actual map",fontsize="x-large")
            plt.imshow(terrainvar,cmap='gray')
            plt.xlabel("<-- West - East -->",fontsize="large")
            plt.ylabel("<-- South - North --->",fontsize="large")
            plt.xticks([]);plt.yticks([])
            plt.show()

        elif choice == 'b': #Ridge
            PolyDeg = 30 #Set polynomial degree of model

            X = DesignMatrixCreator_2dpol(PolyDeg,x,y)
            X_train, X_test, z_train, z_test = train_test_split(X,TrainingData,test_size=0.2) #Create / split / scale the design matrix
            X_train, X_test = scale(X_train, X_test)

            lambdavals = np.logspace(-10,3,100) #Set the possible lambda values
            z_tilde_test, z_tilde_train, beta_optimal = Ridge(X_train, X_test, z_train, z_test, lambdavals)[:3] #Run Ridge model

            MSE_train = MSE(z_train,z_tilde_train) #Calculate the MSE score of model
            MSE_test = MSE(z_test,z_tilde_test)

            ApproxImg = np.zeros(terrainvar.shape) #Create empty shell to be filled with our model of the map

            for y_indx in range(terrainvar.shape[0]): #Attempt at recreating the map, row by row
                X_temp = DesignMatrixCreator_2dpol(PolyDeg,np.arange(terrainvar.shape[1]),y_indx*np.ones(terrainvar.shape[1])) #Create design matrix corresponding to row
                X_temp = scale(X_train, X_temp)[1] #And we scale this matrix
                ApproxImg[y_indx] = X_temp @ beta_optimal #Recreate the image back, row by row
                print(y_indx) #Slow running program, leave this in as indicator on how far you've come
                del X_temp #Needs to be here for program to run.


            plt.figure() #Plot our attempt at recreated map
            plt.title("Approximate map using Ridge \nPolynomial Degree: %i , MSE value: %e" % (PolyDeg,MSE_test),fontsize="x-large")
            plt.imshow(ApproxImg,cmap='gray')
            plt.xlabel("<-- West - East -->",fontsize="large")
            plt.ylabel("<-- South - North --->",fontsize="large")
            plt.xticks([]); plt.yticks([])


            plt.figure() #Plot the actual map
            plt.title("Actual map",fontsize="x-large")
            plt.imshow(terrainvar,cmap='gray')
            plt.xlabel("<-- West - East -->",fontsize="large")
            plt.ylabel("<-- South - North --->",fontsize="large")
            plt.xticks([]);plt.yticks([])
            plt.show()

        elif choice == 'c': #Lasso
            PolyDeg = 30 #Polynomial degree of model

            X = DesignMatrixCreator_2dpol(PolyDeg,x,y)
            X_train, X_test, z_train, z_test = train_test_split(X,TrainingData,test_size=0.2) #Create / split / scale the design matrices
            X_train, X_test = scale(X_train, X_test)

            lambdavals = np.logspace(-10,3,100) #Array of lambda values to run through

            mse_temp_array = np.zeros(lambdavals.shape[0]) #Temporary MSE and beta arrays to be filled when running Lasso
            beta_temp_array = np.zeros([lambdavals.shape[0],X.shape[1]])

            for i, lamb in enumerate(lambdavals): #Run through all values of lambda
                clf = Lasso(alpha=lamb).fit(X_train,z_train) #Create model using lasso
                beta_temp_array[i] = clf.coef_  #Find coefficients of model
                mse_temp_array[i] = MSE(z_test,clf.predict(X_test)) #Find MSE of test data on model

            MSE_test = np.min(mse_temp_array) #Extracting lowest MSE value
            beta_optimal = beta_temp_array[np.argmin(MSE_test)] #Optimal beta corresponding to polynomial degree.

            ApproxImg = np.zeros(terrainvar.shape) #Create empty shell for recreating map

            for y_indx in range(terrainvar.shape[0]): #Run through this, row by row
                X_temp = DesignMatrixCreator_2dpol(PolyDeg,np.arange(terrainvar.shape[1]),y_indx*np.ones(terrainvar.shape[1])) #Create design matrix corresponding to row
                X_temp = scale(X_train,X_temp)[1] #Scale said design matrix
                ApproxImg[y_indx] = X_temp @ beta_optimal #Recreate map for each row
                print(y_indx) #Slow to run, should be left in to give an idea of how far the program has come
                del X_temp #Need this in so that it runs


            plt.figure() #Plot our attempt at recreated map
            plt.title("Approximate map using Lasso \nPolynomial Degree: %i , MSE value: %e" % (PolyDeg,MSE_test),fontsize="x-large")
            plt.imshow(ApproxImg,cmap='gray')
            plt.xlabel("<-- West - East -->",fontsize="large")
            plt.ylabel("<-- South - North --->",fontsize="large")
            plt.xticks([]); plt.yticks([])


            plt.figure() #Plot the actual map
            plt.title("Actual map",fontsize="x-large")
            plt.imshow(terrainvar,cmap='gray')
            plt.xlabel("<-- West - East -->",fontsize="large")
            plt.ylabel("<-- South - North --->",fontsize="large")
            plt.xticks([]);plt.yticks([])
            plt.show()

        else: #Just decided to mock whoever can't press 'a', 'b' or 'c'..
            print("\nLol you can't even follow simple instructions")
            sys.exit(0)


"""
Easter egg! Compare the MSEs for the Franke Function
"""

if exercise == "h": #Plot MSE vs complexity together!
    print("\nYou just activated an easter egg!")
    print("\nWe will now present you with an MSE vs complexity plot for all the methods")
    print("\nThis is for the Franke function only, to be frank.\n")
    input("Press enter to continue.")

    MaxPoly = 30 #Max polynomial degree
    N = 100 #Number of data points
    noise = 2 #Factor of noise
    testsize = 0.2 #Test size when splitting the data

    xy = np.random.rand(N,2)
    x = xy[:,0]; y = xy[:,1] #Create (random) function parameters and corresponding functional values
    z = frankefunc_noise(x,y,noise)

    lambdavals = np.logspace(-10,5,100) #Create array of Lambda values to be used

    MSE_ols_array = np.zeros(MaxPoly)
    MSE_ridge_array = np.zeros(MaxPoly) #Create empty MSE arrays to be filled
    MSE_lasso_array = np.zeros(MaxPoly)

    for polydeg in range(1,MaxPoly+1): #Run through all polynomials
        print(polydeg) #Slow to run, left in to give indication of how far it has run.


        X = DesignMatrixCreator_2dpol(polydeg,x,y)
        X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=testsize) #Create / split / scale the design matrix
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

    polydegs = np.arange(1,MaxPoly+1) #Plot OLS, Ridge and Lasso MSE's together.
    plt.plot(polydegs, MSE_ols_array, label="OLS")
    plt.plot(polydegs, MSE_ridge_array, label="Ridge")
    plt.plot(polydegs, MSE_lasso_array, label="Lasso")
    plt.xlabel("'Complexity' of model (Polynomial degree)",fontsize="large")
    plt.ylabel("Mean Squared Error (MSE)",fontsize="large")
    plt.title("N = %i, test size = %.1f%%, noise = %.2f"% (N,testsize*100,noise),fontsize="x-large")
    plt.legend(); plt.grid(); plt.semilogy()
    plt.show()
