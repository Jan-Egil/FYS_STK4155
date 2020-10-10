# FYS-STK4155, Project 1

 **Authors:** Martin Moen Carstensen, George Stanley Cowie & Jan Egil Ødegård

This program runs several linear regression models to fit data. First to an analytic function called the Franke function, then to actual topographical map data provided by NASA. We will be using Ordinary Least Squares (OLS), Ridge, & Lasso as regression methods. We will be using bootstrap resampling and
Leave-One-Out Cross Validation (LOOCV) resampling methods to optimize our results.

# Structure of repo

The repository is split into the following folders:

*Code*: Contains the code, and separate .tif-files required to run the files.

*Figures*: Contains our results from running the codes. These figures are also discussed in the report.

*PDF*: Contains the project report as a PDF

Within "Code", there is a main.py which is the file to be run. It uses "func.py" to import all packages, as well as the functions hand-crafted for this project.
There is a "plotfrankefunc.py"-program here as well. This is a blatant copy/paste from the project description.

The program is split into one chunk for each of the subproblems in the project description. They are as follows:

**a)** Basic OLS regression. Compare the results with the ones from SKLearns metric-package. Calculations of confidence intervals of regression parameters.

**b)** Complexity vs Error plot. Introduce the Bootstrap resampling method. Use
Bootstrap to perform Bias-variance trade-off analysis. Still using OLS

**c)** Introduce LOOCV and optimize using this resampling technique.

**d)** Much the same as the previous 3 exercises, but replace OLS with Ridge

**e)** Much the same as in exercise d), but replace Ridge with Lasso.

**f)** Not really an exercise for programming. Plots the chosen topographical
map that we will use for the next exercise.

**g)** Much of the same as subproblems a) through e), but with real data as given by NASA SRTM maps. Not that thorough of a analysis and more on the qualitative results.
