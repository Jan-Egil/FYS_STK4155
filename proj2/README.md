# FYS-STK4155, Project 2

 **Authors:** Martin Moen Carstensen, George Stanley Cowie & Jan Egil Ødegård

The programs in this folder contains both classification and linear regression codes using different methods. For the linear regression case, we got programs running OLS and ridge, both using matrix inversion methods and gradient methods. We've also used a neural network to 'linearly regress'. For the classification case, we've used a logistic regression method using softmax regression, as well as also implemented a neural network here.

# Structure of repo

The repository is split into the following folders:

*Code*: Contains the python-scripts.

*Figures*: Contains our results from running the codes. These figures are also seen and discussed in the report.

*PDF*: Contains the project report as a PDF

Within "Code", there are several python scripts for different programs running the aforementioned models.
All of the codes are dependent on having func.py in the same folder, as well as some programs also needing NeuralNetwork.py in the same folder. Keep this in mind dependency-wise.

The programs and what they do are as follows:

**func.py:** All the specifically-for-this-project written functions, all gathered in one file. Imported by all other files in folder.

**NeuralNetwork.py:** A class containing our Neural Network code, imported by NN_franke.py and classification.py

**SGD_vs_OLS_complexity.py:** Performs the SGD algorithm and matrix inversion algorithm for the OLS case, and compares these in regards to the complexity of the model and an accuracy score. (Mean Squared Error)

**SGD_vs_OLS_time.py:** Performs the SGD algorithm and matrix inversion algorithm for the OLS case, and compares these in regards to CPU time spent and an accuracy score (Mean Squared Error)

**SGD_ridge.py:** Performs a comparison between the ridge hyperparameter and the learning rate when applied to an SGD method.

**NN_franke.py:** Models the Franke function using a neural network that is trained, with the possibility to change parameters, design of neural network, etc.. Has a dependency on NeuralNetwork.py

**LogisticRegression.py:** Performs softmax logistic regression on the MNIST hand written digits data set for a given set of learning rate. Returns confusion matrix as well.

**LogistricRegressionVsLearnRate.py:** Performs softmax logistic regression on the MNIST hand written digits data set for a set of different learning rates, comparing these.

**Classification.py:** Neural network, trained to recognize hand written digits from the MNIST data set. Possibility to tweak parameters and design of neural network. Has a dependency on NeuralNetwork.py
