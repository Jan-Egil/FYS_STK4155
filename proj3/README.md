# FYS-STK4155, Project 3

 **Author:** Jan Egil Ødegård

The programs in this folder contain code attempting to solve both Partial Differential Equations (PDEs), namely the Heat Equation, and code attempting to solve non-linear differential equations, both using neural networks. For the PDE case, the program is functioning, but is unimaginable slow, while it is non-functioning for the non-linear differential equation case. The non-linear differential equation should in turn be possible to solve for eigenvectors and eigenvalues for a symmetric, real matrix.

# Structure of repo

The repository is split into the following folders:

*Code*: Contains the python-scripts.

*Figures*: Contains our results from running the codes. These figures are also seen and discussed in the report.

*PDF*: Contains the project report as a PDF

Within "Code", there are several python scripts for different programs running the aforementioned codes. There are some codes that have a dependency on being in the same folder as "Neural_Network_Classes.py". Note that especially the Neural_Network_Classes.py-code is based on the raw code provided in the lecture slides by Morten Hjorth-Jensen, attempted implemented in the code structure we created in project 2. As such, it might bear resemblance to both previous projects, and Morten's code, both at the same time.

The programs and what they do are as follows:

**Neural_Network_Classes.py:** Object-oriented neural network class for both PDE and eigenvalue solving is contained in this program. Eigval_NN_solver.py and PDE_NN_solver.py depends on this program. Includes 2 classes. PDE-class is functioning, while eigval-class is not.

**Eigval_NN_solver.py:** A (pretty unfinished) script which optimally should use the eigenvalue-solver class from Neural_Network_Classes.py to approximate value of eigenvector. Depends on Neural_Network_classes.py to run

**PDE_NN_solver.py:** A script which, given some input of variables, creates and trains a neural network to solve a PDE (the heat equation). Depends on Neural_Network_Classes.py being in the same folder.

**explicit_euler.py:** Applies the Explicit scheme / forward Euler method to solve the heat equation. Used to compare to the Neural Network case.
