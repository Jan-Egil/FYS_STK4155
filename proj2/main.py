
from func import *#importing everything from func.py, including external packages

print("\n\nWhich task do you want to run?")
exercise = input("Press any letter between a & g: ")

"""
Part a)
"""

if exercise == "a":
    M = 5#Minibatch size
    n = 100#number of datapoints
    m = n/M
    epochs = 10
    Tolerance = 1e-10
    
