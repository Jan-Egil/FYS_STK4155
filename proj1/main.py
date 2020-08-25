import numpy as np
import matplotlib.pyplot as plt

def frankefunc(self,x,y):
    term1 = 0.75*np.exp(-((9*x-2)**2)/4 - ((9*x-2)**2)/4)
    term2 = 0.75*np.exp(-((9*x+1)**2)/49 - (9*y+1)/10)
    term3 = 0.5*np.exp(-((9*x-7)**2)/4 - ((9*y-3)**2)/4)
    term4 = 0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 - term4
