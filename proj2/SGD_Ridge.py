from func import *
import time

t = time.process_time()
#do some stuff
elapsed_time = time.process_time() - t

lambvals = np.logspace(-6,3,10)
learning_rates = np.logspace(-6,3,10)
