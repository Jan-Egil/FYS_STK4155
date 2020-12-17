import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
plt.style.use('seaborn')


def init_condition(x):
    return np.sin(np.pi*x)

def analytic(x,t):
    return np.sin(np.pi*x)*np.exp(-np.pi**2*t)

"""
START FREE VARIABLES
"""

sim_length = 0.3
del_x = 1/100
del_t = 1/20000
bc1 = 0 #Boundary condition at x=0
bc2 = 0 #Boundary condition at x=1

"""
END FREE VARIABLES
"""


#Check stability criteria
alpha = del_t/(del_x**2)
if alpha > 0.5:
    print("This does not meet the stability criteria!")
    print("%g/(%g)^2 = %.2f > 0.5" % (del_t,del_x,del_t/(del_x**2)))
    print("\nExiting program...")
    sys.exit(0)

t_steps = np.arange(0 , sim_length+del_t , del_t)
x_points = np.arange(0 , 1+del_x , del_x)
frames = np.zeros([t_steps.shape[0],x_points.shape[0]])

funcval = init_condition(x_points)
frames[0] = funcval

prevval = np.copy(funcval)
print("\n\n\n")
for j,t in enumerate(tqdm(t_steps[:-1])):
    newval = np.zeros(x_points.shape[0])
    newval[0] = bc1
    newval[-1] = bc2
    for i in range(1,x_points.shape[0]-1):
        newval[i] = prevval[i] + alpha*(prevval[i+1]+prevval[i-1]-2*prevval[i])
    frames[j+1] = prevval
    prevval = np.copy(newval)


pointsx,pointst = np.meshgrid(x_points,t_steps)
trueval = np.sin(np.pi*pointsx)*np.exp(-np.pi**2*pointst)

plt.figure()
plt.plot(x_points,frames[0],label="Initial")
plt.plot(x_points,frames[500],label="In the middle")
plt.plot(x_points,frames[-1],label="Final")
plt.legend(fontsize='x-large')
plt.xlabel("Position 'x' [Rel units]",fontsize='x-large')
plt.ylabel("u(x,t) at locked times 't'",fontsize='x-large')
plt.title("Initial, middle-point and final state for simulation of length $t = %.2f$\n $\Delta x = \\frac{1}{%i}$" % (sim_length,del_x**(-1)),fontsize='x-large')

plt.figure()
plt.contourf(x_points,t_steps,frames,levels=50,cmap='viridis')
plt.xlabel("Position 'x' [Rel units]",fontsize='x-large')
plt.ylabel("Time 't' [Rel units]",fontsize='x-large')
plt.title("Evolution of $u(x,t)$ in both 1D space and time.\n$\Delta x = \\frac{1}{%i}$" % (del_x**(-1)),fontsize='x-large')
plt.colorbar().set_label("$u(x,t)$",fontsize='x-large')

plt.figure()
plt.contourf(pointsx,pointst,np.abs(trueval-frames),levels=50,cmap='viridis')
plt.xlabel("Position 'x' [Rel units]",fontsize='x-large')
plt.ylabel("Time 't' [Rel units]",fontsize='x-large')
plt.title("Absolute difference between analytic and numerical value.\n$\Delta x = \\frac{1}{%i}$" % (del_x**(-1)),fontsize='x-large')
plt.colorbar().set_label("Absolute difference",fontsize='x-large')

plt.show()
