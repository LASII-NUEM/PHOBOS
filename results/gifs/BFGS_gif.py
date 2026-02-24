from scipy.optimize import minimize
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation, PillowWriter

points = []
def bfgs_callback(x):
    points.append(x.copy())

def rosenbrock(theta):
    theta = np.atleast_2d(theta) #ensure shape for both minimize and NelderMeadSimplex
    first_half = 100*(theta[:,1]-theta[:,0]**2)**2
    second_half = (1-theta[:,0])**2

    return first_half + second_half #rosenbrock function

x_points = np.arange(-1.5, 1.5, 1e-2)
y_points = np.arange(-1.5, 1.5, 1e-2)
rosen_array = np.zeros(shape=(len(x_points), len(y_points)))
for i in range(len(x_points)):
    for j in range(len(y_points)):
        rosen_array[i,j] = rosenbrock(np.array([x_points[i], y_points[j]]))

#run BFGS
theta = minimize(rosenbrock, np.array([-1.2, 1]), method='L-BFGS-B', callback=bfgs_callback)
points = np.array(points) #convert points to array

fig, ax = plt.subplots()

def animate(i):
    ax.clear()
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    leg = []
    rosen_image = ax.imshow(rosen_array.T, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5])
    guess = ax.scatter(points[:i,0], points[:i,1], c='tab:orange', label='guesses', s=4)
    leg.append('BFGS iteration')
    true_min = ax.scatter(1,1, marker='x', color='tab:red', label=' true minimum')
    leg.append('True minimum')
    ax.legend(leg, loc='lower left')

    return rosen_image, guess, true_min

ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=points.shape[0])
ani.save("./BFGS.gif", dpi=300, writer=PillowWriter(fps=25))