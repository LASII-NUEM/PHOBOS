from scipy.optimize import minimize
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation, PillowWriter

def rosen_grad(theta):
    grad = np.zeros(shape=(2,1))
    theta = np.atleast_2d(theta)  # ensure shape for both minimize and NelderMeadSimplex
    grad[0,:] = 400*theta[0,0]*(-theta[1,0]+theta[0,0]**2) - 2*(1-theta[0,0])
    grad[1,:] = 200*(theta[1,0]-theta[0,0]**2)
    return grad

def rosen_hess(theta):
    hess = np.zeros(shape=(2,2))
    hess[0,0] = 400*(-theta[1,0]+3*theta[0,0]**2)+2
    hess[0,1] = -400*theta[0,0]
    hess[1,0] = -400*theta[0,0]
    hess[1,1] = 200
    return hess

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

#run newton
initial_guess = np.array([-1.2, 1])
initial_guess = np.atleast_2d(initial_guess).T
tol = 1e-6
points = []
n_iter = 0
max_iter = 200
while True:
    n_iter += 1

    if n_iter >= max_iter:
        break

    curr_grad = rosen_grad(initial_guess)
    curr_hess = rosen_hess(initial_guess)
    points.append(initial_guess)

    #evaluate gradient
    if np.linalg.norm(curr_grad)<tol:
        break

    initial_guess = initial_guess - np.linalg.inv(curr_hess)@curr_grad

points = np.array(points)[:,:,0]
fig, ax = plt.subplots()

def animate(i):
    ax.clear()
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    leg = []
    rosen_image = ax.imshow(rosen_array.T, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5])
    guess = ax.scatter(points[:i,0], points[:i,1], c='tab:orange', label='guesses', s=4)
    leg.append('Newton-Raphson iteration')
    true_min = ax.scatter(1,1, marker='x', color='tab:red', label=' true minimum')
    leg.append('True minimum')
    ax.legend(leg, loc='lower left')

    return rosen_image, guess, true_min

ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=points.shape[0])
ani.save("./NewtonRaphson.gif", dpi=300, writer=PillowWriter(fps=25))