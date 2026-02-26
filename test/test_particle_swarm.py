import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def local_best_get(particle_pos,particle_pos_val,p):
    local_best=[0]*p #creating empty local best list
    for j in range(p):  #finding the best particle in each neighbourhood
                        #and storing it in 'local_best'
        local_vals=np.zeros(3)
        local_vals[0]=particle_pos_val[j-2]
        local_vals[1]=particle_pos_val[j-1]
        local_vals[2]=particle_pos_val[j]
        min_index=int(np.argmin(local_vals))
        local_best[j-1]=particle_pos[min_index+j-2][:]
    return np.array(local_best)

def rosenbrock(theta):
    theta = np.atleast_2d(theta)  # ensure shape for both minimize and NelderMeadSimplex
    first_half = 100 * (theta[:, 1] - theta[:, 0] ** 2) ** 2
    second_half = (1 - theta[:, 0]) ** 2

    return first_half + second_half  # rosenbrock function

x_points = np.arange(-1.5, 1.5, 1e-2)
y_points = np.arange(-1.5, 1.5, 1e-2)
rosen_array = np.zeros(shape=(len(x_points), len(y_points)))
for i in range(len(x_points)):
    for j in range(len(y_points)):
        rosen_array[i,j] = rosenbrock(np.array([x_points[i], y_points[j]]))

#Global best Particle Swarm Optimization algorithm
swarm_size = 60 #number of particles in the swarm
n = 2 #dimension of the problem
bounds = [-1.5,1.5]
c1 = 2 #cognitive acceleration
c2 = 2 #social acceleration
max_iter = 1000
n_iter = 0
weight = 0.8
tol = 1e-6

#randomly generate the positions and velocities of the swarm
swarm_positions = np.random.uniform(bounds[0], bounds[1], size=(swarm_size,n)) #array to store the positions
swarm_costs = rosenbrock(swarm_positions) #compute the cost at each swarm
swarm_velocities = np.random.uniform(bounds[0], bounds[1], size=(swarm_size,n)) #array to store the positions
P_best = swarm_positions #the best position found by each particle
cost_P_best = swarm_costs #the cost at the best position found by each particle
idx_best = np.argmin(cost_P_best) #find the index of the best cost up until now
G_best = P_best[idx_best,:]*np.ones_like(swarm_positions) #find the position of the best particle in the swarm
old_best = 0
points = []
swarms = []

while True:
    n_iter += 1
    if n_iter == max_iter:
        break

    points.append(G_best[0, :])
    swarms.append(P_best)
    r1,r2 = np.random.uniform(0,1,2) #random uniform values
    swarm_velocities = weight*swarm_velocities + c1*r1*(P_best-swarm_positions) + c2*r2*(G_best-swarm_positions) #evaluate the new velocity
    swarm_positions += swarm_velocities #update the positions given the velocity
    swarm_costs = rosenbrock(swarm_positions) #update the costs
    cost_mask = swarm_costs < cost_P_best #find where the new positions return the better costs
    P_best[cost_mask] = swarm_positions[cost_mask] #update the best positions
    P_best = np.clip(P_best, bounds[0], bounds[1]) #respect the bounds
    cost_P_best[cost_mask] = swarm_costs[cost_mask] #update the best costs
    idx_best = np.argmin(cost_P_best) #find the new minimum
    G_best = P_best[idx_best,:]*np.ones_like(swarm_positions) #update new global best

    if np.abs(cost_P_best[idx_best]-old_best) < tol:
        break

points = np.array(points) #convert points to array
swarms = np.array(swarms) #convert swarms to array
fig, ax = plt.subplots()

def animate(i):
    ax.clear()
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    leg = []
    rosen_image = ax.imshow(rosen_array.T, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5])
    curr_swarm = ax.scatter(swarms[i,:,0], swarms[i,:,1], c='tab:blue', label='guesses', s=2)
    leg.append('Swarm at the PSO iteration')
    guess = ax.scatter(points[:i,0], points[:i,1], c='tab:orange', label='guesses', s=4)
    leg.append('Best particle at the PSO iteration')
    true_min = ax.scatter(1,1, marker='x', color='tab:red', label=' true minimum')
    leg.append('True minimum')
    ax.legend(leg, loc='lower left')

    return rosen_image, curr_swarm, guess, true_min

ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=points.shape[0])
ani.save("./PSO.gif", dpi=300, writer=PillowWriter(fps=25))