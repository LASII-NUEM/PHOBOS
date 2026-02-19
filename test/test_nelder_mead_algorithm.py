import numpy as np
from scipy.optimize import minimize
import time

def powell(theta):
    theta = np.atleast_2d(theta) #ensure shape for both minimize and NelderMeadSimplex
    q1 = (theta[:,0]+theta[:,1])**2
    q2 = 5*(theta[:,2]-theta[:,3])**2
    q3 = (theta[:,1]-2*theta[:,2])**4
    q4 = 10*(theta[:,0]-theta[:,3])**4
    
    return q1 + q2 + q3 + q4

def rosenbrock(theta):
    theta = np.atleast_2d(theta) #ensure shape for both minimize and NelderMeadSimplex
    first_half = 100*(theta[:,1]-theta[:,0]**2)**2
    second_half = (1-theta[:,0])**2

    return first_half + second_half #rosenbrock function

def centroid(coordinates):
    return np.sum(coordinates, axis=0)/len(coordinates)

def NelderMeadSimplex(func, theta, alfa=1, beta=2, gamma=0.5, step=0.0025, tol=1e-8, max_iter=1000):
    '''
    :param func: pointer to the cost function of the minimization problem
    :param theta: initial point for the simplex (P[0])
    :param alfa: reflection coefficient
    :param beta: expansion coefficient
    :param gamma: contraction coefficient
    :param step: step to genereta the Simplex from P[0]
    :param tol: tolerance of the algorithm for stop criterium (std(y) < tol)
    :param max_iter: total allowed iterations of the algorithm
    :return:
    '''

    #control variables of the algorithm
    n = len(theta) #points that define the simplex
    y_idx = np.arange(0, n+1, 1) #indexes to extract l, h and s
    iter = 0 #iteration counter

    #define the simplex
    step_mtx = step*np.roll(np.eye(n+1,n),-1) #matrix that defines the vertices
    simplex = step_mtx + np.tile(theta[:,np.newaxis], n+1).T #apply the step

    #Nelder-Mead simplex algorithm
    while True:
        iter += 1 #update the iteration counter
        y = func(simplex) #compute the cost at each vertex of the simplex

        #stop criterion
        delta = np.std(y)
        if delta < tol:
            break

        h = np.argmax(y) #index of the maximum cost
        yh = func(simplex[h,:][:,np.newaxis].T)
        y_idx_nH = y_idx !=h #mask to ensure i!=h in the comparison
        l = np.argmin(y) #index of the minimum cost
        yl = func(simplex[l,:][:,np.newaxis].T) #update the cost at the lower bound
        y_idx_cent = (y_idx!=h)&(y_idx!=l) #mask to detect second highest cost
        P_cent = centroid(simplex[y_idx_nH]) #compute the centroid without h
        s = np.argmax(y[y_idx_cent]) #index of the second highest cost
        y_s = func(simplex[s,:][:,np.newaxis].T)

        #reflection
        P_r = P_cent + alfa*(P_cent-simplex[h,:])
        y_r = func(P_r[:,np.newaxis].T)

        #expansion
        if y_r < yl:
            P_e = P_cent + beta*(P_r-P_cent)
            y_e = func(P_e[:,np.newaxis].T)
            if y_e < y_r:
                simplex[h,:] = P_e
            elif y_e >= y_r:
                simplex[h,:] = P_r

        #contraction
        elif y_r >= y_s:
            if y_r<y[h]:
                simplex[h,:] = P_r

            P_c = P_cent + gamma*(simplex[h,:]-P_cent)
            y_c = func(P_c[:, np.newaxis].T)

            if y_c > yh:
                simplex[y_idx_cent,:] = 0.5*(simplex[y_idx_cent,:]+simplex[l,:])
                y = func(simplex)

            elif y_c <= yh:
                simplex[h,:] = P_c
        else:
            simplex[h,:] = P_r

        #break
        if iter == max_iter:
            break

    opt_point = np.argmin(y) #optimal point at the minimal value of the cost

    return simplex[opt_point,:], y[opt_point]

#Nelder-Mead Simplex solution
t_init = time.time()
x_simplex, y_simplex = NelderMeadSimplex(rosenbrock, np.array([-1.2,1]), tol=1e-8)
print(f'[Simplex] t_elapse = {time.time()-t_init}')
print(f'[Simplex] opt_params = {x_simplex}')
print(f'[Simplex] min. cost = {y_simplex}')

#BFGS solution
t_init = time.time()
res_BFGS = minimize(rosenbrock, np.array([-1.2,1]), tol=1e-8, method="Nelder-Mead")
print(f'[BFGS] t_elapse = {time.time()-t_init}')
print(f'[BFGS] opt_params = {res_BFGS.x}')
print(f'[BFGS] opt_params = {res_BFGS.fun}')