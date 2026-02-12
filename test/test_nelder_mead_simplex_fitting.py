import numpy as np
from framework import file_lcr, fitting_utils, equivalent_circuits
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def centroid(coordinates):
    return np.sum(coordinates, axis=0)/len(coordinates)

def CUMSE(theta, args):
    z_hat = circuit_impedance(theta, [args[1], args[2]])
    z_hat = z_hat.astype("complex").T
    args[0] = args[0].astype("complex")
    SSE = np.sum(((args[0].real-z_hat.real)**2) + ((args[0].imag-z_hat.imag)**2), axis=1)

    return SSE/len(z_hat)

def NMSE(z:np.ndarray, z_hat:np.ndarray):
    '''
    :param z: the observed values (real measurements)
    :param z_hat: the predicted values from the fitted circuit
    :return: NMSE of the fit
    '''

    #validate 'z'
    if not isinstance(z, np.ndarray):
        raise TypeError(f'[EquivalentCircuit] "z" must be a Numpy Array! Curr. type = {type(z)}')

    #validate 'z_hat'
    if not isinstance(z_hat, np.ndarray):
        raise TypeError(f'[EquivalentCircuit] "z_hat" must be a Numpy Array! Curr. type = {type(z_hat)}')

    #validate shape
    if len(z) != len(z_hat):
        raise ValueError(f'[EquivalentCircuit] "z" and "z_hat" must match in length!')

    #normalized mean squared error
    SSE = np.sum(((z_hat.real-z.real)**2) + ((z_hat.imag-z.imag)**2)) #sum of squared errors
    SSO = np.sum((z.real**2) + (z.imag**2)) #sum of squared measurements

    return SSE/SSO

alfa=1
beta=2
gamma=0.5
step=0.05
tol=1e-8
max_iter=None

#PHOBOS spectroscopy acquisition
spec_ice_obj = file_lcr.read('../data/testICE_30_01_26/c_ice.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
freq_thresh = 100
freq_mask = spec_ice_obj.freqs >= freq_thresh
spec_ice_obj.freqs = spec_ice_obj.freqs[freq_mask]
spec_ice_obj.Cp = spec_ice_obj.Cp[freq_mask]
spec_ice_obj.Rp = spec_ice_obj.Rp[freq_mask]
spec_ice_obj.n_freqs = len(spec_ice_obj.freqs)

fit_obj = fitting_utils.EquivalentCircuit("Longo2020", spec_ice_obj, spec_ice_obj.freqs) #quivalent circuit object
circuit_impedance = equivalent_circuits.Longo2020
scaling =  np.array([1e5, 1e-6, 1e7, 1e-2, 1e4, 1e-1, 1, 1])
args = [fit_obj.z_meas, 2*np.pi*spec_ice_obj.freqs, scaling]
theta = np.array([1, 1, 1, 1, 1, 1, 0.5, 1])
func = CUMSE

#control variables of the algorithm
n = len(theta) #points that define the simplex
y_idx = np.arange(0, n+1, 1) #indexes to extract l, h and s
iter = 0 #iteration counter

if max_iter is None:
    max_iter = n*200

costs = []

#define the simplex
step_mtx = step*np.roll(np.eye(n+1,n),0) #matrix that defines the vertices
simplex = step_mtx + np.tile(theta[:,np.newaxis], n+1).T #apply the step
bounds = np.array(fitting_utils.function_handlers["longo2020"]["bounds"])

#Nelder-Mead simplex algorithm
while True:
    iter += 1 #update the iteration counter
    simplex = np.clip(simplex, bounds[:,0], bounds[:,1])
    y = func(simplex, args) #compute the cost at each vertex of the simplex

    #stop criterion
    delta = np.std(y)
    if delta < tol:
        break

    h = np.argmax(y) #index of the maximum cost
    yh = func(simplex[h,:][:,np.newaxis].T, args)
    y_idx_nH = y_idx !=h #mask to ensure i!=h in the comparison
    l = np.argmin(y) #index of the minimum cost
    yl = func(simplex[l,:][:,np.newaxis].T, args) #update the cost at the lower bound
    y_idx_cent = (y_idx!=h)&(y_idx!=l) #mask to detect second highest cost
    P_cent = np.mean(simplex[y_idx_nH], axis=0) #compute the centroid without h
    s = np.argmax(y[y_idx_cent]) #index of the second highest cost
    y_s = func(simplex[s,:][:,np.newaxis].T, args)

    #reflection
    P_r = P_cent + alfa*(P_cent-simplex[h,:])
    y_r = func(P_r[:,np.newaxis].T, args)

    #expansion
    if y_r < yl:
        P_e = P_cent + beta*(P_r-P_cent)
        y_e = func(P_e[:,np.newaxis].T, args)
        if y_e < y_r:
            simplex[h,:] = P_e
        elif y_e >= y_r:
            simplex[h,:] = P_r

    #contraction
    elif y_r >= y_s:
        if y_r<y[h]:
            simplex[h,:] = P_r

        P_c = P_cent + gamma*(simplex[h,:]-P_cent)
        y_c = func(P_c[:, np.newaxis].T, args)

        if y_c > yh:
            simplex[y_idx_cent,:] = 0.5*(simplex[y_idx_cent,:]+simplex[l,:])
            y = func(simplex, args)

        elif y_c <= yh:
            simplex[h,:] = P_c
    else:
        simplex[h,:] = P_r

    curr_opt_point = np.argmin(y)
    costs.append(y[curr_opt_point])

    #break
    if iter == max_iter:
        break

opt_point = np.argmin(y) #optimal point at the minimal value of the cost
params = simplex[opt_point,:]
fit_params = circuit_impedance(params, [2*np.pi*spec_ice_obj.freqs, scaling]) #compute the circuit real output for the optimal values
params *= scaling
fit_params_SIMPLEX = fit_params.astype("complex")
nmse_simplex = NMSE(fit_obj.z_meas.astype("complex"), fit_params.astype("complex"))

init_guess_BFGS = np.array([1, 1, 1, 1, 1, 1, 0.5, 1])
scaling_array_BFGS =  np.array([1e3, 1e-7, 1e6, 1e-2, 1e4, 1e-1, 1, 1])

init_guess_NLLS = np.array([1, 1, 1, 1, 1, 1, 0.5, 1])
scaling_array_NLLS =  np.array([1e3, 1e-7, 1e6, 1e-2, 1e4, 1e-1, 1, 1])

#Impedance fitting w/ BFGS
fit_params_BFGS = fit_obj.fit_circuit(init_guess_BFGS, scaling_array_BFGS, method="BFGS", verbose=True)
fit_params_NLLS = fit_obj.fit_circuit(init_guess_NLLS, scaling_array_NLLS, method="NLLS", verbose=True)

#plot
fig, ax = plt.subplots()
leg = []
ax.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue", s=20)
leg.append('measured')
ax.plot(fit_params_BFGS.opt_fit.real, -fit_params_BFGS.opt_fit.imag, color="tab:orange")
leg.append('BFGS')
ax.plot(fit_params_NLLS.opt_fit.real, -fit_params_NLLS.opt_fit.imag, color="tab:green")
leg.append('NLLS')
ax.plot(fit_params_SIMPLEX.real, -fit_params_SIMPLEX.imag, color="tab:red")
leg.append('Nelder-Mead Simplex')
x1, x2, y1, y2 = -1000, 10000, 1000, 12000
axins = ax.inset_axes([0.5, 0.18, 0.4, 0.4],
                      xlim=(x1, x2), ylim=(y1, y2))
axins.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue")
axins.plot(fit_params_BFGS.opt_fit.real, -fit_params_BFGS.opt_fit.imag, color="tab:orange")
axins.plot(fit_params_NLLS.opt_fit.real, -fit_params_NLLS.opt_fit.imag, color="tab:green")
axins.plot(fit_params_SIMPLEX.real, -fit_params_SIMPLEX.imag, color="tab:red")
ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5)
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.legend(leg)
plt.grid()
plt.show()

plt.figure()
plt.plot(np.arange(0, iter, 1), costs)
plt.grid()
plt.show()