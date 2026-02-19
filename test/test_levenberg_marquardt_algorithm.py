import numpy as np
from framework import file_lcr, fitting_utils
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def generate_Jacobian(cost_fun, theta:np.ndarray, y:np.ndarray, delta_params:np.ndarray, args=()):
    '''
    :param cost_fun:
    :param theta:
    :param y:
    :param delta_params:
    :param args:
    :return: the Jacobian matrix computed by the central finite difference
    '''

    m = len(y) #number of predicted points
    n = len(theta) #number of parameters to be minimized
    J = np.zeros(shape=(m,n)) #Jacobian matrix

    for i in range(n):
        placeholder_theta = theta[i]
        curr_step = delta_params[i]*(1+np.abs(placeholder_theta))
        if curr_step!=0:
            theta[i] = placeholder_theta + curr_step #x[i+h]
            forward_y = cost_fun(theta, args) #evaluate the cost at the i+h sample
            forward_y = forward_y.astype("complex").real
            theta[i] = placeholder_theta - curr_step #x[i-h]
            backwards_y = cost_fun(theta, args) #evaluate the cost at the i-h sample
            backwards_y = backwards_y.astype("complex").real
            J[:,i] = (forward_y - backwards_y)/(2*curr_step) #(x[i+h]-x[i-h])/2h

        theta[i] = placeholder_theta #rebuild the parameters array

    return J

spec_obj = file_lcr.read('../data/testICE_30_01_26/cice.csv', n_samples=3, electrode="cell", acquisition_mode="spectrum", aggregate=np.mean)
freq_thresh = 100
freq_mask = spec_obj.freqs >= freq_thresh
spec_obj.freqs = spec_obj.freqs[freq_mask]
spec_obj.Cp = spec_obj.Cp[freq_mask]
spec_obj.Rp = spec_obj.Rp[freq_mask]
spec_obj.n_freqs = len(spec_obj.freqs)
fit_obj = fitting_utils.EquivalentCircuit("Longo2020", spec_obj, spec_obj.freqs) #quivalent circuit object
theta = np.array([1, 1, 1, 1, 1, 1, 0.5, 1])
scaling = np.array([1e4, 1e-7, 1e6, 1e-2, 1e4, 1e-1, 1, 1])
args=[fit_obj.z_meas, 2*np.pi*spec_obj.freqs, scaling]
bounds=fitting_utils.function_handlers["longo2020"]["bounds"]
max_iter = None
tol=1e-6
cost_func = fitting_utils.function_handlers["longo2020"]["function_ptr"]
damping = 1e5

#ensure proper shapes from the signals
args[0] = args[0].astype("complex")
args[0] = np.atleast_1d(args[0].real).flatten()
args[1] = np.atleast_1d(args[1]).flatten()

#control variables of the algorithm
n_params = len(theta)
n_points = len(args[0])

#handle 'max_iter'
if max_iter is None:
    max_iter = 100*n_params**2

#handle 'bounds'
if bounds is not None:
    if isinstance(bounds, list):
        bounds = np.array(bounds) #convert to numpy array
        lower_bounds = bounds[:,0] #lower bounds for each parameter
        upper_bounds = bounds[:,1] #uppper bounds for each parameter
    else:
        lower_bounds = bounds[:,0] #lower bounds for each parameter
        upper_bounds = bounds[:,1] #uppper bounds for each parameter

#default parameters for the algorithm
W = np.abs(1/(args[0].T @ args[0])) #weighting matrix
W = np.abs(W)*np.ones(n_points)
delta_params = 0.001*np.ones(n_params) #fractional increment of the derivatives

#initialize matrices
y_hat = cost_func(theta, args[1:]) #compute the cost for the current guess
y_hat = y_hat.astype("complex")
y_hat = np.atleast_1d(y_hat.real).flatten()
J = generate_Jacobian(cost_func, theta, y_hat, delta_params, args=args[1:]) #compute the Jacobian
res = args[0].real-y_hat.real #compute the residues
chi_sqr = res.T @ (W*res) #sum of the weighted squared errors
last_chi_sqr = chi_sqr #last compute chi square
H = J.T @ (J*W[:,np.newaxis]) #compute the hessian
b = J.T @ (W*res)  #system solution

#levenber-marquardt algorithm
iter = 0 #counter to monitor iterations
while True:
    iter += 1 #update the iteration number
    A = H + damping*np.diag(np.diag(H)) #(JᵀWJ + λ*diag(JᵀWJ))

    #solve for h
    try:
        h_first_half = np.linalg.inv(A.T @ A)
        h_second_half = A.T @ b
        h = h_first_half @ h_second_half
    except:
        #ensure matrix is well-conditioned
        while np.linalg.cond(A) > 1e15:
            A = A + 1e-6 * np.sum(np.diag(A)) / n_params * np.eye(n_params)
            h_first_half = np.linalg.inv(A.T @ A)
        h_second_half = A.T @ b
        h = h_first_half @ h_second_half

    #update step
    curr_theta = theta + h #update guesses
    curr_theta = np.clip(curr_theta, lower_bounds, upper_bounds) #apply constraints
    y_hat = cost_func(curr_theta, args[1:]) #update the cost for the current guess
    y_hat = y_hat.astype("complex")
    y_hat = np.atleast_1d(y_hat.real).flatten()
    res = args[0].real-y_hat #update the residues
    chi_sqr = res.T @ (W*res) #update the sum of the weighted squared errors

    #check if the iteration will be accepted
    if chi_sqr < last_chi_sqr:
        damping = damping/9
        theta = curr_theta
        last_chi_sqr = chi_sqr
        J = generate_Jacobian(cost_func, theta, y_hat, delta_params, args=args[1:])  # update the Jacobian
        H = J.T @ (J * W[:,np.newaxis]) #compute the hessian
        b = J.T @ (W * res) #system solution

        #stop criteria
        if np.linalg.norm(J*W[:, np.newaxis])<tol:
            break

    else:
        damping = damping*11
        chi_sqr = last_chi_sqr

    if iter == max_iter:
        break

opt_params = theta
y_hat = fitting_utils.function_handlers["longo2020"]["function_ptr"](opt_params, [2*np.pi*spec_obj.freqs, scaling])
y_hat = y_hat.astype("complex")

fig, ax = plt.subplots()
leg = []
ax.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue", s=20)
leg.append('measured')
ax.plot(y_hat.real, -y_hat.imag, color="tab:orange")
leg.append('Nelder-Mead Simplex')
x1, x2, y1, y2 = -1000, 10000, 1000, 12000
axins = ax.inset_axes([0.5, 0.18, 0.4, 0.4],
                      xlim=(x1, x2), ylim=(y1, y2))
axins.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue")
axins.plot(y_hat.real, -y_hat.imag, color="tab:orange")
ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5)
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.legend(leg)
plt.grid()
plt.show()
