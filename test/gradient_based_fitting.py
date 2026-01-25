from framework import file_lcr, characterization_utils
import numpy as np
from scipy.optimize import curve_fit, least_squares, minimize
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

def MSE(z_hat, z):
    SSE = np.sum((np.abs(z)-np.abs(z_hat))**2) #sum of squared errors
    return SSE/len(z_hat)

def model_MSE(theta, args):
    z_hat = Bannwart2006(theta, args[1]) #compute the model for the arguments
    SSE = np.sum((np.abs(args[0])-np.abs(z_hat))**2) #sum of squared errors
    return SSE/len(z_hat)

def Bannwart2006(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent circuit
    '''

    #expand thetas into the components with scalling
    R1 = theta[0]*1e3
    tau1 = theta[1]*1e-7
    R2 = theta[2]*1e6
    tau2 = theta[3]*1e-2
    R3 = theta[4]*1e2
    tau3 = theta[5]*1e-1
    n3 = theta[6]*1
    tau4 = theta[7]*1

    omega = 2*np.pi*args #Hz to rad/s
    Z_b1 = R1/(1 + 1j*omega*tau1) #p(R1,C1) block
    Z_b2n = R2 + (R3/(1 + (1j*omega*tau3)**n3)) #num of the p(C2, R2-p(R3, CPE)) block
    Z_b2d = 1 + 1j*omega*tau2 + (1j*omega*tau4)/(1 + (1j*omega*tau3)**n3) #den of the p(C2, R2-p(R3, CPE)) block
    Z_b2 = Z_b2n/Z_b2d #p(C2, R2-p(R3, CPE)) block
    return Z_b1 + Z_b2

#PHOBOS spectroscopy acquisition
spec_air_obj = file_lcr.read('../data/testICE_12_12_25/c0.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_ice_obj = file_lcr.read('../data/testICE_12_12_25/cice.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_h2o_obj = file_lcr.read('../data/testICE_12_12_25/c1.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)

#dielectric parameters
h2o_eps_real, h2o_eps_imag = characterization_utils.dielectric_params_corrected(spec_h2o_obj, spec_air_obj, spec_h2o_obj.freqs) #compute the spectrum based on the experimental data
h2o_z_real, h2o_z_imag = characterization_utils.complex_impedance(spec_h2o_obj, spec_h2o_obj.freqs) #compute the complex impedance based on the experimental data
h2o_z = h2o_z_real - 1j*h2o_z_imag
ice_eps_real, ice_eps_imag = characterization_utils.dielectric_params_corrected(spec_ice_obj, spec_air_obj, spec_ice_obj.freqs) #compute the spectrum based on the experimental data
ice_z_real, ice_z_imag = characterization_utils.complex_impedance(spec_ice_obj, spec_ice_obj.freqs) #compute the complex impedance based on the experimental data
ice_z = ice_z_real - 1j*ice_z_imag

R1_candidates = np.arange(1,5,1)
tau1_candidates = np.arange(1,5,1)
R2_candidates = np.arange(1,5,1)
tau2_candidates = np.arange(1,5,1)
R3_candidates = np.arange(80,110,10)
tau3_candidates = np.arange(1,5,1)
n3_candidates = np.arange(1,5,1)
tau4_candidates = np.arange(1,5,1)
bounds = [(0,np.inf), (0,np.inf), (0,np.inf), (0,np.inf), (0,np.inf), (0,np.inf), (0,1), (0,np.inf)] #set the bounds
opt_MSE = np.inf #variable to store the optimal MSE value
opt_fit = None #object of the optimal circuit
opt_obj = None #object of the optimal minimizer

#non-linear curve fitting
#opt_obj = least_squares(MSE, [2.3e3, 2.7e-7, 9.82e6, 2.81e-2, 36.26e2, 1.16e-1, 0.61, 1.55], args=([ice_z, spec_ice_obj.freqs],))
t_init = time.time()
opt_obj = minimize(model_MSE,[1.5, 1, 0.9, 1, 48, 1.5, 1, 2], args=([ice_z, spec_ice_obj.freqs],) , bounds=bounds, method='L-BFGS-B')
print(f'elapsed = {time.time() - t_init} s')
opt_fit = Bannwart2006(opt_obj.x, spec_ice_obj.freqs)

curr_MSE = MSE(ice_z, opt_fit) #compute the MSE
print('Fitted params: ')
print(f'R1 = {opt_obj.x[0]*1e3}')
print(f'tau1 = {opt_obj.x[1]*1e-7}')
print(f'R2 = {opt_obj.x[2]*1e6}')
print(f'tau2 = {opt_obj.x[3]*1e-2}')
print(f'R3 = {opt_obj.x[4]*1e2}')
print(f'tau3 = {opt_obj.x[5]*1e-1}')
print(f'n3 = {opt_obj.x[6]}')
print(f'tau4 = {opt_obj.x[7]}')

#plot
plt.figure()
leg = []
plt.scatter(ice_z_real, ice_z_imag, marker='o', color="tab:blue")
leg.append('ice measured')
plt.plot(opt_fit.real, -opt_fit.imag, color="tab:orange")
leg.append('ice model (impedance.py)')
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.legend(leg)
plt.grid()
plt.show()

plt.figure()
plt.subplot(1,2,1)
leg = []
plt.scatter(np.log10(spec_ice_obj.freqs), np.abs(ice_z))
leg.append('ice measured')
plt.plot(np.log10(spec_ice_obj.freqs), np.abs(opt_fit), color="tab:orange")
leg.append('ice measured')
plt.ylabel("|Z|")
plt.xlabel("log(Frequency)")
plt.legend(leg)
plt.grid()

plt.subplot(1,2,2)
leg = []
plt.scatter(np.log10(spec_ice_obj.freqs), -np.angle(ice_z.astype('complex')))
leg.append('ice measured')
plt.plot(np.log10(spec_ice_obj.freqs), -np.angle(opt_fit), color="tab:orange")
leg.append('ice measured')
plt.ylabel("-∠Z")
plt.xlabel("log(Frequency)")
plt.legend(leg)
plt.grid()
plt.show()