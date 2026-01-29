import sklearn.metrics
from scipy.optimize import curve_fit
from sklearn. metrics import r2_score
from framework import fitting_utils, file_lcr
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

def Longo2020_real(omega, R1, tau1, R2, tau2, R3, tau3, n3, tau4):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R||C - C || (R - R||CPE) circuit
    '''

    #expand thetas into the components with scaling
    scaling = np.array([1e3, 1e-7, 1e6, 1e-2, 1e2, 1e-1, 1, 1])
    R1 = R1*scaling[0]
    tau1 = tau1*scaling[1]
    R2 = R2*scaling[2]
    tau2 = tau2*scaling[3]
    R3 = R3*scaling[4]
    tau3 = tau3*scaling[5]
    n3 = n3*scaling[6]
    tau4 = tau4*scaling[7]

    #impedance computation
    Z_b1 = R1/(1+1j*omega*tau1) #p(R1,C1) block
    Z_b2n = R2 + (R3/(1+(1j*omega*tau3)**n3)) #num of the p(C2, R2-p(R3, CPE)) block
    Z_b2d = 1 + 1j*omega*tau2+(1j*omega*tau4)/(1+(1j*omega*tau3)**n3) #den of the p(C2, R2-p(R3, CPE)) block
    Z_b2 = Z_b2n/Z_b2d #p(C2, R2-p(R3, CPE)) block
    Z = Z_b1 + Z_b2
    Z = Z.astype("complex")
    return Z.real

def Longo2020_imag(omega, R1, tau1, R2, tau2, R3, tau3, n3, tau4):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R||C - C || (R - R||CPE) circuit
    '''

    #expand thetas into the components with scaling
    scaling = np.array([1e3, 1e-7, 1e6, 1e-2, 1e2, 1e-1, 1, 1])
    R1 = R1*scaling[0]
    tau1 = tau1*scaling[1]
    R2 = R2*scaling[2]
    tau2 = tau2*scaling[3]
    R3 = R3*scaling[4]
    tau3 = tau3*scaling[5]
    n3 = n3*scaling[6]
    tau4 = tau4*scaling[7]

    #impedance computation
    Z_b1 = R1/(1+1j*omega*tau1) #p(R1,C1) block
    Z_b2n = R2 + (R3/(1+(1j*omega*tau3)**n3)) #num of the p(C2, R2-p(R3, CPE)) block
    Z_b2d = 1 + 1j*omega*tau2+(1j*omega*tau4)/(1+(1j*omega*tau3)**n3) #den of the p(C2, R2-p(R3, CPE)) block
    Z_b2 = Z_b2n/Z_b2d #p(C2, R2-p(R3, CPE)) block
    Z = Z_b1 + Z_b2
    Z = Z.astype("complex")
    return Z.imag

#PHOBOS spectroscopy acquisition
spec_ice_obj = file_lcr.read('../data/testICE_10_12_25/cice.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_h2o_obj = file_lcr.read('../data/testICE_10_12_25/c1.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)

#Impedance fitting w/ BFGS
fit_obj = fitting_utils.EquivalentCircuit("Longo2020", spec_ice_obj, spec_ice_obj.freqs) #quivalent circuit object
fit_params = fit_obj.fit_circuit(np.array([1.6, 1, 0.9, 1, 48, 1.5, 1, 2]),
                                 np.array([1e3, 1e-7, 1e6, 1e-2, 1e2, 1e-1, 1, 1]),
                                 method="BFGS")

#Impedance fitting w/ NLLS
bounds = fitting_utils.function_handlers["longo2020"]["bounds"]
bounds = np.array(bounds)
bounds = ((bounds[:,0]), (bounds[:,1]))
t_init = time.time()
popt_real, pcov_real = curve_fit(Longo2020_real, 2*np.pi*spec_ice_obj.freqs, fit_obj.z_meas_real,
                 p0=np.array([1.6, 0.1, 0.9, 1, 48, 1.5, 1, 2]), bounds=bounds)
nlls_fit_real = Longo2020_real(2*np.pi*spec_ice_obj.freqs, *popt_real)
popt_imag, pcov_imag = curve_fit(Longo2020_imag, 2*np.pi*spec_ice_obj.freqs, fit_obj.z_meas_imag,
                 p0=np.array([1.6, 0.1, 0.9, 1, 48, 1.5, 1, 2]), bounds=bounds)
nlls_fit_imag = Longo2020_imag(2*np.pi*spec_ice_obj.freqs, *popt_real)
nlls_fit = nlls_fit_real + 1j*nlls_fit_imag
nlls_fit = nlls_fit.astype("complex")
print(f'[NLLS] Fit elapsed time: {time.time() - t_init} s')
opt_cost_nlls = np.sum(((fit_obj.z_meas_real-nlls_fit_real)**2)+((fit_obj.z_meas_imag-nlls_fit_imag)**2))
opt_cost_nlls = opt_cost_nlls/len(fit_obj.z_meas)

def CUMSE(z_hat, z):
    SSE = np.sum(((z_hat.real - z.real)**2) + ((z_hat.imag - z.imag)**2)) #sum of squared errors
    SOV = np.sum(z.real**2 + z.imag**2)
    return SSE/SOV
#chisqaure
norm_cum_bfgs = CUMSE(fit_obj.z_meas.astype("complex"), fit_params.opt_fit.astype("complex"))
norm_cum_nlls = CUMSE(fit_obj.z_meas.astype("complex"), nlls_fit.astype("complex"))

#plot
plt.figure()
leg = []
plt.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue")
leg.append('measured')
plt.plot(fit_params.opt_fit.real, -fit_params.opt_fit.imag, color="tab:orange")
leg.append('BFGS')
plt.plot(nlls_fit_real, -nlls_fit_imag, color="tab:green")
leg.append('NLLS')
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.legend(leg)
plt.grid()
plt.show()

plt.figure()
plt.subplot(1,2,1)
leg = []
plt.scatter(np.log10(spec_ice_obj.freqs), np.abs(fit_obj.z_meas))
leg.append('measured')
plt.plot(np.log10(spec_ice_obj.freqs), np.abs(fit_params.opt_fit), color="tab:orange")
leg.append('BFGS')
plt.plot(np.log10(spec_ice_obj.freqs), np.abs(nlls_fit), color="tab:green")
leg.append('NLLS')
plt.ylabel("|Z|")
plt.xlabel("log(Frequency)")
plt.legend(leg)
plt.grid()

plt.subplot(1,2,2)
leg = []
plt.scatter(np.log10(spec_ice_obj.freqs), -np.angle(fit_obj.z_meas.astype('complex')))
leg.append('measured')
plt.plot(np.log10(spec_ice_obj.freqs), -np.angle(fit_params.opt_fit), color="tab:orange")
leg.append('BFGS')
plt.plot(np.log10(spec_ice_obj.freqs), -np.angle(nlls_fit), color="tab:green")
leg.append('NLLS')
plt.ylabel("-∠Z")
plt.xlabel("log(Frequency)")
plt.legend(leg)
plt.grid()
plt.show()

