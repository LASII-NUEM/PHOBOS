from framework import file_lcr, fitting_utils, bias_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#PHOBOS spectroscopy acquisition
model = "Longo2020"
spec_obj = file_lcr.read('../data/testICE_30_01_26/c_ice.csv', n_samples=3, electrode="cell", acquisition_mode="spectrum", aggregate=np.mean)
freq_thresh = 100
freq_mask = spec_obj.freqs >= freq_thresh
spec_obj.freqs = spec_obj.freqs[freq_mask]
spec_obj.Cp = spec_obj.Cp[freq_mask]
spec_obj.Rp = spec_obj.Rp[freq_mask]
spec_obj.n_freqs = len(spec_obj.freqs)

#Monte Carlo fittings
circuits = {"Longo2020": {"guess": np.array([1, 1, 1, 1, 1, 1, 0.5, 1]),
                          "scale_BFGS": np.array([1e5, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]),
                          "scale_NLLS": np.array([1e5, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]),
                          "scale_DLS": np.array([1e4, 1e-7, 1e6, 1e-2, 1e4, 1e-1, 1, 1]),
                          "scale_SIMPLEX": np.array([1e4, 1e-6, 1e6, 1e-2, 1e3, 1e-1, 1, 1]),
                          "scale_PSO": np.array([1e3, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1])}}
theta = np.array([6200., 7e-7, 192000, 0.005, 1000, 0.45, 0.2, 0.01]) #real parameters of the simulated circuit
n_spectra = 500 #number of simulations
noise_var = 1e-4 #standard deviation of the Gaussian noise
fit_obj = fitting_utils.EquivalentCircuit(model, spec_obj, spec_obj.freqs) #equivalent circuit object
omega = 2*np.pi*spec_obj.freqs
Z_simu = fitting_utils.function_handlers[model.lower()]["function_ptr"](theta, args=[omega, np.ones_like(theta)])
Z_simu = Z_simu.astype("complex")
mc_obj = bias_utils.MonteCarloFit(model, n_spectra, noise_var, theta, spec_obj.freqs) #Monte Carlo fit object
Z_fit, theta_fit = mc_obj.fit_simulations(circuits[model]["guess"], circuits[model]["scale_BFGS"], method='BFGS', verbose=True, plot=True) #fit the simulations
fit_mean, fit_std = mc_obj.compute_statistics(Z_fit, theta_fit, verbose=True) #compute the fit statistics
