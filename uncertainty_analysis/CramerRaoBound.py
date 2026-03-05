from framework import file_lcr, equivalent_circuits, bias_utils, fitting_utils
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
fit_obj = fitting_utils.EquivalentCircuit(model, spec_obj, spec_obj.freqs) #equivalent circuit object

#simulate the circuit with the true parameters
theta = np.array([6200., 7e-7, 192000, 0.005, 1000, 0.45, 0.2, 0.01])
omega = 2*np.pi*spec_obj.freqs
simu_Z = equivalent_circuits.Longo2020(theta, args=[omega, np.ones_like(theta)])
simu_Z = simu_Z.astype("complex")

#Monte Carlo simulations
n_spectra = 500
noise_var = 1e-4
noise_mtx_real = np.random.normal(0, noise_var/np.sqrt(2), size=(n_spectra, len(simu_Z))) #zero-mean uncorrelated Gaussian noise
noise_mtx_imag = np.random.normal(0, noise_var/np.sqrt(2), size=(n_spectra, len(simu_Z))) #zero-mean uncorrelated Gaussian noise
spectra_Z = (simu_Z.real+noise_mtx_real)-1j*(simu_Z.imag+noise_mtx_imag) #generate all Monte Carlo spectra
spectra_Z = spectra_Z.astype("complex")

#fit all the simulated spectra
circuits = {"Longo2020": {"guess": np.array([1, 1, 1, 1, 1, 1, 0.5, 1]),
                          "scale_BFGS": np.array([1e5, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]),
                          "scale_NLLS": np.array([1e5, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]),
                          "scale_DLS": np.array([1e4, 1e-7, 1e6, 1e-2, 1e4, 1e-1, 1, 1]),
                          "scale_SIMPLEX": np.array([1e5, 1e-6, 1e7, 1e-2, 1e4, 1e-1, 1, 1]),
                          "scale_PSO": np.array([1e3, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1])}}
spectra_theta = np.zeros(shape=(n_spectra, len(theta)))
spectra_fit = np.zeros(shape=np.shape(spectra_Z), dtype='complex')

#fitting arguments
init_guess_BFGS = circuits[model]["guess"]
scaling_array_BFGS = circuits[model]["scale_BFGS"]

init_guess_NLLS = circuits[model]["guess"]
scaling_array_NLLS = circuits[model]["scale_NLLS"]

init_guess_DLS = circuits[model]["guess"]
scaling_array_DLS  = circuits[model]["scale_DLS"]

init_guess_simplex = circuits[model]["guess"]
scaling_array_simplex =  circuits[model]["scale_SIMPLEX"]

init_guess_PSO = circuits[model]["guess"]
scaling_array_PSO = circuits[model]["scale_PSO"]

for s in range(0, n_spectra):
    print(f'[MonteCarlo] Spectra {s+1}/{n_spectra}')
    curr_obj = fitting_utils.EquivalentCircuit(model, [spectra_Z.real[s,:], spectra_Z.imag[s,:]], spec_obj.freqs) #equivalent circuit object
    fit_params = curr_obj.fit_circuit(init_guess_BFGS, scaling_array_BFGS, method="BFGS", verbose=False)
    # fit_params = curr_obj.fit_circuit(init_guess_NLLS, scaling_array_NLLS, method="NLLS", verbose=False)
    # fit_params = curr_obj.fit_circuit(init_guess_DLS, scaling_array_DLS, method="DLS", verbose=False)
    #fit_params = curr_obj.fit_circuit(init_guess_simplex, scaling_array_simplex, method="Nelder-Mead", verbose=False)
    # fit_params = curr_obj.fit_circuit(np.zeros(shape=(len(circuits[model]["guess"]),)), scaling_array_PSO, method="PSO", verbose=False)
    spectra_fit[s,:] = fit_params.opt_fit.astype("complex")
    spectra_theta[s,:] = fit_params.opt_params_scaled

spectra_fit = spectra_fit.astype("complex")

#compute the inter-quantile range of each parameter for box plotting
theta_q1 = np.quantile(spectra_theta, 0.25, axis=0) #first quantile
theta_q3 = np.quantile(spectra_theta, 0.75, axis=0) #third quantile
theta_IQR = theta_q3-theta_q1 #inter quantile range
theta_lower_bound = theta_q1-1.5*theta_IQR #upper acceptable bound
theta_upper_bound = theta_q3+1.5*theta_IQR #lower acceptable bound

#compute the average and standard error for each parameter
theta_mean = np.mean(spectra_theta, axis=0)
theta_sse = np.sum((theta-spectra_theta)**2, axis=0) #sum of squared errors
theta_var = (1/(n_spectra*(n_spectra-1)))*theta_sse #variance
theta_std = np.sqrt(theta_var) #standard deviation

#print the stats
params = fitting_utils.function_handlers[model.lower()]["fit_params"]
for i in range(0,len(theta)):
    print(f'[{params[i]}] estimated: {theta_mean[i]} ± {theta_std[i]} / true: {theta[i]}')

#plot the points and the fittings
leg = []
for i in range(0,n_spectra):
    plt.scatter(spectra_Z.real[i,:], spectra_Z.imag[i,:], c='black', s=5)
    plt.plot(spectra_fit.real[i,:], -spectra_fit.imag[i,:], color='red', linewidth=1)
    if i == n_spectra - 1:
        leg.append('Monte Carlo')
        leg.append('Fit')
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.legend(leg)
plt.grid()
plt.show()


