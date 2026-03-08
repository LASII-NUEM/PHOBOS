from framework import file_lcr, fitting_utils, bias_utils
import numpy as np
import scipy.stats as stats
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
algorithm = 'BFGS'
circuits = {"Longo2020": {"guess": np.array([1, 1, 1, 1, 1, 1, 0.5, 1]),
                          "scale_BFGS": np.array([1e3, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]),
                          "scale_NLLS": np.array([1e5, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]),
                          "scale_DLS": np.array([1e3, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]),
                          "scale_Nelder-Mead": np.array([1e5, 1e-6, 1e7, 1e-2, 1e4, 1e-1, 1, 1]),
                          "scale_PSO": np.array([1e3, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1])}}
theta = np.array([6200., 7e-7, 1920000, 0.005, 1000, 0.45, 0.2, 0.01]) #real parameters of the simulated circuit
n_spectra = int(1e4) #number of simulations
noise_var = 1e-4 #standard deviation of the Gaussian noise
fit_obj = fitting_utils.EquivalentCircuit(model, spec_obj, spec_obj.freqs) #equivalent circuit object
omega = 2*np.pi*spec_obj.freqs
Z_simu = fitting_utils.function_handlers[model.lower()]["function_ptr"](theta, args=[omega, np.ones_like(theta)])
Z_simu = Z_simu.astype("complex")
mc_obj = bias_utils.MonteCarloFit(model, n_spectra, noise_var, theta, spec_obj.freqs) #Monte Carlo fit object
Z_fit, theta_fit = mc_obj.fit_simulations(circuits[model]["guess"], circuits[model][f"scale_{algorithm}"], method=algorithm, verbose=True, plot=False) #fit the simulations
fit_mean, fit_std = mc_obj.compute_statistics(Z_fit, theta_fit, verbose=True) #compute the fit statistics

# plt.figure(2)
# leg = []
# for i in range(0, n_spectra):
#     plt.plot(Z_fit.real[i, :], -Z_fit.imag[i, :], color='red', linewidth=1, zorder=1)
#     plt.scatter(mc_obj.Z_noise.real[i, :], mc_obj.Z_noise.imag[i, :], c='black', s=8, zorder=2)
#     if i == n_spectra-1:
#         leg.append('Fit')
#         leg.append('Simulated Spectra')
# plt.xlabel("Z'")
# plt.ylabel("Z''")
# plt.legend(leg)
# plt.grid()
# plt.show()

#generate the density plot of standard deviations
n_spectra_cluster = 125
spectra_per_clusters = np.floor(n_spectra/n_spectra_cluster)
clustered_params = np.reshape(theta_fit, (int(n_spectra_cluster), int(spectra_per_clusters), len(theta))) #[clusters, samples, params]
clustered_mean = np.mean(clustered_params, axis=0)

plt.figure(3)
plt.suptitle(algorithm)
for i in range(len(theta)):
    plt.subplot(2,4,i+1)
    count, bins, ignored = plt.hist(clustered_mean[:,i], bins=20, density=False,
                                    color='lightblue', edgecolor='black', alpha=0.7,
                                    label=f'{fitting_utils.function_handlers[model.lower()]["fit_params"][i]}')

    #Gaussian Fit
    mu, std = stats.norm.fit(clustered_mean[:,i])
    x = np.linspace(min(bins), max(bins), 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p * (bins[1] - bins[0]) * n_spectra_cluster, 'b--', linewidth=2, label='MC Gaussian fit')
    ks_stat, p_value = stats.kstest((clustered_mean[:,i]-mu)/std, 'norm')

    plt.title(f'p value = {p_value:.4f}')
    plt.ylabel('#')
    plt.xlabel('Parameter distribution')
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()