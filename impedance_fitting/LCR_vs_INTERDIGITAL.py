from framework import fitting_utils, file_lcr, characterization_utils, data_types
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#PHOBOS spectroscopy acquisition
spec_obj = data_types.PHOBOSData('../data/testICE_10_03_26/c_test.csv', n_samples=1, electrode="cell", acquisition_mode="freq", aggregate=np.mean)
lower_freq_thresh = 40
freq_mask = spec_obj.freqs >= lower_freq_thresh
frame = 5
spec_obj.freqs = spec_obj.freqs[freq_mask]
spec_obj.Cp = spec_obj.Cp[frame,:,freq_mask].flatten()
spec_obj.Rp = spec_obj.Rp[frame,:,freq_mask].flatten()
spec_obj.n_freqs = len(spec_obj.freqs)

#initial arguments
circuits = {"Longo2020": {"guess": np.array([1, 1, 1, 1, 1, 1, 0.5, 1]),
                          "scale_BFGS": np.array([1e5, 1e-5, 1e3, 1e-2, 1e3, 1e-4, 1, 1]),
                          "scale_NLLS": np.array([1e3, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]),
                          "scale_DLS": np.array([1e3, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]),
                          "scale_SIMPLEX": np.array([1e4, 1e-7, 1e5, 1e-2, 1e3, 1e-1, 1, 1]),
                          "scale_PSO": np.array([1e3, 1e-7, 1e5, 1e-2, 1e3, 1e-1, 1, 1])},
            "Zurich2021": {"guess": np.array([1, 1, 0.5, 1, 1, 1]),
                          "scale_BFGS": np.array([1e3, 1e-6, 1, 1e5, 1e3, 1e-8]),
                          "scale_NLLS": np.array([1e5, 1e-8, 1, 1e5, 1e3, 1e-8]),
                          "scale_DLS": np.array([1e5, 1e-8, 1, 1e4, 1e3, 1e-8]),
                          "scale_SIMPLEX": np.array([1e5, 1e-8, 1, 1e5, 1e3, 1e-7]),
                           "scale_PSO":np.array([1e5, 1e-8, 1, 1e6, 1e3, 1e-10])},
            "Zhang2024": {"guess": np.array([1, 0.5, 1, 1, 1, 1]),
                          "scale_BFGS": np.array([1e-9, 1, 1e6, 1e3, 1e2, 1e-8]),
                          "scale_NLLS": np.array([1e-9, 1, 1e6, 1e3, 1e2, 1e-8]),
                          "scale_DLS":  np.array([1e-9, 1, 1e6, 1e3, 1e4, 1e-8]),
                          "scale_SIMPLEX": np.array([1e-7, 1, 1e5, 1e5, 1e4, 1e-8]),
                          "scale_PSO": np.array([1e-9, 1, 1e6, 1e6, 1e3, 1e-6])},
            "Yang2025": {"guess": np.array([1, 1, 1, 0.5, 1, 1]),
                          "scale_BFGS": np.array([1e4, 1e5, 1e-9, 1, 1e4, 1e-8]),
                          "scale_NLLS": np.array([1e4, 1e5, 1e-9, 1, 1e4, 1e-8]),
                          "scale_DLS":  np.array([1e4, 1e4, 1e-9, 1, 1e4, 1e-8]),
                          "scale_SIMPLEX": np.array([1e3, 1e4, 1e-9, 1, 1e4, 1e-8]),
                          "scale_PSO": np.array([1e4, 1e7, 1e-9, 1, 1e3, 1e-6])}} #list of circuits to attempt fitting the data

#minimizer arguments
model = "Longo2020"
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

#Impedance fitting w/ BFGS
Z_real, Z_imag = characterization_utils.complex_impedance(spec_obj, spec_obj.freqs)
fit_obj = fitting_utils.EquivalentCircuit(model, [Z_real, Z_imag], spec_obj.freqs) #equivalent circuit object
fit_params_BFGS = fit_obj.fit_circuit(init_guess_BFGS, scaling_array_BFGS, method="BFGS", verbose=True)
#fit_params_NLLS = fit_obj.fit_circuit(init_guess_NLLS, scaling_array_NLLS, method="NLLS", verbose=True)
#fit_params_DLS = fit_obj.fit_circuit(init_guess_DLS, scaling_array_DLS, method="DLS", verbose=True)
fit_params_simplex = fit_obj.fit_circuit(init_guess_simplex, scaling_array_simplex, method="Nelder-Mead", verbose=True)
#fit_params_PSO = fit_obj.fit_circuit(np.zeros(shape=(len(circuits[model]["guess"]),)), scaling_array_PSO, method="PSO", verbose=True)

#plot
fig, ax = plt.subplots()
leg = []
ax.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue", s=20)
leg.append('measured')
ax.plot(fit_params_BFGS.opt_fit.real, -fit_params_BFGS.opt_fit.imag, color="tab:orange")
leg.append('BFGS')
# ax.plot(fit_params_NLLS.opt_fit.real, -fit_params_NLLS.opt_fit.imag, color="tab:green")
# leg.append('NLLS')
# ax.plot(fit_params_DLS.opt_fit.real, -fit_params_DLS.opt_fit.imag, color="tab:purple")
# leg.append('DLS')
ax.plot(fit_params_simplex.opt_fit.real, -fit_params_simplex.opt_fit.imag, color="tab:red")
leg.append('Nelder-Mead Simplex')
# ax.plot(fit_params_PSO.opt_fit.real, -fit_params_PSO.opt_fit.imag, color="tab:brown")
#leg.append('PSO')
# x1, x2, y1, y2 = -1000, 10000, 1000, 12000
# axins = ax.inset_axes([0.5, 0.18, 0.4, 0.4],
#                       xlim=(x1, x2), ylim=(y1, y2))
# axins.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue")
# axins.plot(fit_params_BFGS.opt_fit.real, -fit_params_BFGS.opt_fit.imag, color="tab:orange")
# axins.plot(fit_params_NLLS.opt_fit.real, -fit_params_NLLS.opt_fit.imag, color="tab:green")
# axins.plot(fit_params_DLS.opt_fit.real, -fit_params_DLS.opt_fit.imag, color="tab:purple")
# axins.plot(fit_params_simplex.opt_fit.real, -fit_params_simplex.opt_fit.imag, color="tab:red")
# axins.plot(fit_params_PSO.opt_fit.real, -fit_params_PSO.opt_fit.imag, color="tab:brown")
# ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5)
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.legend(leg)
plt.grid()
plt.show()

