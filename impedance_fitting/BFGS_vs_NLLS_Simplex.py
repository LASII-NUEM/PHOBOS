from framework import fitting_utils, file_lcr
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#PHOBOS spectroscopy acquisition
spec_obj = file_lcr.read('../data/testICE_30_01_26/c_ice.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
freq_thresh = 100
freq_mask = spec_obj.freqs >= freq_thresh
spec_obj.freqs = spec_obj.freqs[freq_mask]
spec_obj.Cp = spec_obj.Cp[freq_mask]
spec_obj.Rp = spec_obj.Rp[freq_mask]
spec_obj.n_freqs = len(spec_obj.freqs)

#minimizer arguments
init_guess_BFGS = np.array([1, 1, 1, 1, 1, 1, 0.5, 1])
scaling_array_BFGS = np.array([1e3, 1e-7, 1e6, 1e-2, 1e4, 1e-1, 1, 1])

init_guess_NLLS = np.array([1, 1, 1, 1, 1, 1, 0.5, 1])
scaling_array_NLLS = np.array([1e3, 1e-7, 1e6, 1e-2, 1e4, 1e-1, 1, 1])

init_guess_simplex = np.array([1, 1, 1, 1, 1, 1, 0.5, 1])
scaling_array_simplex =  np.array([1e5, 1e-6, 1e7, 1e-2, 1e4, 1e-1, 1, 1])

#Impedance fitting w/ BFGS
fit_obj = fitting_utils.EquivalentCircuit("Longo2020", spec_obj, spec_obj.freqs) #quivalent circuit object
fit_params_BFGS = fit_obj.fit_circuit(init_guess_BFGS, scaling_array_BFGS, method="BFGS", verbose=True)
fit_params_NLLS = fit_obj.fit_circuit(init_guess_NLLS, scaling_array_NLLS, method="NLLS", verbose=True)
fit_params_simplex = fit_obj.fit_circuit(init_guess_simplex, scaling_array_simplex, method="Nelder-Mead", verbose=True)

#plot
fig, ax = plt.subplots()
leg = []
ax.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue", s=20)
leg.append('measured')
ax.plot(fit_params_BFGS.opt_fit.real, -fit_params_BFGS.opt_fit.imag, color="tab:orange")
leg.append('BFGS')
ax.plot(fit_params_NLLS.opt_fit.real, -fit_params_NLLS.opt_fit.imag, color="tab:green")
leg.append('NLLS')
ax.plot(fit_params_simplex.opt_fit.real, -fit_params_simplex.opt_fit.imag, color="tab:red")
leg.append('Nelder-Mead Simplex')
x1, x2, y1, y2 = -1000, 10000, 1000, 12000
axins = ax.inset_axes([0.5, 0.18, 0.4, 0.4],
                      xlim=(x1, x2), ylim=(y1, y2))
axins.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue")
axins.plot(fit_params_BFGS.opt_fit.real, -fit_params_BFGS.opt_fit.imag, color="tab:orange")
axins.plot(fit_params_NLLS.opt_fit.real, -fit_params_NLLS.opt_fit.imag, color="tab:green")
axins.plot(fit_params_simplex.opt_fit.real, -fit_params_simplex.opt_fit.imag, color="tab:red")
ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5)
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.legend(leg)
plt.grid()
plt.show()

plt.figure()
plt.subplot(1,2,1)
leg = []
plt.scatter(spec_obj.freqs, np.abs(fit_obj.z_meas), s=20)
leg.append('measured')
plt.plot(spec_obj.freqs, np.abs(fit_params_BFGS.opt_fit), color="tab:orange")
leg.append('BFGS')
plt.plot(spec_obj.freqs, np.abs(fit_params_NLLS.opt_fit), color="tab:green")
leg.append('NLLS')
plt.plot(spec_obj.freqs, np.abs(fit_params_simplex.opt_fit), color="tab:red")
leg.append('Nelder-Mead Simplex')
plt.ylabel("|Z|")
plt.xlabel("Frequency [Hz]")
plt.xscale('log')
plt.legend(leg)
plt.grid()

plt.subplot(1,2,2)
leg = []
plt.scatter(spec_obj.freqs, np.angle(fit_obj.z_meas.astype('complex')), s=20)
leg.append('measured')
plt.plot(spec_obj.freqs, np.angle(fit_params_BFGS.opt_fit), color="tab:orange")
leg.append('BFGS')
plt.plot(spec_obj.freqs, np.angle(fit_params_NLLS.opt_fit), color="tab:green")
leg.append('NLLS')
plt.plot(spec_obj.freqs, np.angle(fit_params_simplex.opt_fit), color="tab:red")
leg.append('Nelder-Mead Simplex')
plt.ylabel("∠Z [rad]")
plt.xlabel("Frequency [Hz]")
plt.xscale('log')
plt.legend(leg)
plt.grid()
plt.show()

