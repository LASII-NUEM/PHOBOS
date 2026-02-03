from framework import file_lcr, fitting_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#PHOBOS spectroscopy acquisition
spec_ice_obj = file_lcr.read('../data/testICE_12_12_25/cice.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_h2o_obj = file_lcr.read('../data/testICE_12_12_25/c1.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)

#Impedance fitting
fit_obj = fitting_utils.EquivalentCircuit("Zurich2021", spec_ice_obj, spec_ice_obj.freqs) #quivalent circuit object
initial_guess = np.array([1, 1, 1, 1, 1, 1])
scaling_array = np.array([1e4, 1e-7, 1, 1e5, 1e3, 1e-8])
fit_params = fit_obj.fit_circuit(initial_guess, scaling_array, method="BFGS")

#plot
fig, ax = plt.subplots()
leg = []
plt.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue")
ax.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue")
leg.append('ice measured')
plt.plot(fit_params.opt_fit.real, -fit_params.opt_fit.imag, color="tab:orange")
ax.plot(fit_params.opt_fit.real, -fit_params.opt_fit.imag, color="tab:orange")
leg.append('Longo2020')
x1, x2, y1, y2 = -1000, 20000, 1000, 12000
axins = ax.inset_axes([0.5, 0.18, 0.4, 0.4],
                      xlim=(x1, x2), ylim=(y1, y2))
axins.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue")
axins.plot(fit_params.opt_fit.real, -fit_params.opt_fit.imag, color="tab:orange")
#axins.grid()
ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5)
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.legend(leg)
plt.grid()
plt.show()

plt.figure()
plt.subplot(1,2,1)
leg = []
plt.scatter(np.log10(spec_ice_obj.freqs), np.abs(fit_obj.z_meas))
leg.append('ice measured')
plt.plot(np.log10(spec_ice_obj.freqs), np.abs(fit_params.opt_fit), color="tab:orange")
leg.append('Longo2020')
plt.ylabel("|Z|")
plt.xlabel("log(Frequency)")
plt.legend(leg)
plt.grid()

plt.subplot(1,2,2)
leg = []
plt.scatter(np.log10(spec_ice_obj.freqs), -np.angle(fit_obj.z_meas.astype('complex')))
leg.append('ice measured')
plt.plot(np.log10(spec_ice_obj.freqs), -np.angle(fit_params.opt_fit), color="tab:orange")
leg.append('Longo2020')
plt.ylabel("-∠Z")
plt.xlabel("log(Frequency)")
plt.legend(leg)
plt.grid()