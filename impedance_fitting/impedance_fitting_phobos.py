from framework import file_lcr, fitting_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

#PHOBOS spectroscopy acquisition
spec_ice_obj = file_lcr.read('../data/testICE_10_12_25/cice.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_h2o_obj = file_lcr.read('../data/testICE_10_12_25/c1.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)

#Impedance fitting
fit_obj = fitting_utils.EquivalentCircuit("Longo2020", spec_ice_obj, spec_ice_obj.freqs) #quivalent circuit object
fit_params = fit_obj.fit_circuit(np.array([1.6, 1, 0.9, 1, 48, 1.5, 1, 2]),
                                 np.array([1e3, 1e-7, 1e6, 1e-2, 1e2, 1e-1, 1, 1]),
                                 method="BFGS")

#plot
plt.figure()
leg = []
plt.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue")
leg.append('ice measured')
plt.plot(fit_params.opt_fit.real, -fit_params.opt_fit.imag, color="tab:orange")
leg.append('Longo2020')
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
plt.show()