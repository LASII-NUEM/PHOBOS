from framework import file_lcr, fitting_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#PHOBOS spectroscopy acquisition
spec_ice_obj = file_lcr.read('../data/testICE_30_01_26/cice.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
fit_obj = fitting_utils.LinearKramersKronig(spec_ice_obj, spec_ice_obj.freqs, c=0.45, max_iter=100, add_capacitor=True)

#plot
fig, ax = plt.subplots()
leg = []
ax.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue")
leg.append('ice measured')
ax.plot(fit_obj.z_hat.real, fit_obj.z_hat.imag, color="tab:orange")
x1, x2, y1, y2 = -1000, 10000, 1000, 12000
axins = ax.inset_axes([0.5, 0.18, 0.4, 0.4],
                      xlim=(x1, x2), ylim=(y1, y2))
axins.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue")
axins.plot(fit_obj.z_hat.real, fit_obj.z_hat.imag, color="tab:orange")
ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5)
leg.append(f'Kramers-Kronig: M = {fit_obj.fit_components}')
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.legend(leg)
plt.grid()
plt.show()

plt.figure()
leg = []
plt.plot(spec_ice_obj.freqs, fit_obj.fit_residues_real, '-o')
leg.append('Δ_re')
plt.plot(spec_ice_obj.freqs, -fit_obj.fit_residues_imag, '-o')
leg.append('Δ_imag')
plt.ylabel('Δ [%]')
plt.legend(leg)
plt.xlabel("Frequency [Hz]")
plt.xscale('log')
plt.grid()
plt.show()