from framework import file_lcr, fitting_utils, characterization_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#PHOBOS spectroscopy acquisition
#spec_ice_obj = file_lcr.read('../data/testICE_30_01_26/cice.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)

#LCR free-run acquisition
spec_ice_obj = file_lcr.read('../data/testICE_30_01_26/c_test.csv', n_samples=1, sweeptype="cell", acquisition_mode="freq", aggregate=np.mean)
z_meas_real, z_meas_imag = characterization_utils.complex_impedance(spec_ice_obj, spec_ice_obj.freqs)
z_meas = z_meas_real-1j*z_meas_imag
freqs_mask = spec_ice_obj.freqs > 100
spec_ice_obj.freqs = spec_ice_obj.freqs[freqs_mask]
fit_obj = fitting_utils.LinearKramersKronig([z_meas_real[10,0,freqs_mask], z_meas_imag[10,0,freqs_mask]], spec_ice_obj.freqs, c=0.5, max_iter=100, add_capacitor=True)
print(f'χ² = {fit_obj.chi_square}')

#plot
fig, ax = plt.subplots()
leg = []
ax.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue")
leg.append('ice measured')
ax.plot(fit_obj.z_hat.real, -fit_obj.z_hat.imag, color="tab:orange")
x1, x2, y1, y2 = -1000, 10000, 1000, 12000
axins = ax.inset_axes([0.5, 0.18, 0.4, 0.4],
                      xlim=(x1, x2), ylim=(y1, y2))
axins.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue")
axins.plot(fit_obj.z_hat.real, -fit_obj.z_hat.imag, color="tab:orange")
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
plt.plot(spec_ice_obj.freqs, fit_obj.fit_residues_imag, '-o')
leg.append('Δ_imag')
plt.ylabel('Δ [%]')
plt.legend(leg)
plt.xlabel("Frequency [Hz]")
plt.xscale('log')
plt.grid()
plt.show()