from framework import file_lcr, characterization_utils
import numpy as np
from impedance.models.circuits import CustomCircuit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#PHOBOS spectroscopy acquisition
spec_air_obj = file_lcr.read('../data/testICE_12_12_25/c0.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_ice_obj = file_lcr.read('../data/testICE_12_12_25/cice.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_h2o_obj = file_lcr.read('../data/testICE_12_12_25/c1.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)

#dielectric parameters
h2o_eps_real, h2o_eps_imag = characterization_utils.dielectric_params_corrected(spec_h2o_obj, spec_air_obj, spec_h2o_obj.freqs) #compute the spectrum based on the experimental data
h2o_z_real, h2o_z_imag = characterization_utils.complex_impedance(spec_h2o_obj, spec_h2o_obj.freqs) #compute the complex impedance based on the experimental data
h2o_z = h2o_z_real - 1j*h2o_z_imag
ice_eps_real, ice_eps_imag = characterization_utils.dielectric_params_corrected(spec_ice_obj, spec_air_obj, spec_ice_obj.freqs) #compute the spectrum based on the experimental data
ice_z_real, ice_z_imag = characterization_utils.complex_impedance(spec_ice_obj, spec_ice_obj.freqs) #compute the complex impedance based on the experimental data
ice_z = ice_z_real - 1j*ice_z_imag

initial_guess = [.1, .005, .1, .9, .005, .1, 200, .1, .9]
circuit = CustomCircuit('R_0-p(R_1,CPE_1)-p(R_2-Wo_1,CPE_2)',
                        initial_guess=initial_guess)
circuit.fit(spec_ice_obj.freqs, ice_z)
opt_fit = circuit.predict(spec_ice_obj.freqs)
print(circuit)

#plot
plt.figure()
leg = []
plt.scatter(ice_z_real, ice_z_imag, marker='o', color="tab:blue")
leg.append('ice measured')
plt.plot(opt_fit.real, -opt_fit.imag, color="tab:orange")
leg.append('ice model (impedance.py)')
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.legend(leg)
plt.grid()
plt.show()

plt.figure()
plt.subplot(1,2,1)
leg = []
plt.plot(np.log10(spec_ice_obj.freqs), np.abs(ice_z))
leg.append('ice measured')
plt.plot(np.log10(spec_ice_obj.freqs), np.abs(opt_fit))
leg.append('ice measured')
plt.ylabel("|Z|")
plt.xlabel("log(Frequency)")
plt.legend(leg)
plt.grid()

plt.subplot(1,2,2)
leg = []
plt.plot(np.log10(spec_ice_obj.freqs), -np.angle(ice_z.astype('complex')))
leg.append('ice measured')
plt.plot(np.log10(spec_ice_obj.freqs), -np.angle(opt_fit))
leg.append('ice measured')
plt.ylabel("-∠Z")
plt.xlabel("log(Frequency)")
plt.legend(leg)
plt.grid()
plt.show()
