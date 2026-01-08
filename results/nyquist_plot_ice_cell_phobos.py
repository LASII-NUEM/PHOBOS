from framework import file_lcr, characterization_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#PHOBOS spectroscopy acquisition
spec_ice_obj = file_lcr.read('../data/testICE_10_12_25/cice.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)

#dielectric parameters
exp_z_real, exp_z_imag = characterization_utils.complex_impedance(spec_ice_obj, spec_ice_obj.freqs) #compute the spectrum based on the experimental data

plt.figure(1)
leg = []
plt.plot(exp_z_real, exp_z_imag)
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.title('Nyquist plot (ice)')
plt.grid()
plt.show()