from framework import file_csv, characterization_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#PHOBOS spectroscopy acquisition
spec_air_obj = file_csv.read('../data/testICE_10_12_25/c0.csv', 3, sweeptype="spectrum", aggregate=np.mean)
spec_ice_obj = file_csv.read('../data/testICE_10_12_25/cice.csv', 3, sweeptype="spectrum", aggregate=np.mean)

#dielectric parameters
exp_eps_real, exp_eps_imag = characterization_utils.dielectric_params_corrected(spec_ice_obj, spec_air_obj, spec_ice_obj.freqs) #compute the spectrum based on the experimental data
ideal_eps_real, ideal_eps_imag = characterization_utils.dielectric_params_Artemov2013(spec_ice_obj.freqs, medium="ice")

plt.figure(1)
leg = []
plt.plot(np.log10(spec_ice_obj.freqs), exp_eps_real)
leg.append("ε' meas.")
plt.plot(np.log10(spec_ice_obj.freqs), exp_eps_imag)
leg.append("ε'' meas.")
plt.plot(np.log10(spec_ice_obj.freqs), ideal_eps_real, linestyle='dotted', color='black')
leg.append("ε' Artemov")
plt.plot(np.log10(spec_ice_obj.freqs), ideal_eps_imag, linestyle='dashed', color='black')
leg.append("ε'' Artemov")
plt.xlabel("log(frequency)")
plt.ylabel("ε', ε'' (ice)")
plt.legend(leg)
plt.grid()
plt.show()