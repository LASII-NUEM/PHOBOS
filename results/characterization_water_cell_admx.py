from framework import file_admx, characterization_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#ADMX2001 spectroscopy acquisition
spec_air_obj = file_admx.read('../data/testICE_16_12_25/c0.csv', sweeptype="cell", acquisition_mode="spectrum")
spec_water_obj = file_admx.read('../data/testICE_16_12_25/c1.csv', sweeptype="cell", acquisition_mode="spectrum")

#dielectric parameters
exp_eps_real, exp_eps_imag = characterization_utils.dielectric_params_corrected(spec_water_obj, spec_air_obj, spec_water_obj.freqs) #compute the spectrum based on the experimental data
ideal_eps_real, ideal_eps_imag = characterization_utils.dielectric_params_Artemov2013(spec_water_obj.freqs, medium="water")

plt.figure(1)
leg = []
plt.plot(np.log10(spec_water_obj.freqs), exp_eps_real)
leg.append("ε' meas.")
plt.plot(np.log10(spec_water_obj.freqs), exp_eps_imag)
leg.append("ε'' meas.")
plt.plot(np.log10(spec_water_obj.freqs), ideal_eps_real, linestyle='dotted', color='black')
leg.append("ε' Artemov")
plt.plot(np.log10(spec_water_obj.freqs), ideal_eps_imag, linestyle='dashed', color='black')
leg.append("ε'' Artemov")
plt.xlabel("log(frequency)")
plt.ylabel("ε', ε'' (water)")
plt.legend(leg)
plt.grid()
plt.show()