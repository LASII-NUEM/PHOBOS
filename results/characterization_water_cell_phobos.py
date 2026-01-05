from framework import file_lcr, characterization_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#PHOBOS spectroscopy acquisition
spec_air_obj = file_lcr.read('../data/testICE_10_12_25/c0.csv', n_samples=3, sweeptype="spectrum", aggregate=np.mean)
spec_h2o_obj = file_lcr.read('../data/testICE_10_12_25/c1.csv', n_samples=3, sweeptype="spectrum", aggregate=np.mean)

#dielectric parameters
exp_eps_real, exp_eps_imag = characterization_utils.dielectric_params_corrected(spec_h2o_obj, spec_air_obj, spec_h2o_obj.freqs) #compute the spectrum based on the experimental data
ideal_eps_real, ideal_eps_imag = characterization_utils.dielectric_params_Artemov2013(spec_h2o_obj.freqs, medium="water")

plt.figure(1)
leg = []
plt.plot(np.log10(spec_h2o_obj.freqs), exp_eps_real)
leg.append("ε' meas.")
plt.plot(np.log10(spec_h2o_obj.freqs), exp_eps_imag)
leg.append("ε'' meas.")
plt.plot(np.log10(spec_h2o_obj.freqs), ideal_eps_real, linestyle='dotted', color='black')
leg.append("ε' Artemov")
plt.plot(np.log10(spec_h2o_obj.freqs), ideal_eps_imag, linestyle='dashed', color='black')
leg.append("ε'' Artemov")
plt.xlabel("log(frequency)")
plt.ylabel("ε', ε'' (water)")
plt.legend(leg)
plt.grid()
plt.show()