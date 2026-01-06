from framework import file_lcr, characterization_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#PHOBOS spectroscopy acquisition
spec_air_obj = file_lcr.read('../data/testICE_03_12_25/c0.csv', n_samples=2, sweeptype="flange", acquisition_mode="spectrum", aggregate=np.mean)
spec_h2o_obj = file_lcr.read('../data/testICE_03_12_25/c1.csv', n_samples=2, sweeptype="flange", acquisition_mode="spectrum", aggregate=np.mean)

#define Cp and Rp used to compute the dielectric parameters (i.e., mean Cp/Rp for the first sample at all modes)
spec_air_obj.Cp = np.mean(spec_air_obj.Cp[0,:,:], axis=1)
spec_air_obj.Rp = np.mean(spec_air_obj.Rp[0,:,:], axis=1)
spec_h2o_obj.Cp = np.mean(spec_h2o_obj.Cp[0,:,:], axis=1)
spec_h2o_obj.Rp = np.mean(spec_h2o_obj.Rp[0,:,:], axis=1)

#define Cp and Rp used to compute the dielectric parameters (i.e., Cp/Rp for the first sample at all modes)
# spec_air_obj.Cp = spec_air_obj.Cp[0,:,:]
# spec_air_obj.Rp = spec_air_obj.Rp[0,:,:]
# spec_h2o_obj.Cp = spec_h2o_obj.Cp[0,:,:]
# spec_h2o_obj.Rp = spec_h2o_obj.Rp[0,:,:]

#dielectric parameters
exp_eps_real, exp_eps_imag = characterization_utils.dielectric_params_generic(spec_h2o_obj, spec_air_obj, spec_h2o_obj.freqs) #compute the spectrum based on the experimental data
ideal_eps_real, ideal_eps_imag = characterization_utils.dielectric_params_Artemov2013(spec_h2o_obj.freqs, medium="water")

plt.figure(1)
plt.plot(np.log10(spec_h2o_obj.freqs), exp_eps_real, label="ε' meas.")
plt.plot(np.log10(spec_h2o_obj.freqs), exp_eps_imag, label="ε'' meas.")
plt.plot(np.log10(spec_h2o_obj.freqs), ideal_eps_real, label="ε' Artemov", linestyle='dotted', color='black')
plt.plot(np.log10(spec_h2o_obj.freqs), ideal_eps_imag, label="ε'' Artemov", linestyle='dashed', color='black')
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='best')
plt.xlabel("log(frequency)")
plt.ylabel("ε', ε'' (water)")
plt.grid()
plt.show()