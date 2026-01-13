from framework import file_lcr, file_ia, characterization_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#PHOBOS spectroscopy acquisition
# __________________________________________________
# LCR data
# spec_air_obj = file_lcr.read('../data/test_media_12_01/c0.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
# spec_mineral_obj = file_lcr.read('../data/test_media_12_01/mineral.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
#
# #dielectric parameters
# exp_eps_real, exp_eps_imag = characterization_utils.dielectric_params_corrected(spec_mineral_obj, spec_air_obj, spec_mineral_obj.freqs) #compute the spectrum based on the experimental data
# exp_z_real, exp_z_imag = characterization_utils.complex_impedance(spec_mineral_obj, spec_mineral_obj.freqs) #compute the complex impedance based on the experimental data
# tan_delta = exp_eps_imag/exp_eps_real #tan_delta = eps''/eps'
# exp_sigma_real, exp_sigma_imag = characterization_utils.complex_conductivity(spec_mineral_obj, spec_air_obj, spec_mineral_obj.freqs, eps_func=characterization_utils.dielectric_params_corrected) #conductivity

# __________________________________________________
# IA data
spec_ia_obj = file_ia.read('../data/test_media_13_01/4294A_DataTransfer_0310.xls')

# for tap
exp_eps_real, exp_eps_imag = characterization_utils.dielectric_params_corrected(spec_ia_obj["tap"], spec_ia_obj["C0"], spec_ia_obj["tap"].freqs) #compute the spectrum based on the experimental data
exp_z_real, exp_z_imag = characterization_utils.complex_impedance(spec_ia_obj["tap"], spec_ia_obj["tap"].freqs) #compute the complex impedance based on the experimental data
tan_delta = exp_eps_imag/exp_eps_real #tan_delta = eps''/eps'
exp_sigma_real, exp_sigma_imag = characterization_utils.complex_conductivity(spec_ia_obj["tap"], spec_ia_obj["C0"], spec_ia_obj["tap"].freqs, eps_func=characterization_utils.dielectric_params_corrected) #conductivity


#Electrode polarization frequency
f_ep = spec_ia_obj["tap"].freqs[np.argmax(tan_delta)] #EP relaxation frequency
f_min_zimag = spec_ia_obj["tap"].freqs[np.argmin(exp_z_imag)] #frequency that separates the bulk and surface effects

fig, ax1 = plt.subplots()
ax1.plot(np.log10(spec_ia_obj["tap"].freqs), exp_eps_real/1e5, color="tab:blue")
ax1.set_ylabel("ε' x 10⁵", color="tab:blue")
ax1.set_xlabel("log(frequency)")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax2 = ax1.twinx()
ax2.plot(np.log10(spec_ia_obj["tap"].freqs), exp_eps_imag/1e5, color="tab:orange")
ax2.set_ylabel("ε'' x 10⁵", color="tab:orange")
ax2.tick_params(axis='y', labelcolor="tab:orange")
ax1.grid()
fig.tight_layout()

plt.figure()
leg = []
plt.plot(exp_z_real, exp_z_imag)
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.grid()

plt.figure()
leg = []
plt.plot(np.log10(spec_ia_obj["tap"].freqs), tan_delta)
plt.xlabel("log(frequency)")
plt.ylabel("tanδ")
plt.grid()


fig, ax1 = plt.subplots()
ax1.plot(np.log10(spec_ia_obj["tap"].freqs), exp_sigma_real, color="tab:blue")
ax1.set_ylabel("σ'", color="tab:blue")
ax1.set_xlabel("log(frequency)")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax2 = ax1.twinx()
ax2.plot(np.log10(spec_ia_obj["tap"].freqs), exp_sigma_imag, color="tab:orange")
ax2.set_ylabel("σ''", color="tab:orange")
ax2.tick_params(axis='y', labelcolor="tab:orange")
ax1.grid()
fig.tight_layout()


plt.figure()
leg = []
plt.plot(exp_sigma_real, exp_sigma_imag)
plt.xlabel("σ'")
plt.ylabel("σ''")
plt.grid()

plt.show()