from framework import file_lcr, characterization_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#PHOBOS spectroscopy acquisition
spec_air_obj = file_lcr.read('../data/testICE_10_12_25/c0.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_h2o_obj = file_lcr.read('../data/testICE_10_12_25/c1.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)

#dielectric parameters
exp_eps_real, exp_eps_imag = characterization_utils.dielectric_params_corrected(spec_h2o_obj, spec_air_obj, spec_h2o_obj.freqs) #compute the spectrum based on the experimental data
exp_z_real, exp_z_imag = characterization_utils.complex_impedance(spec_h2o_obj, spec_h2o_obj.freqs) #compute the complex impedance based on the experimental data
tan_delta = exp_eps_imag/exp_eps_real #tan_delta = eps''/eps'
exp_sigma_real, exp_sigma_imag = characterization_utils.complex_conductivity(spec_h2o_obj, spec_air_obj, spec_h2o_obj.freqs, eps_func=characterization_utils.dielectric_params_corrected) #conductivity

#Electrode polarization frequency
f_ep = spec_h2o_obj.freqs[np.argmax(tan_delta)] #EP relaxation frequency
f_min_zimag = spec_h2o_obj.freqs[np.argmin(exp_z_imag)] #frequency that separates the bulk and surface effects

fig, ax1 = plt.subplots()
ax1.plot(np.log10(spec_h2o_obj.freqs), exp_eps_real, color="tab:blue")
ax1.set_ylabel("ε'.", color="tab:blue")
ax1.set_xlabel("log(frequency)")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax2 = ax1.twinx()
ax2.plot(np.log10(spec_h2o_obj.freqs), exp_eps_imag, color="tab:orange")
ax2.set_ylabel("ε''", color="tab:orange")
ax2.tick_params(axis='y', labelcolor="tab:orange")
ax1.grid()
fig.tight_layout()
plt.show()

plt.figure()
leg = []
plt.plot(exp_z_real, exp_z_imag)
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.grid()
plt.show()

plt.figure()
leg = []
plt.plot(np.log10(spec_h2o_obj.freqs), tan_delta)
plt.xlabel("log(frequency)")
plt.ylabel("tanδ")
plt.grid()
plt.show()

plt.figure()
fig, ax1 = plt.subplots()
ax1.plot(np.log10(spec_h2o_obj.freqs), exp_sigma_real, color="tab:blue")
ax1.set_ylabel("σ'", color="tab:blue")
ax1.set_xlabel("log(frequency)")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax2 = ax1.twinx()
ax2.plot(np.log10(spec_h2o_obj.freqs), exp_sigma_imag, color="tab:orange")
ax2.set_ylabel("σ''", color="tab:orange")
ax2.tick_params(axis='y', labelcolor="tab:orange")
ax1.grid()
fig.tight_layout()
plt.show()

plt.figure()
leg = []
plt.plot(exp_sigma_real, exp_sigma_imag)
plt.xlabel("σ'")
plt.ylabel("σ''")
plt.grid()
plt.show()