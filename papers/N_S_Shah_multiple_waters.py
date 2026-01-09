from framework import file_lcr, characterization_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#PHOBOS spectroscopy acquisition
spec_air_obj = file_lcr.read('../data/testICE_09_01_25/c0.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_deionized_obj = file_lcr.read('../data/testICE_09_01_25/deionized.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_distilled_obj = file_lcr.read('../data/testICE_09_01_25/distilled.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_mineral_obj = file_lcr.read('../data/testICE_09_01_25/mineral.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_tap_obj = file_lcr.read('../data/testICE_09_01_25/tap.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)

#mineral
#dielectric parameters
mineral_eps_real, mineral_eps_imag = characterization_utils.dielectric_params_corrected(spec_mineral_obj, spec_air_obj, spec_mineral_obj.freqs) #compute the spectrum based on the experimental data
mineral_z_real, mineral_z_imag = characterization_utils.complex_impedance(spec_mineral_obj, spec_mineral_obj.freqs) #compute the complex impedance based on the experimental data
mineral_tan_delta = mineral_eps_imag/mineral_eps_real #tan_delta = eps''/eps'
mineral_sigma_real, mineral_sigma_imag = characterization_utils.complex_conductivity(spec_mineral_obj, spec_air_obj, spec_mineral_obj.freqs, eps_func=characterization_utils.dielectric_params_corrected) #conductivity

#Electrode polarization frequency
mineral_f_ep = spec_mineral_obj.freqs[np.argmax(mineral_tan_delta)] #EP relaxation frequency
mineral_f_min_zimag = spec_mineral_obj.freqs[np.argmin(mineral_z_imag)] #frequency that separates the bulk and surface effects

#tap
#dielectric parameters
tap_eps_real, tap_eps_imag = characterization_utils.dielectric_params_corrected(spec_tap_obj, spec_air_obj, spec_tap_obj.freqs) #compute the spectrum based on the experimental data
tap_z_real, tap_z_imag = characterization_utils.complex_impedance(spec_tap_obj, spec_tap_obj.freqs) #compute the complex impedance based on the experimental data
tap_tan_delta = tap_eps_imag/tap_eps_real #tan_delta = eps''/eps'
tap_sigma_real, tap_sigma_imag = characterization_utils.complex_conductivity(spec_tap_obj, spec_air_obj, spec_tap_obj.freqs, eps_func=characterization_utils.dielectric_params_corrected) #conductivity

#Electrode polarization frequency
tap_f_ep = spec_tap_obj.freqs[np.argmax(tap_tan_delta)] #EP relaxation frequency
tap_f_min_zimag = spec_tap_obj.freqs[np.argmin(tap_z_imag)] #frequency that separates the bulk and surface effects

plt.figure()
leg = []
plt.subplot(2,1,1)
plt.plot(np.log10(spec_mineral_obj.freqs), mineral_eps_real/1e5)
leg.append('mineral')
plt.plot(np.log10(spec_tap_obj.freqs), tap_eps_real/1e5)
leg.append('tap')
plt.ylabel("ε' x 10⁵")
plt.legend(leg)
plt.grid()

plt.subplot(2,1,2)
leg = []
plt.plot(np.log10(spec_mineral_obj.freqs), mineral_eps_imag/1e5)
leg.append('mineral')
plt.plot(np.log10(spec_tap_obj.freqs), tap_eps_imag/1e5)
leg.append('tap')
plt.xlabel("log(frequency)")
plt.ylabel("ε'' x 10⁵")
plt.grid()
plt.legend(leg)
plt.tight_layout()
plt.show()

plt.figure()
leg = []
plt.plot(mineral_z_real, mineral_z_imag)
leg.append('mineral')
plt.plot(tap_z_real, tap_z_imag)
leg.append('tap')
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.grid()
plt.legend(leg)
plt.show()

plt.figure()
leg = []
plt.plot(np.log10(spec_mineral_obj.freqs), mineral_tan_delta)
leg.append('mineral')
plt.plot(np.log10(spec_tap_obj.freqs), tap_tan_delta)
leg.append('tap')
plt.xlabel("log(frequency)")
plt.ylabel("tanδ")
plt.grid()
plt.legend(leg)
plt.show()

fig, ax1 = plt.subplots()
ax1.plot(np.log10(spec_mineral_obj.freqs), mineral_sigma_real, label="σ' mineral", color="tab:blue")
ax1.plot(np.log10(spec_tap_obj.freqs), tap_sigma_real, label="σ' tap", color="tab:orange")
ax1.set_ylabel("σ'")
ax1.set_xlabel("log(frequency)")
ax1.tick_params(axis='y')
ax2 = ax1.twinx()
ax2.plot(np.log10(spec_mineral_obj.freqs), mineral_sigma_imag, label="σ'' mineral", linestyle="dotted", color="tab:blue")
ax2.plot(np.log10(spec_tap_obj.freqs), tap_sigma_imag, label="σ'' tap", linestyle="dotted", color="tab:orange")
ax2.set_ylabel("σ''")
ax2.tick_params(axis='y')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
ax1.grid()
fig.tight_layout()
plt.show()

plt.figure()
leg = []
plt.plot(mineral_sigma_real, mineral_sigma_imag)
leg.append('mineral')
plt.plot(tap_sigma_real, tap_sigma_imag)
leg.append('tap')
plt.xlabel("σ'")
plt.ylabel("σ''")
plt.grid()
plt.legend(leg)
plt.tight_layout()
plt.show()

plt.figure()
plt.subplot(1,2,1)
leg = []
plt.plot(np.log10(spec_deionized_obj.freqs), spec_deionized_obj.Cp)
leg.append('deionized')
plt.plot(np.log10(spec_distilled_obj.freqs), spec_distilled_obj.Cp)
leg.append('distilled')
plt.plot(np.log10(spec_mineral_obj.freqs), spec_mineral_obj.Cp)
leg.append('mineral')
plt.plot(np.log10(spec_tap_obj.freqs), spec_tap_obj.Cp)
leg.append('tap')
plt.ylabel("Cp")
plt.xlabel("log(frequency)")
plt.legend(leg)
plt.grid()

plt.subplot(1,2,2)
leg = []
plt.plot(np.log10(spec_deionized_obj.freqs), spec_deionized_obj.Rp)
leg.append('deionized')
plt.plot(np.log10(spec_distilled_obj.freqs), spec_distilled_obj.Rp)
leg.append('distilled')
plt.plot(np.log10(spec_mineral_obj.freqs), spec_mineral_obj.Rp)
leg.append('mineral')
plt.plot(np.log10(spec_tap_obj.freqs), spec_tap_obj.Rp)
leg.append('tap')
plt.ylabel("Rp")
plt.xlabel("log(frequency)")
plt.legend(leg)
plt.grid()
