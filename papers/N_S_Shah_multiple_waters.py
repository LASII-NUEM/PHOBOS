from framework import file_lcr, file_ia, characterization_utils, visualization_utils
import numpy as np

#PHOBOS spectroscopy acquisition
spec_air_obj = file_lcr.read('../data/testICE_09_01_26/c0.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_deionized_obj = file_lcr.read('../data/testICE_09_01_26/deionized.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_distilled_obj = file_lcr.read('../data/testICE_09_01_26/distilled.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_mineral_obj = file_lcr.read('../data/testICE_09_01_26/mineral.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_tap_obj = file_lcr.read('../data/testICE_09_01_26/tap.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)

#Electrode polarization frequency
#deionized
deionized_eps_real, deionized_eps_imag = characterization_utils.dielectric_params_corrected(spec_deionized_obj, spec_air_obj, spec_deionized_obj.freqs) #compute the spectrum based on the experimental data
deionized_z_real, deionized_z_imag = characterization_utils.complex_impedance(spec_deionized_obj, spec_deionized_obj.freqs) #compute the complex impedance based on the experimental data
deionized_tan_delta = deionized_eps_imag/deionized_eps_real #tan_delta = eps''/eps'
deionized_f_ep = spec_deionized_obj.freqs[np.argmax(deionized_tan_delta)] #EP relaxation frequency
deionized_f_min_zimag = spec_deionized_obj.freqs[np.argmin(deionized_z_imag)] #frequency that separates the bulk and surface effects

#distilled
distilled_eps_real, distilled_eps_imag = characterization_utils.dielectric_params_corrected(spec_distilled_obj, spec_air_obj, spec_distilled_obj.freqs) #compute the spectrum based on the experimental data
distilled_z_real, distilled_z_imag = characterization_utils.complex_impedance(spec_distilled_obj, spec_distilled_obj.freqs) #compute the complex impedance based on the experimental data
distilled_tan_delta = distilled_eps_imag/distilled_eps_real #tan_delta = eps''/eps'
distilled_f_ep = spec_distilled_obj.freqs[np.argmax(distilled_tan_delta)] #EP relaxation frequency
distilled_f_min_zimag = spec_distilled_obj.freqs[np.argmin(distilled_z_imag)] #frequency that separates the bulk and surface effects

#mineral
mineral_eps_real, mineral_eps_imag = characterization_utils.dielectric_params_corrected(spec_mineral_obj, spec_air_obj, spec_mineral_obj.freqs) #compute the spectrum based on the experimental data
mineral_z_real, mineral_z_imag = characterization_utils.complex_impedance(spec_mineral_obj, spec_mineral_obj.freqs) #compute the complex impedance based on the experimental data
mineral_tan_delta = mineral_eps_imag/mineral_eps_real #tan_delta = eps''/eps'
mineral_f_ep = spec_mineral_obj.freqs[np.argmax(mineral_tan_delta)] #EP relaxation frequency
mineral_f_min_zimag = spec_mineral_obj.freqs[np.argmin(mineral_z_imag)] #frequency that separates the bulk and surface effects

#tap
tap_eps_real, tap_eps_imag = characterization_utils.dielectric_params_corrected(spec_tap_obj, spec_air_obj, spec_tap_obj.freqs) #compute the spectrum based on the experimental data
tap_z_real, tap_z_imag = characterization_utils.complex_impedance(spec_tap_obj, spec_tap_obj.freqs) #compute the complex impedance based on the experimental data
tap_tan_delta = tap_eps_imag/tap_eps_real #tan_delta = eps''/eps'
tap_f_ep = spec_tap_obj.freqs[np.argmax(tap_tan_delta)] #EP relaxation frequency
tap_f_min_zimag = spec_tap_obj.freqs[np.argmin(tap_z_imag)] #frequency that separates the bulk and surface effects

#plots
data_medium = [spec_mineral_obj, spec_tap_obj] #list with the objects that will be plotted
freqs = spec_mineral_obj.freqs #swept frequencies
visualization_utils.permittivity_by_freq_logx(data_medium, spec_air_obj, freqs,
                                              eps_func=characterization_utils.dielectric_params_corrected,
                                              labels=["mineral", "tap"], yaxis_scale=1e5)

visualization_utils.tan_delta_logx(data_medium, spec_air_obj, freqs,
                                   eps_func=characterization_utils.dielectric_params_corrected,
                                   labels=["mineral", "tap"])

visualization_utils.nyquist(data_medium, freqs, labels=["mineral", "tap"])

visualization_utils.conductivity_by_freq_logx(data_medium, spec_air_obj, freqs,
                                              eps_func=characterization_utils.dielectric_params_corrected,
                                              labels=["mineral", "tap"])

visualization_utils.cole_cole_conductivity(data_medium, spec_air_obj, freqs,
                                           eps_func=characterization_utils.dielectric_params_corrected,
                                           labels=["mineral", "tap"])

visualization_utils.cole_cole_permittivity(data_medium, spec_air_obj, freqs,
                                           eps_func=characterization_utils.dielectric_params_corrected,
                                           labels=["mineral", "tap"])


