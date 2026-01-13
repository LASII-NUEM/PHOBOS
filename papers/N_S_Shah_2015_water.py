from framework import file_lcr, file_ia, characterization_utils, visualization_utils
import numpy as np

#LCR spectroscopy acquisition
#spec_air_obj = file_lcr.read('../data/testICE_12_12_25/c0.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
#spec_obj = file_lcr.read('../data/testICE_12_12_25/c1.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)

#dielectric parameters
# exp_eps_real, exp_eps_imag = characterization_utils.dielectric_params_corrected(spec_obj, spec_air_obj, spec_obj.freqs) #compute the spectrum based on the experimental data
# exp_z_real, exp_z_imag = characterization_utils.complex_impedance(spec_obj, spec_obj.freqs) #compute the complex impedance based on the experimental data
# tan_delta = exp_eps_imag/exp_eps_real #tan_delta = eps''/eps'
# exp_sigma_real, exp_sigma_imag = characterization_utils.complex_conductivity(spec_obj, spec_air_obj, spec_obj.freqs, eps_func=characterization_utils.dielectric_params_corrected) #conductivity
#
# #Electrode polarization frequency
# f_ep = spec_obj.freqs[np.argmax(tan_delta)] #EP relaxation frequency
# f_min_zimag = spec_obj.freqs[np.argmin(exp_z_imag)] #frequency that separates the bulk and surface effects

#IA spectroscopy acquisition
spec_obj = file_ia.read('../data/testICE_13_01_26/4294A_DataTransfer_0310.xls')

target_medium = "tap"
exp_eps_real, exp_eps_imag = characterization_utils.dielectric_params_corrected(spec_obj[target_medium], spec_obj["C0"], spec_obj[target_medium].freqs) #compute the spectrum based on the experimental data
exp_z_real, exp_z_imag = characterization_utils.complex_impedance(spec_obj[target_medium], spec_obj[target_medium].freqs) #compute the complex impedance based on the experimental data
tan_delta = exp_eps_imag/exp_eps_real #tan_delta = eps''/eps'
exp_sigma_real, exp_sigma_imag = characterization_utils.complex_conductivity(spec_obj[target_medium], spec_obj["C0"], spec_obj[target_medium].freqs, eps_func=characterization_utils.dielectric_params_corrected) #conductivity


#Electrode polarization frequency
f_ep = spec_obj[target_medium].freqs[np.argmax(tan_delta)] #EP relaxation frequency
f_min_zimag = spec_obj[target_medium].freqs[np.argmin(exp_z_imag)] #frequency that separates the bulk and surface effects
spec_air_obj = spec_obj["C0"]
spec_obj = spec_obj[target_medium]

#plots
visualization_utils.permittivity_by_freq_logx(spec_obj, spec_air_obj, spec_obj.freqs,
                                              eps_func=characterization_utils.dielectric_params_corrected,
                                              title="tap", yaxis_scale=1e5)

visualization_utils.tan_delta_logx(spec_obj, spec_air_obj, spec_obj.freqs,
                                   eps_func=characterization_utils.dielectric_params_corrected,
                                   title="tap")

visualization_utils.nyquist(spec_obj, spec_obj.freqs, title="tap")

visualization_utils.conductivity_by_freq_logx(spec_obj, spec_air_obj, spec_obj.freqs,
                                              eps_func=characterization_utils.dielectric_params_corrected,
                                              title="tap")

visualization_utils.cole_cole_conductivity(spec_obj, spec_air_obj, spec_obj.freqs,
                                           eps_func=characterization_utils.dielectric_params_corrected,
                                           title="tap")

visualization_utils.cole_cole_permittivity(spec_obj, spec_air_obj, spec_obj.freqs,
                                           eps_func=characterization_utils.dielectric_params_corrected,
                                           title="tap")