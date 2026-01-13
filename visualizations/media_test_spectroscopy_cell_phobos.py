from framework import file_lcr, visualization_utils, characterization_utils
import numpy as np

#PHOBOS spectroscopy acquisition
spec_air_obj = file_lcr.read('../data/testICE_09_01_26/c0.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_deionized_obj = file_lcr.read('../data/testICE_09_01_26/deionized.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_distilled_obj = file_lcr.read('../data/testICE_09_01_26/distilled.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_mineral_obj = file_lcr.read('../data/testICE_09_01_26/mineral.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_tap_obj = file_lcr.read('../data/testICE_09_01_26/tap.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)


#plots
#OBS: all visualization_utils functions return the computed values if addressed to a variable
#i.e., data = visualization_utils.nyquist(data_medium, freqs, labels=["mineral", "tap"])
#return Z' and Z'' for each input experiment in a dictionary structure

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