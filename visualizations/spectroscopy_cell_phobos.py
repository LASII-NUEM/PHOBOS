from framework import file_lcr, visualization_utils, characterization_utils
import numpy as np

#PHOBOS spectroscopy acquisition
spec_air_obj = file_lcr.read('../data/testICE_09_01_26/c0.csv', n_samples=3, electrode="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_deionized_obj = file_lcr.read('../data/testICE_09_01_26/deionized.csv', n_samples=3, electrode="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_distilled_obj = file_lcr.read('../data/testICE_09_01_26/distilled.csv', n_samples=3, electrode="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_mineral_obj = file_lcr.read('../data/testICE_09_01_26/mineral.csv', n_samples=3, electrode="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_tap_obj = file_lcr.read('../data/testICE_09_01_26/tap.csv', n_samples=3, electrode="cell", acquisition_mode="spectrum", aggregate=np.mean)

#plots
#OBS: all visualization_utils functions return the computed values if addressed to a variable
#i.e., data = visualization_utils.nyquist(spec_distilled_obj, freqs, labels="distilled")
#return Z' and Z'' for each input experiment in a dictionary structure

freqs = spec_distilled_obj.freqs #swept frequencies
visualization_utils.permittivity_by_freq_logx(spec_distilled_obj, spec_air_obj, freqs,
                                              eps_func=characterization_utils.dielectric_params_corrected,
                                              title="distilled", yaxis_scale=1e5)

visualization_utils.tan_delta_logx(spec_distilled_obj, spec_air_obj, freqs,
                                   eps_func=characterization_utils.dielectric_params_corrected,
                                   title="distilled")

visualization_utils.nyquist(spec_distilled_obj, freqs, title="distilled")

visualization_utils.conductivity_by_freq_logx(spec_distilled_obj, spec_air_obj, freqs,
                                              eps_func=characterization_utils.dielectric_params_corrected,
                                              title="distilled")

visualization_utils.cole_cole_conductivity(spec_distilled_obj, spec_air_obj, freqs,
                                           eps_func=characterization_utils.dielectric_params_corrected,
                                           title="distilled")

visualization_utils.cole_cole_permittivity(spec_distilled_obj, spec_air_obj, freqs,
                                           eps_func=characterization_utils.dielectric_params_corrected,
                                           title="distilled")