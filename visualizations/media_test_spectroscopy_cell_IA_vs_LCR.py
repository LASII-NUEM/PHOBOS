from framework import file_lcr, file_ia, visualization_utils, characterization_utils
import numpy as np

# //----------------------------//

#Impedance Analyzer load acquisition
spec_ia_obj = file_ia.read('../data/test_media_13_01/4294A_DataTransfer_0310.xls')
spec_air_obj = spec_ia_obj["c0"]
spec_tap_obj= spec_ia_obj["tap"]
spec_mineral_obj= spec_ia_obj["mineral"]
spec_distilled_obj= spec_ia_obj["distilled"]
spec_deionized_obj= spec_ia_obj["deionized"]

# //----------------------------//

# PHOBOS spectroscopy load acquisition
lcr_spec_air_obj = file_lcr.read('../data/test_media_12_01/c0.csv', n_samples=3, electrode="cell", acquisition_mode="spectrum", aggregate=np.mean)
lcr_spec_deionized_obj = file_lcr.read('../data/test_media_12_01/deionized.csv', n_samples=3, electrode="cell", acquisition_mode="spectrum", aggregate=np.mean)
lcr_spec_distilled_obj = file_lcr.read('../data/test_media_12_01/distilled.csv', n_samples=3, electrode="cell", acquisition_mode="spectrum", aggregate=np.mean)
lcr_spec_mineral_obj = file_lcr.read('../data/test_media_12_01/mineral.csv', n_samples=3, electrode="cell", acquisition_mode="spectrum", aggregate=np.mean)
lcr_spec_tap_obj = file_lcr.read('../data/test_media_12_01/tap.csv', n_samples=3, electrode="cell", acquisition_mode="spectrum", aggregate=np.mean)

## //----------------------------//


#plots
#OBS: all visualization_utils functions return the computed values if addressed to a variable
#i.e., data = visualization_utils.nyquist(data_medium, freqs, labels=["mineral", "tap"])
#return Z' and Z'' for each input experiment in a dictionary structure

freqs = spec_mineral_obj.freqs #swept frequencies

# Plots of Mineral and Tap water data for IA equip
# data_medium = [spec_mineral_obj, spec_tap_obj] #list with the objects that will be plotted
# visualization_utils.permittivity_by_freq_logx(data_medium, spec_air_obj, freqs,
#                                               eps_func=characterization_utils.dielectric_params_corrected,
#                                               labels=["mineral", "tap"], title="IA results",yaxis_scale=1e5)
#
# visualization_utils.tan_delta_logx(data_medium, spec_air_obj, freqs,
#                                    eps_func=characterization_utils.dielectric_params_corrected,
#                                    labels=["mineral", "tap"],title="IA results")
#
# visualization_utils.nyquist(data_medium, freqs, labels=["mineral", "tap"],title="IA results")
#
# visualization_utils.conductivity_by_freq_logx(data_medium, spec_air_obj, freqs,
#                                               eps_func=characterization_utils.dielectric_params_corrected,
#                                               labels=["mineral", "tap"],title="IA results")
#
# visualization_utils.cole_cole_conductivity(data_medium, spec_air_obj, freqs,
#                                            eps_func=characterization_utils.dielectric_params_corrected,
#                                            labels=["mineral", "tap"],title="IA results")
#
# visualization_utils.cole_cole_permittivity(data_medium, spec_air_obj, freqs,
#                                            eps_func=characterization_utils.dielectric_params_corrected,
#                                            labels=["mineral", "tap"],title="IA results")
#

# data_medium2 = [spec_distilled_obj, spec_deionized_obj] #list with the objects that will be plotted
# visualization_utils.permittivity_by_freq_logx(data_medium2, spec_air_obj, freqs,
#                                               eps_func=characterization_utils.dielectric_params_corrected,
#                                               labels=["distilled", "deionized"], title="IA results",yaxis_scale=1e5)
#
# visualization_utils.tan_delta_logx(data_medium2, spec_air_obj, freqs,
#                                    eps_func=characterization_utils.dielectric_params_corrected,
#                                    labels=["distilled", "deionized"],title="IA results")
#
# visualization_utils.nyquist(data_medium2, freqs, labels=["distilled", "deionized"],title="IA results")
#
# visualization_utils.conductivity_by_freq_logx(data_medium2, spec_air_obj, freqs,
#                                               eps_func=characterization_utils.dielectric_params_corrected,
#                                               labels=["distilled", "deionized"],title="IA results")
#
# visualization_utils.cole_cole_conductivity(data_medium2, spec_air_obj, freqs,
#                                            eps_func=characterization_utils.dielectric_params_corrected,
#                                            labels=["distilled", "deionized"],title="IA results")
#
# visualization_utils.cole_cole_permittivity(data_medium2, spec_air_obj, freqs,
#                                            eps_func=characterization_utils.dielectric_params_corrected,
#                                            labels=["distilled", "deionized"],title="IA results")


# data_medium3 = [spec_mineral_obj, spec_tap_obj,spec_distilled_obj, spec_deionized_obj] #list with the objects that will be plotted
# visualization_utils.permittivity_by_freq_logx(data_medium3, spec_air_obj, freqs,
#                                               eps_func=characterization_utils.dielectric_params_corrected, medium="water", artemov = False,
#                                               labels=["mineral", "tap","distilled", "deionized"], title="IA results",yaxis_scale=1e5)

# visualization_utils.tan_delta_logx(data_medium3, spec_air_obj, freqs,
#                                    eps_func=characterization_utils.dielectric_params_corrected,
#                                    labels=["mineral", "tap","distilled", "deionized"],title="IA results")
#
# visualization_utils.nyquist(data_medium3, freqs, labels=["mineral", "tap","distilled", "deionized"],title="IA results")
#
# visualization_utils.conductivity_by_freq_logx(data_medium3, spec_air_obj, freqs,
#                                               eps_func=characterization_utils.dielectric_params_corrected,
#                                               labels=["mineral", "tap","distilled", "deionized"],title="IA results")
#
# visualization_utils.cole_cole_conductivity(data_medium3, spec_air_obj, freqs,
#                                            eps_func=characterization_utils.dielectric_params_corrected,
#                                            labels=["mineral", "tap","distilled", "deionized"],title="IA results")
#
# visualization_utils.cole_cole_permittivity(data_medium3, spec_air_obj, freqs,
#                                            eps_func=characterization_utils.dielectric_params_corrected,
#                                            labels=["mineral", "tap","distilled", "deionized"],title="IA results")
#
#

data_medium_ia_lcr = [spec_mineral_obj, spec_tap_obj,lcr_spec_mineral_obj, lcr_spec_tap_obj] #list with the objects that will be plotted
visualization_utils.permittivity_by_freq_logx(data_medium_ia_lcr, spec_air_obj, freqs,
                                              eps_func=characterization_utils.dielectric_params_corrected,medium="water", artemov = False,
                                              labels=["IA mineral", "IA tap","LCR mineral", "LCR tap"], title="IA vs LCR results",yaxis_scale=1e5)

# visualization_utils.tan_delta_logx(data_medium_ia_lcr, spec_air_obj, freqs,
#                                    eps_func=characterization_utils.dielectric_params_corrected,
#                                    labels=["IA mineral", "IA tap","LCR mineral", "LCR tap"],title="IA vs LCR results")
#
# visualization_utils.nyquist(data_medium_ia_lcr, freqs, labels=["IA mineral", "IA tap","LCR mineral", "LCR tap"],title="IA vs LCR results")
#
# visualization_utils.conductivity_by_freq_logx(data_medium_ia_lcr, spec_air_obj, freqs,
#                                               eps_func=characterization_utils.dielectric_params_corrected,
#                                               labels=["IA mineral", "IA tap","LCR mineral", "LCR tap"],title="IA vs LCR results")
#
# visualization_utils.cole_cole_conductivity(data_medium_ia_lcr, spec_air_obj, freqs,
#                                            eps_func=characterization_utils.dielectric_params_corrected,
#                                            labels=["IA mineral", "IA tap","LCR mineral", "LCR tap"],title="IA vs LCR results")
#
# visualization_utils.cole_cole_permittivity(data_medium_ia_lcr, spec_air_obj, freqs,
#                                            eps_func=characterization_utils.dielectric_params_corrected,
#                                            labels=["IA mineral", "IA tap","LCR mineral", "LCR tap"],title="IA vs LCR results")


# data_medium_ia_lcr2 = [spec_distilled_obj, spec_deionized_obj,lcr_spec_distilled_obj, lcr_spec_deionized_obj] #list with the objects that will be plotted
# visualization_utils.permittivity_by_freq_logx(data_medium_ia_lcr2, spec_air_obj, freqs,
#                                               eps_func=characterization_utils.dielectric_params_corrected,
#                                               labels=["IA distilled", "IA deionized","LCR distilled", "LCR deionized"], title="IA vs LCR results",yaxis_scale=1e5)
# #
# visualization_utils.tan_delta_logx(data_medium_ia_lcr2, spec_air_obj, freqs,
#                                    eps_func=characterization_utils.dielectric_params_corrected,
#                                     labels=["IA distilled", "IA deionized","LCR distilled", "LCR deionized"],title="IA vs LCR results")
#
# visualization_utils.nyquist(data_medium_ia_lcr2, freqs,  labels=["IA distilled", "IA deionized","LCR distilled", "LCR deionized"],title="IA vs LCR results")
#
# visualization_utils.conductivity_by_freq_logx(data_medium_ia_lcr2, spec_air_obj, freqs,
#                                               eps_func=characterization_utils.dielectric_params_corrected,
#                                                labels=["IA distilled", "IA deionized","LCR distilled", "LCR deionized"],title="IA vs LCR results")
#
# visualization_utils.cole_cole_conductivity(data_medium_ia_lcr2, spec_air_obj, freqs,
#                                            eps_func=characterization_utils.dielectric_params_corrected,
#                                             labels=["IA distilled", "IA deionized","LCR distilled", "LCR deionized"],title="IA vs LCR results")
#
# visualization_utils.cole_cole_permittivity(data_medium_ia_lcr2, spec_air_obj, freqs,
#                                            eps_func=characterization_utils.dielectric_params_corrected,
#                                             labels=["IA distilled", "IA deionized","LCR distilled", "LCR deionized"],title="IA vs LCR results")

