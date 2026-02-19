import pickle
from framework import visualization_utils, characterization_utils
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#load the processed files
#see: ./batch_processing/dump_batch_processing.py to generate the required .pkl file
batch_data_path = f'../data/IceMedia/IA_F_mineral_05_02/IA_F_mineral_05_02.pkl'

if not os.path.isfile(batch_data_path):
    raise FileNotFoundError(f'[load_batch_processing] Batch data file {batch_data_path} not found!')

with open(batch_data_path, 'rb') as handle:
    batch_obj = pickle.load(handle)

#plots
#OBS: all visualization_utils functions return the computed values if addressed to a varPhobos LCRble
#i.e., data = visualization_utils.nyquist(data_medium, freqs, labels=["mineral", "tap"])
#return Z' and Z'' for each input experiment in a dictionary structure

data_medium = batch_obj.ia_obj.spec_h2o_obj.media_obj
spec_air_obj = batch_obj.ia_obj.spec_air_obj
freqs = data_medium.freqs #swept frequencies
visualization_utils.permittivity_by_freq_logx(data_medium, spec_air_obj, freqs,
                                              eps_func=characterization_utils.dielectric_params_corrected,
                                              labels=["water"], title="Phobos LCR results",yaxis_scale=1e5)

visualization_utils.tan_delta_logx(data_medium, spec_air_obj, freqs,
                                   eps_func=characterization_utils.dielectric_params_corrected,
                                   labels=["water"],title="Phobos LCR results")

visualization_utils.nyquist(data_medium, freqs, labels=["water"],title="Phobos LCR results")

visualization_utils.conductivity_by_freq_logx(data_medium, spec_air_obj, freqs,
                                              eps_func=characterization_utils.dielectric_params_corrected,
                                              labels=["water"],title="Phobos LCR results")

visualization_utils.cole_cole_conductivity(data_medium, spec_air_obj, freqs,
                                           eps_func=characterization_utils.dielectric_params_corrected,
                                           labels=["water"],title="Phobos LCR results")

visualization_utils.cole_cole_permittivity(data_medium, spec_air_obj, freqs,
                                           eps_func=characterization_utils.dielectric_params_corrected,
                                           labels=["water"],title="Phobos LCR results")

# #plot temperature vs. capacitance normalized
# fig, ax1 = plt.subplots()
# phobos_obj = batch_obj.ia_obj.freerun_obj
# for freq_idx in range(0, phobos_obj.n_freqs):
#     ax1.plot(phobos_obj.electrode_human_timestamps, phobos_obj.agg_Cp_norm[:,freq_idx], linestyle='dashed')
# ax1.set_xlim([phobos_obj.electrode_human_timestamps[1], phobos_obj.electrode_human_timestamps[-3]])
# ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
# ax1.set_ylabel('Capacitance [-]', color="tab:blue")
# ax1.tick_params(axis='y', labelcolor="tab:blue")
# ax1.grid()
# lines, labels = ax1.get_legend_handles_labels()
# ax1.legend(lines, labels, loc=0)
# fig.tight_layout()
# plt.show()
#
# #plot temperature vs. resistance normalized
# fig, ax1 = plt.subplots()
# for freq_idx in range(0, phobos_obj.n_freqs):
#     ax1.plot(phobos_obj.electrode_human_timestamps, phobos_obj.Rp_norm[:, freq_idx], linestyle='dashed')
# ax1.set_xlim([phobos_obj.electrode_human_timestamps[1], phobos_obj.electrode_human_timestamps[-3]])
# ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
# ax1.set_ylabel('Resistance [-]', color="tab:blue")
# ax1.tick_params(axis='y', labelcolor="tab:blue")
# ax1.grid()
# lines, labels = ax1.get_legend_handles_labels()
# ax1.legend(lines, labels, loc=0)
# fig.tight_layout()
# plt.show()


#plot fittings ice vs. water
fig1, axs1 = plt.subplots(nrows=6, ncols=2, figsize=(10,10))

#longo2020
axs1[0,0].plot(batch_obj.ia_obj.spec_h2o_obj.fit_data["Longo2020"]["z_meas_real"], batch_obj.ia_obj.spec_h2o_obj.fit_data["Longo2020"]["z_meas_imag"], color="tab:blue", label="water measured")
axs1[0,0].plot(batch_obj.ia_obj.spec_h2o_obj.fit_data["Longo2020"]["z_hat_real"][:,0], -batch_obj.ia_obj.spec_h2o_obj.fit_data["Longo2020"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Longo2020 (BFGS)")
axs1[0,0].plot(batch_obj.ia_obj.spec_h2o_obj.fit_data["Longo2020"]["z_hat_real"][:,1], -batch_obj.ia_obj.spec_h2o_obj.fit_data["Longo2020"]["z_hat_imag"][:,1], color="tab:green", linestyle="dashed", label="Longo2020 (NLLS)")
axs1[0,0].legend()
axs1[0,0].grid()
axs1[0,0].set_xlabel("Z'")
axs1[0,0].set_ylabel("Z''")
axs1[0,0].set_ylim([0,300])

axs1[0,1].plot(batch_obj.ia_obj.spec_ice_obj.fit_data["Longo2020"]["z_meas_real"], batch_obj.ia_obj.spec_ice_obj.fit_data["Longo2020"]["z_meas_imag"], color="tab:blue", label="ice measured")
axs1[0,1].plot(batch_obj.ia_obj.spec_ice_obj.fit_data["Longo2020"]["z_hat_real"][:,0], -batch_obj.ia_obj.spec_ice_obj.fit_data["Longo2020"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Longo2020 (BFGS)")
axs1[0,1].plot(batch_obj.ia_obj.spec_ice_obj.fit_data["Longo2020"]["z_hat_real"][:,1], -batch_obj.ia_obj.spec_ice_obj.fit_data["Longo2020"]["z_hat_imag"][:,1], color="tab:green", linestyle="dashed", label="Longo2020 (NLLS)")
axs1[0,1].legend()
axs1[0,1].grid()
axs1[0,1].set_xlabel("Z'")
axs1[0,1].set_ylabel("Z''")

#zurich2021
axs1[1,0].plot(batch_obj.ia_obj.spec_h2o_obj.fit_data["Zurich2021"]["z_meas_real"], batch_obj.ia_obj.spec_h2o_obj.fit_data["Zurich2021"]["z_meas_imag"], color="tab:blue", label="water measured")
axs1[1,0].plot(batch_obj.ia_obj.spec_h2o_obj.fit_data["Zurich2021"]["z_hat_real"][:,0], -batch_obj.ia_obj.spec_h2o_obj.fit_data["Zurich2021"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Zurich2021 (BFGS)")
axs1[1,0].plot(batch_obj.ia_obj.spec_h2o_obj.fit_data["Zurich2021"]["z_hat_real"][:,1], -batch_obj.ia_obj.spec_h2o_obj.fit_data["Zurich2021"]["z_hat_imag"][:,1], color="tab:green", linestyle="dashed", label="Zurich2021 (NLLS)")
axs1[1,0].legend()
axs1[1,0].grid()
axs1[1,0].set_xlabel("Z'")
axs1[1,0].set_ylabel("Z''")
axs1[1,0].set_ylim([0,300])

axs1[1,1].plot(batch_obj.ia_obj.spec_ice_obj.fit_data["Zurich2021"]["z_meas_real"], batch_obj.ia_obj.spec_ice_obj.fit_data["Zurich2021"]["z_meas_imag"], color="tab:blue", label="ice measured")
axs1[1,1].plot(batch_obj.ia_obj.spec_ice_obj.fit_data["Zurich2021"]["z_hat_real"][:,0], -batch_obj.ia_obj.spec_ice_obj.fit_data["Zurich2021"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Zurich2021 (BFGS)")
axs1[1,1].plot(batch_obj.ia_obj.spec_ice_obj.fit_data["Zurich2021"]["z_hat_real"][:,1], -batch_obj.ia_obj.spec_ice_obj.fit_data["Zurich2021"]["z_hat_imag"][:,1], color="tab:green", linestyle="dashed", label="Zurich2021 (NLLS)")
axs1[1,1].legend()
axs1[1,1].grid()
axs1[1,1].set_xlabel("Z'")
axs1[1,1].set_ylabel("Z''")

#zhang2024
axs1[2,0].plot(batch_obj.ia_obj.spec_h2o_obj.fit_data["Zhang2024"]["z_meas_real"], batch_obj.ia_obj.spec_h2o_obj.fit_data["Zhang2024"]["z_meas_imag"], color="tab:blue", label="water measured")
axs1[2,0].plot(batch_obj.ia_obj.spec_h2o_obj.fit_data["Zhang2024"]["z_hat_real"][:,0], -batch_obj.ia_obj.spec_h2o_obj.fit_data["Zhang2024"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Zhang2024 (BFGS)")
axs1[2,0].plot(batch_obj.ia_obj.spec_h2o_obj.fit_data["Zhang2024"]["z_hat_real"][:,1], -batch_obj.ia_obj.spec_h2o_obj.fit_data["Zhang2024"]["z_hat_imag"][:,1], color="tab:green", linestyle="dashed", label="Zhang2024 (NLLS)")
axs1[2,0].legend()
axs1[2,0].grid()
axs1[2,0].set_xlabel("Z'")
axs1[2,0].set_ylabel("Z''")
axs1[2,0].set_ylim([0,300])

axs1[2,1].plot(batch_obj.ia_obj.spec_ice_obj.fit_data["Zhang2024"]["z_meas_real"], batch_obj.ia_obj.spec_ice_obj.fit_data["Zhang2024"]["z_meas_imag"], color="tab:blue", label="ice measured")
axs1[2,1].plot(batch_obj.ia_obj.spec_ice_obj.fit_data["Zhang2024"]["z_hat_real"][:,0], -batch_obj.ia_obj.spec_ice_obj.fit_data["Zhang2024"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Zhang2024 (BFGS)")
axs1[2,1].plot(batch_obj.ia_obj.spec_ice_obj.fit_data["Zhang2024"]["z_hat_real"][:,1], -batch_obj.ia_obj.spec_ice_obj.fit_data["Zhang2024"]["z_hat_imag"][:,1], color="tab:green", linestyle="dashed", label="Zhang2024 (NLLS)")
axs1[2,1].legend()
axs1[2,1].grid()
axs1[2,1].set_xlabel("Z'")
axs1[2,1].set_ylabel("Z''")

#yang2025
axs1[3,0].plot(batch_obj.ia_obj.spec_h2o_obj.fit_data["Yang2025"]["z_meas_real"], batch_obj.ia_obj.spec_h2o_obj.fit_data["Yang2025"]["z_meas_imag"], color="tab:blue", label="water measured")
axs1[3,0].plot(batch_obj.ia_obj.spec_h2o_obj.fit_data["Yang2025"]["z_hat_real"][:,0], -batch_obj.ia_obj.spec_h2o_obj.fit_data["Yang2025"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Yang2025 (BFGS)")
axs1[3,0].plot(batch_obj.ia_obj.spec_h2o_obj.fit_data["Yang2025"]["z_hat_real"][:,1], -batch_obj.ia_obj.spec_h2o_obj.fit_data["Yang2025"]["z_hat_imag"][:,1], color="tab:green", linestyle="dashed", label="Yang2025 (NLLS)")
axs1[3,0].legend()
axs1[3,0].grid()
axs1[3,0].set_xlabel("Z'")
axs1[3,0].set_ylabel("Z''")
axs1[3,0].set_ylim([0,300])

axs1[3,1].plot(batch_obj.ia_obj.spec_ice_obj.fit_data["Yang2025"]["z_meas_real"], batch_obj.ia_obj.spec_ice_obj.fit_data["Yang2025"]["z_meas_imag"], color="tab:blue", label="ice measured")
axs1[3,1].plot(batch_obj.ia_obj.spec_ice_obj.fit_data["Yang2025"]["z_hat_real"][:,0], -batch_obj.ia_obj.spec_ice_obj.fit_data["Yang2025"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Yang2025 (BFGS)")
axs1[3,1].plot(batch_obj.ia_obj.spec_ice_obj.fit_data["Yang2025"]["z_hat_real"][:,1], -batch_obj.ia_obj.spec_ice_obj.fit_data["Yang2025"]["z_hat_imag"][:,1], color="tab:green", linestyle="dashed", label="Yang2025 (NLLS)")
axs1[3,1].legend()
axs1[3,1].grid()
axs1[3,1].set_xlabel("Z'")
axs1[3,1].set_ylabel("Z''")

#Fouquet2005
axs1[4,0].plot(batch_obj.ia_obj.spec_h2o_obj.fit_data["Fouquet2005"]["z_meas_real"], batch_obj.ia_obj.spec_h2o_obj.fit_data["Fouquet2005"]["z_meas_imag"], color="tab:blue", label="water measured")
axs1[4,0].plot(batch_obj.ia_obj.spec_h2o_obj.fit_data["Fouquet2005"]["z_hat_real"][:,0], -batch_obj.ia_obj.spec_h2o_obj.fit_data["Fouquet2005"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Fouquet2005 (BFGS)")
axs1[4,0].plot(batch_obj.ia_obj.spec_h2o_obj.fit_data["Fouquet2005"]["z_hat_real"][:,1], -batch_obj.ia_obj.spec_h2o_obj.fit_data["Fouquet2005"]["z_hat_imag"][:,1], color="tab:green", linestyle="dashed", label="Fouquet2005 (NLLS)")
axs1[4,0].legend()
axs1[4,0].grid()
axs1[4,0].set_xlabel("Z'")
axs1[4,0].set_ylabel("Z''")
axs1[4,0].set_ylim([0,300])

axs1[4,1].plot(batch_obj.ia_obj.spec_ice_obj.fit_data["Fouquet2005"]["z_meas_real"], batch_obj.ia_obj.spec_ice_obj.fit_data["Fouquet2005"]["z_meas_imag"], color="tab:blue", label="ice measured")
axs1[4,1].plot(batch_obj.ia_obj.spec_ice_obj.fit_data["Fouquet2005"]["z_hat_real"][:,0], -batch_obj.ia_obj.spec_ice_obj.fit_data["Fouquet2005"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Fouquet2005 (BFGS)")
axs1[4,1].plot(batch_obj.ia_obj.spec_ice_obj.fit_data["Fouquet2005"]["z_hat_real"][:,1], -batch_obj.ia_obj.spec_ice_obj.fit_data["Fouquet2005"]["z_hat_imag"][:,1], color="tab:green", linestyle="dashed", label="Fouquet2005 (NLLS)")
axs1[4,1].legend()
axs1[4,1].grid()
axs1[4,1].set_xlabel("Z'")
axs1[4,1].set_ylabel("Z''")

#Awayssa2025
axs1[5,0].plot(batch_obj.ia_obj.spec_h2o_obj.fit_data["Awayssa2025"]["z_meas_real"], batch_obj.ia_obj.spec_h2o_obj.fit_data["Awayssa2025"]["z_meas_imag"], color="tab:blue", label="water measured")
axs1[5,0].plot(batch_obj.ia_obj.spec_h2o_obj.fit_data["Awayssa2025"]["z_hat_real"][:,0], -batch_obj.ia_obj.spec_h2o_obj.fit_data["Awayssa2025"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Awayssa2025 (BFGS)")
axs1[5,0].plot(batch_obj.ia_obj.spec_h2o_obj.fit_data["Awayssa2025"]["z_hat_real"][:,1], -batch_obj.ia_obj.spec_h2o_obj.fit_data["Awayssa2025"]["z_hat_imag"][:,1], color="tab:green", linestyle="dashed", label="Awayssa2025 (NLLS)")
axs1[5,0].legend()
axs1[5,0].grid()
axs1[5,0].set_xlabel("Z'")
axs1[5,0].set_ylabel("Z''")
axs1[5,0].set_ylim([0,300])

axs1[5,1].plot(batch_obj.ia_obj.spec_ice_obj.fit_data["Awayssa2025"]["z_meas_real"], batch_obj.ia_obj.spec_ice_obj.fit_data["Awayssa2025"]["z_meas_imag"], color="tab:blue", label="ice measured")
axs1[5,1].plot(batch_obj.ia_obj.spec_ice_obj.fit_data["Awayssa2025"]["z_hat_real"][:,0], -batch_obj.ia_obj.spec_ice_obj.fit_data["Awayssa2025"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Awayssa2025 (BFGS)")
axs1[5,1].plot(batch_obj.ia_obj.spec_ice_obj.fit_data["Awayssa2025"]["z_hat_real"][:,1], -batch_obj.ia_obj.spec_ice_obj.fit_data["Awayssa2025"]["z_hat_imag"][:,1], color="tab:green", linestyle="dashed", label="Awayssa2025 (NLLS)")
axs1[5,1].legend()
axs1[5,1].grid()
axs1[5,1].set_xlabel("Z'")
axs1[5,1].set_ylabel("Z''")
plt.show()