import pickle
from framework import visualization_utils, characterization_utils
import os
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#load the processed files
#see: ./batch_processing/dump_batch_processing.py to generate the required .pkl file


def plotfitting(batch_obj_equip, title = None):
    # plot fittings ice vs. water
    fig1, axs1 = plt.subplots(nrows=6, ncols=2, figsize=(10, 10))
    fig1.suptitle(title, fontsize=16)
    # longo2020
    axs1[0, 0].plot(batch_obj_equip.spec_h2o_obj.fit_data["Longo2020"]["z_meas_real"],
                    batch_obj_equip.spec_h2o_obj.fit_data["Longo2020"]["z_meas_imag"], color="tab:blue",
                    label="water measured")
    axs1[0, 0].plot(batch_obj_equip.spec_h2o_obj.fit_data["Longo2020"]["z_hat_real"][:, 0],
                    -batch_obj_equip.spec_h2o_obj.fit_data["Longo2020"]["z_hat_imag"][:, 0], color="tab:orange",
                    linestyle="dotted", label="Longo2020 (BFGS)")
    axs1[0, 0].plot(batch_obj_equip.spec_h2o_obj.fit_data["Longo2020"]["z_hat_real"][:, 1],
                    -batch_obj_equip.spec_h2o_obj.fit_data["Longo2020"]["z_hat_imag"][:, 1], color="tab:green",
                    linestyle="dashed", label="Longo2020 (NLLS)")
    axs1[0, 0].legend()
    axs1[0, 0].grid()
    axs1[0, 0].set_xlabel("Z'")
    axs1[0, 0].set_ylabel("Z''")
    axs1[0, 0].set_ylim([0, 300])

    axs1[0, 1].plot(batch_obj_equip.spec_ice_obj.fit_data["Longo2020"]["z_meas_real"],
                    batch_obj_equip.spec_ice_obj.fit_data["Longo2020"]["z_meas_imag"], color="tab:blue",
                    label="ice measured")
    axs1[0, 1].plot(batch_obj_equip.spec_ice_obj.fit_data["Longo2020"]["z_hat_real"][:, 0],
                    -batch_obj_equip.spec_ice_obj.fit_data["Longo2020"]["z_hat_imag"][:, 0], color="tab:orange",
                    linestyle="dotted", label="Longo2020 (BFGS)")
    axs1[0, 1].plot(batch_obj_equip.spec_ice_obj.fit_data["Longo2020"]["z_hat_real"][:, 1],
                    -batch_obj_equip.spec_ice_obj.fit_data["Longo2020"]["z_hat_imag"][:, 1], color="tab:green",
                    linestyle="dashed", label="Longo2020 (NLLS)")
    axs1[0, 1].legend()
    axs1[0, 1].grid()
    axs1[0, 1].set_xlabel("Z'")
    axs1[0, 1].set_ylabel("Z''")

    # zurich2021
    axs1[1, 0].plot(batch_obj_equip.spec_h2o_obj.fit_data["Zurich2021"]["z_meas_real"],
                    batch_obj_equip.spec_h2o_obj.fit_data["Zurich2021"]["z_meas_imag"], color="tab:blue",
                    label="water measured")
    axs1[1, 0].plot(batch_obj_equip.spec_h2o_obj.fit_data["Zurich2021"]["z_hat_real"][:, 0],
                    -batch_obj_equip.spec_h2o_obj.fit_data["Zurich2021"]["z_hat_imag"][:, 0], color="tab:orange",
                    linestyle="dotted", label="Zurich2021 (BFGS)")
    axs1[1, 0].plot(batch_obj_equip.spec_h2o_obj.fit_data["Zurich2021"]["z_hat_real"][:, 1],
                    -batch_obj_equip.spec_h2o_obj.fit_data["Zurich2021"]["z_hat_imag"][:, 1], color="tab:green",
                    linestyle="dashed", label="Zurich2021 (NLLS)")
    axs1[1, 0].legend()
    axs1[1, 0].grid()
    axs1[1, 0].set_xlabel("Z'")
    axs1[1, 0].set_ylabel("Z''")
    axs1[1, 0].set_ylim([0, 300])

    axs1[1, 1].plot(batch_obj_equip.spec_ice_obj.fit_data["Zurich2021"]["z_meas_real"],
                    batch_obj_equip.spec_ice_obj.fit_data["Zurich2021"]["z_meas_imag"], color="tab:blue",
                    label="ice measured")
    axs1[1, 1].plot(batch_obj_equip.spec_ice_obj.fit_data["Zurich2021"]["z_hat_real"][:, 0],
                    -batch_obj_equip.spec_ice_obj.fit_data["Zurich2021"]["z_hat_imag"][:, 0], color="tab:orange",
                    linestyle="dotted", label="Zurich2021 (BFGS)")
    axs1[1, 1].plot(batch_obj_equip.spec_ice_obj.fit_data["Zurich2021"]["z_hat_real"][:, 1],
                    -batch_obj_equip.spec_ice_obj.fit_data["Zurich2021"]["z_hat_imag"][:, 1], color="tab:green",
                    linestyle="dashed", label="Zurich2021 (NLLS)")
    axs1[1, 1].legend()
    axs1[1, 1].grid()
    axs1[1, 1].set_xlabel("Z'")
    axs1[1, 1].set_ylabel("Z''")

    # zhang2024
    axs1[2, 0].plot(batch_obj_equip.spec_h2o_obj.fit_data["Zhang2024"]["z_meas_real"],
                    batch_obj_equip.spec_h2o_obj.fit_data["Zhang2024"]["z_meas_imag"], color="tab:blue",
                    label="water measured")
    axs1[2, 0].plot(batch_obj_equip.spec_h2o_obj.fit_data["Zhang2024"]["z_hat_real"][:, 0],
                    -batch_obj_equip.spec_h2o_obj.fit_data["Zhang2024"]["z_hat_imag"][:, 0], color="tab:orange",
                    linestyle="dotted", label="Zhang2024 (BFGS)")
    axs1[2, 0].plot(batch_obj_equip.spec_h2o_obj.fit_data["Zhang2024"]["z_hat_real"][:, 1],
                    -batch_obj_equip.spec_h2o_obj.fit_data["Zhang2024"]["z_hat_imag"][:, 1], color="tab:green",
                    linestyle="dashed", label="Zhang2024 (NLLS)")
    axs1[2, 0].legend()
    axs1[2, 0].grid()
    axs1[2, 0].set_xlabel("Z'")
    axs1[2, 0].set_ylabel("Z''")
    axs1[2, 0].set_ylim([0, 300])

    axs1[2, 1].plot(batch_obj_equip.spec_ice_obj.fit_data["Zhang2024"]["z_meas_real"],
                    batch_obj_equip.spec_ice_obj.fit_data["Zhang2024"]["z_meas_imag"], color="tab:blue",
                    label="ice measured")
    axs1[2, 1].plot(batch_obj_equip.spec_ice_obj.fit_data["Zhang2024"]["z_hat_real"][:, 0],
                    -batch_obj_equip.spec_ice_obj.fit_data["Zhang2024"]["z_hat_imag"][:, 0], color="tab:orange",
                    linestyle="dotted", label="Zhang2024 (BFGS)")
    axs1[2, 1].plot(batch_obj_equip.spec_ice_obj.fit_data["Zhang2024"]["z_hat_real"][:, 1],
                    -batch_obj_equip.spec_ice_obj.fit_data["Zhang2024"]["z_hat_imag"][:, 1], color="tab:green",
                    linestyle="dashed", label="Zhang2024 (NLLS)")
    axs1[2, 1].legend()
    axs1[2, 1].grid()
    axs1[2, 1].set_xlabel("Z'")
    axs1[2, 1].set_ylabel("Z''")

    # yang2025
    axs1[3, 0].plot(batch_obj_equip.spec_h2o_obj.fit_data["Yang2025"]["z_meas_real"],
                    batch_obj_equip.spec_h2o_obj.fit_data["Yang2025"]["z_meas_imag"], color="tab:blue",
                    label="water measured")
    axs1[3, 0].plot(batch_obj_equip.spec_h2o_obj.fit_data["Yang2025"]["z_hat_real"][:, 0],
                    -batch_obj_equip.spec_h2o_obj.fit_data["Yang2025"]["z_hat_imag"][:, 0], color="tab:orange",
                    linestyle="dotted", label="Yang2025 (BFGS)")
    axs1[3, 0].plot(batch_obj_equip.spec_h2o_obj.fit_data["Yang2025"]["z_hat_real"][:, 1],
                    -batch_obj_equip.spec_h2o_obj.fit_data["Yang2025"]["z_hat_imag"][:, 1], color="tab:green",
                    linestyle="dashed", label="Yang2025 (NLLS)")
    axs1[3, 0].legend()
    axs1[3, 0].grid()
    axs1[3, 0].set_xlabel("Z'")
    axs1[3, 0].set_ylabel("Z''")
    axs1[3, 0].set_ylim([0, 300])

    axs1[3, 1].plot(batch_obj_equip.spec_ice_obj.fit_data["Yang2025"]["z_meas_real"],
                    batch_obj_equip.spec_ice_obj.fit_data["Yang2025"]["z_meas_imag"], color="tab:blue",
                    label="ice measured")
    axs1[3, 1].plot(batch_obj_equip.spec_ice_obj.fit_data["Yang2025"]["z_hat_real"][:, 0],
                    -batch_obj_equip.spec_ice_obj.fit_data["Yang2025"]["z_hat_imag"][:, 0], color="tab:orange",
                    linestyle="dotted", label="Yang2025 (BFGS)")
    axs1[3, 1].plot(batch_obj_equip.spec_ice_obj.fit_data["Yang2025"]["z_hat_real"][:, 1],
                    -batch_obj_equip.spec_ice_obj.fit_data["Yang2025"]["z_hat_imag"][:, 1], color="tab:green",
                    linestyle="dashed", label="Yang2025 (NLLS)")
    axs1[3, 1].legend()
    axs1[3, 1].grid()
    axs1[3, 1].set_xlabel("Z'")
    axs1[3, 1].set_ylabel("Z''")

    # Fouquet2005
    axs1[4, 0].plot(batch_obj_equip.spec_h2o_obj.fit_data["Fouquet2005"]["z_meas_real"],
                    batch_obj_equip.spec_h2o_obj.fit_data["Fouquet2005"]["z_meas_imag"], color="tab:blue",
                    label="water measured")
    axs1[4, 0].plot(batch_obj_equip.spec_h2o_obj.fit_data["Fouquet2005"]["z_hat_real"][:, 0],
                    -batch_obj_equip.spec_h2o_obj.fit_data["Fouquet2005"]["z_hat_imag"][:, 0], color="tab:orange",
                    linestyle="dotted", label="Fouquet2005 (BFGS)")
    axs1[4, 0].plot(batch_obj_equip.spec_h2o_obj.fit_data["Fouquet2005"]["z_hat_real"][:, 1],
                    -batch_obj_equip.spec_h2o_obj.fit_data["Fouquet2005"]["z_hat_imag"][:, 1], color="tab:green",
                    linestyle="dashed", label="Fouquet2005 (NLLS)")
    axs1[4, 0].legend()
    axs1[4, 0].grid()
    axs1[4, 0].set_xlabel("Z'")
    axs1[4, 0].set_ylabel("Z''")
    axs1[4, 0].set_ylim([0, 300])

    axs1[4, 1].plot(batch_obj_equip.spec_ice_obj.fit_data["Fouquet2005"]["z_meas_real"],
                    batch_obj_equip.spec_ice_obj.fit_data["Fouquet2005"]["z_meas_imag"], color="tab:blue",
                    label="ice measured")
    axs1[4, 1].plot(batch_obj_equip.spec_ice_obj.fit_data["Fouquet2005"]["z_hat_real"][:, 0],
                    -batch_obj_equip.spec_ice_obj.fit_data["Fouquet2005"]["z_hat_imag"][:, 0], color="tab:orange",
                    linestyle="dotted", label="Fouquet2005 (BFGS)")
    axs1[4, 1].plot(batch_obj_equip.spec_ice_obj.fit_data["Fouquet2005"]["z_hat_real"][:, 1],
                    -batch_obj_equip.spec_ice_obj.fit_data["Fouquet2005"]["z_hat_imag"][:, 1], color="tab:green",
                    linestyle="dashed", label="Fouquet2005 (NLLS)")
    axs1[4, 1].legend()
    axs1[4, 1].grid()
    axs1[4, 1].set_xlabel("Z'")
    axs1[4, 1].set_ylabel("Z''")

    # Awayssa2025
    axs1[5, 0].plot(batch_obj_equip.spec_h2o_obj.fit_data["Awayssa2025"]["z_meas_real"],
                    batch_obj_equip.spec_h2o_obj.fit_data["Awayssa2025"]["z_meas_imag"], color="tab:blue",
                    label="water measured")
    axs1[5, 0].plot(batch_obj_equip.spec_h2o_obj.fit_data["Awayssa2025"]["z_hat_real"][:, 0],
                    -batch_obj_equip.spec_h2o_obj.fit_data["Awayssa2025"]["z_hat_imag"][:, 0], color="tab:orange",
                    linestyle="dotted", label="Awayssa2025 (BFGS)")
    axs1[5, 0].plot(batch_obj_equip.spec_h2o_obj.fit_data["Awayssa2025"]["z_hat_real"][:, 1],
                    -batch_obj_equip.spec_h2o_obj.fit_data["Awayssa2025"]["z_hat_imag"][:, 1], color="tab:green",
                    linestyle="dashed", label="Awayssa2025 (NLLS)")
    axs1[5, 0].legend()
    axs1[5, 0].grid()
    axs1[5, 0].set_xlabel("Z'")
    axs1[5, 0].set_ylabel("Z''")
    axs1[5, 0].set_ylim([0, 300])

    axs1[5, 1].plot(batch_obj_equip.spec_ice_obj.fit_data["Awayssa2025"]["z_meas_real"],
                    batch_obj_equip.spec_ice_obj.fit_data["Awayssa2025"]["z_meas_imag"], color="tab:blue",
                    label="ice measured")
    axs1[5, 1].plot(batch_obj_equip.spec_ice_obj.fit_data["Awayssa2025"]["z_hat_real"][:, 0],
                    -batch_obj_equip.spec_ice_obj.fit_data["Awayssa2025"]["z_hat_imag"][:, 0], color="tab:orange",
                    linestyle="dotted", label="Awayssa2025 (BFGS)")
    axs1[5, 1].plot(batch_obj_equip.spec_ice_obj.fit_data["Awayssa2025"]["z_hat_real"][:, 1],
                    -batch_obj_equip.spec_ice_obj.fit_data["Awayssa2025"]["z_hat_imag"][:, 1], color="tab:green",
                    linestyle="dashed", label="Awayssa2025 (NLLS)")
    axs1[5, 1].legend()
    axs1[5, 1].grid()
    axs1[5, 1].set_xlabel("Z'")
    axs1[5, 1].set_ylabel("Z''")
    plt.show()

base_path  = f'../data/IceMedia/'
batch_data_path = []

for root, dirs, files in os.walk(base_path):
    for f in files:
        if f.endswith(".pkl"):
            if not os.path.isfile(os.path.join(root, f)):
                raise FileNotFoundError(f'[load_batch_processing] Batch data file {batch_data_path} not found!')
            batch_data_path.append(os.path.join(root, f))

IA_plotwater={}
IA_plotice = {}
LCR_plotwater={}
LCR_plotice = {}
for batch_file in batch_data_path:
    with open(batch_file, 'rb') as handle:
        batch_obj = pickle.load(handle)

    title = os.path.splitext(os.path.basename(batch_file))[0]

    if title.startswith("IA"):

        batch_obj_equip = batch_obj.ia_obj
        data_medium_water = batch_obj.ia_obj.spec_h2o_obj.media_obj
        data_medium_ice = batch_obj.ia_obj.spec_ice_obj.media_obj
        spec_air_obj = batch_obj.ia_obj.spec_air_obj
        IA_plotwater[title] = data_medium_water
        IA_plotice[title] = data_medium_ice
        IA_plotAir = spec_air_obj
        freqs = data_medium_water.freqs  # swept frequencies
    else:
        batch_obj_equip = batch_obj.lcr_obj
        data_medium_water = batch_obj.lcr_obj.spec_h2o_obj.media_obj
        data_medium_ice = batch_obj.lcr_obj.spec_ice_obj.media_obj
        spec_air_obj = batch_obj.lcr_obj.spec_air_obj
        LCR_plotwater[title] = data_medium_water
        LCR_plotice[title] = data_medium_ice
        LCR_plotAir = spec_air_obj
        freqs = data_medium_water.freqs  # swept frequencies

    # visualization_utils.permittivity_by_freq_logx(data_medium_water, spec_air_obj, freqs,
    #                                               eps_func=characterization_utils.dielectric_params_corrected,
    #                                               labels=["water"], title=title,yaxis_scale=1e5)
    #
    # visualization_utils.tan_delta_logx(data_medium_water, spec_air_obj, freqs,
    #                                    eps_func=characterization_utils.dielectric_params_corrected,
    #                                    labels=["water"],title=title)
    #
    # visualization_utils.nyquist(data_medium_water, freqs, labels=["water"],title=title)
    #
    # visualization_utils.conductivity_by_freq_logx(data_medium_water, spec_air_obj, freqs,
    #                                               eps_func=characterization_utils.dielectric_params_corrected,
    #                                               labels=["water"],title=title)
    #
    # visualization_utils.cole_cole_conductivity(data_medium_water, spec_air_obj, freqs,
    #                                            eps_func=characterization_utils.dielectric_params_corrected,
    #                                            labels=["water"],title=title)
    #
    # visualization_utils.cole_cole_permittivity(data_medium_water, spec_air_obj, freqs,
    #                                            eps_func=characterization_utils.dielectric_params_corrected,
    #                                            labels=["water"],title=title)
    #
    # visualization_utils.permittivity_by_freq_logx(data_medium_ice, spec_air_obj, freqs,
    #                                               eps_func=characterization_utils.dielectric_params_corrected,
    #                                               labels=["ice"], title=title,yaxis_scale=1e5)
    #
    # visualization_utils.tan_delta_logx(data_medium_ice, spec_air_obj, freqs,
    #                                    eps_func=characterization_utils.dielectric_params_corrected,
    #                                    labels=["ice"],title=title)
    #
    # visualization_utils.nyquist(data_medium_ice, freqs, labels=["ice"],title=title)
    #
    # visualization_utils.conductivity_by_freq_logx(data_medium_ice, spec_air_obj, freqs,
    #                                               eps_func=characterization_utils.dielectric_params_corrected,
    #                                               labels=["ice"],title=title)
    #
    # visualization_utils.cole_cole_conductivity(data_medium_ice, spec_air_obj, freqs,
    #                                            eps_func=characterization_utils.dielectric_params_corrected,
    #                                            labels=["ice"],title=title)
    #
    # visualization_utils.cole_cole_permittivity(data_medium_ice, spec_air_obj, freqs,
    #                                            eps_func=characterization_utils.dielectric_params_corrected,
    #                                            labels=["ice"],title=title)
    # plotfitting(batch_obj_equip, title=title)

visualization_utils.permittivity_by_freq_logx(IA_plotwater, IA_plotAir, freqs,
                                              eps_func=characterization_utils.dielectric_params_corrected,
                                              labels=["Distilled", "Mineral", "Salt", "Tap"], title=key, yaxis_scale=1e5)

# visualization_utils.tan_delta_logx(plotdict[key], C0_obj, freqs,
#                                    eps_func=characterization_utils.dielectric_params_corrected,
#                                    labels=["Water", "ICE"], title=title)
#
# visualization_utils.nyquist(plotdict[key], freqs, labels=["Water", "ICE"], title=title)
#
# visualization_utils.conductivity_by_freq_logx(plotdict[key], C0_obj, freqs,
#                                               eps_func=characterization_utils.dielectric_params_corrected,
#                                               labels=["Water", "ICE"], title=title)
#
# visualization_utils.cole_cole_conductivity(plotdict[key], C0_obj, freqs,
#                                            eps_func=characterization_utils.dielectric_params_corrected,
#                                            labels=["Water", "ICE"], title=title)
#
# visualization_utils.cole_cole_permittivity(plotdict[key], C0_obj, freqs,
#                                            eps_func=characterization_utils.dielectric_params_corrected,
#                                            labels=["Water", "ICE"], title=title)