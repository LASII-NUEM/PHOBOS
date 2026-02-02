import numpy as np
from framework import file_lcr, file_ia, visualization_utils, characterization_utils
import os
import csv
import matplotlib.pyplot as plt

def saveZfile(a, b, c,filename, folder, header=("Freq (Hz)","Z(real)","Z(imag)"), filetype=".txt"):

    a, b, c = map(np.ravel, map(np.asarray, (a, b, c)))
    if not (len(a) == len(b) == len(c)):
        raise ValueError("arrays must have same length")

    os.makedirs(folder, exist_ok=True)

    if filetype == ".csv":
        path = os.path.join(folder, f"{filename}.csv")
        with open(path, "w", newline="", encoding="ascii") as f:
            w = csv.writer(f)
            if header: w.writerow(header)
            w.writerows(zip(a, b, c))
    if filetype == ".txt":
        path = os.path.join(folder, f"{filename}.txt")

        data = np.column_stack((a, b, c))
        np.savetxt(path, data, delimiter=",", comments="", fmt="%.15g")
    return

## //----------------------------//
#LCR bridge load acquisition

LCR_C0_freezer = '../data/freezerVSchiller/LCR_F_19_01/C0.csv'
LCR_C1_freezer = '../data/freezerVSchiller/LCR_F_19_01/C1.csv'
LCR_Cice_freezer = '../data/freezerVSchiller/LCR_F_19_01/C_ice.csv'

LCR_C0_chiller = '../data/freezerVSchiller/LCR_C_30_01/C0.csv'
LCR_C1_chiller = '../data/freezerVSchiller/LCR_C_30_01/C1.csv'
LCR_Cice_chiller = '../data/freezerVSchiller/LCR_C_30_01/C_ice.csv'

LCR_C_C0_obj = file_lcr.read(LCR_C0_chiller, n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
LCR_C_C1_obj = file_lcr.read(LCR_C1_chiller, n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
LCR_C_Cice_obj = file_lcr.read(LCR_Cice_chiller, n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)

LCR_F_C0_obj = file_lcr.read(LCR_C0_freezer, n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
LCR_F_C1_obj = file_lcr.read(LCR_C1_freezer, n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
LCR_F_Cice_obj = file_lcr.read(LCR_Cice_freezer, n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)


# //----------------------------//
#Impedance Analyzer load acquisition

IA_freezer = '../data/freezerVSchiller/IA_F_20_01/Spectrum.xls'
IA_chiller = '../data/freezerVSchiller/IA_C_29_01/Spectrum.xls'

IA_F_obj = file_ia.read(IA_freezer)
IA_F_C0_obj = IA_F_obj["c0"]
IA_F_C1_obj= IA_F_obj["c1"]
IA_F_Cice_obj= IA_F_obj["cice"]

IA_C_obj = file_ia.read(IA_chiller)
IA_C_C0_obj = IA_C_obj["c0"]
IA_C_C1_obj= IA_C_obj["c1"]
IA_C_Cice_obj= IA_C_obj["cice"]


# //----------------------------//
#         Swept frequencies
LCR_freqs = LCR_F_C0_obj.freqs
IA_freqs = IA_F_C0_obj.freqs


# //----------------------------//
#         save Impedance
IA_F_C1_z_real,   IA_F_C1_z_imag = characterization_utils.complex_impedance(IA_F_C1_obj,IA_freqs)
IA_F_Cice_z_real,   IA_F_Cice_z_imag = characterization_utils.complex_impedance(IA_F_Cice_obj,IA_freqs)
IA_C_C1_z_real,   IA_C_C1_z_imag = characterization_utils.complex_impedance(IA_C_C1_obj,IA_freqs)
IA_C_Cice_z_real,   IA_C_Cice_z_imag = characterization_utils.complex_impedance(IA_C_Cice_obj,IA_freqs)

LCR_F_C1_z_real,   LCR_F_C1_z_imag = characterization_utils.complex_impedance(LCR_F_C1_obj,LCR_freqs)
LCR_F_Cice_z_real, LCR_F_Cice_z_imag = characterization_utils.complex_impedance(LCR_F_Cice_obj,LCR_freqs)
LCR_C_C1_z_real,   LCR_C_C1_z_imag = characterization_utils.complex_impedance(LCR_C_C1_obj,LCR_freqs)
LCR_C_Cice_z_real, LCR_C_Cice_z_imag = characterization_utils.complex_impedance(LCR_C_Cice_obj,LCR_freqs)

folderZview_IA = '../data/freezerVSchiller/Zview'
saveZfile(IA_freqs,IA_F_C1_z_real, -IA_F_C1_z_imag, filename = "IA_F_C1", folder = folderZview_IA)
saveZfile(IA_freqs,IA_F_Cice_z_real, -IA_F_Cice_z_real, filename = "IA_F_Cice", folder = folderZview_IA)
saveZfile(IA_freqs,IA_C_C1_z_real, -IA_C_C1_z_real, filename = "IA_C_C1", folder = folderZview_IA)
saveZfile(IA_freqs,IA_C_Cice_z_real, -IA_C_Cice_z_real, filename = "IA_C_Cice", folder = folderZview_IA)

folderZview_LCR = '../data/freezerVSchiller/Zview'
saveZfile(LCR_freqs,LCR_F_C1_z_real, -LCR_F_C1_z_imag, filename = "LCR_F_C1", folder = folderZview_LCR)
saveZfile(LCR_freqs,LCR_F_Cice_z_real, -LCR_F_Cice_z_real, filename = "LCR_F_Cice", folder = folderZview_LCR)
saveZfile(LCR_freqs,LCR_C_C1_z_real, -LCR_C_C1_z_real, filename = "LCR_C_C1", folder = folderZview_LCR)
saveZfile(LCR_freqs,LCR_C_Cice_z_real, -LCR_C_Cice_z_real, filename = "LCR_C_Cice", folder = folderZview_LCR)

# //----------------------------//
#         Visualization

# IA_idx = np.argmin(np.abs(IA_freqs-100))
# # IA_c_C1_z_real = IA_C_C1_z_real[IA_idx:]
# # IA_c_C1_z_imag = IA_C_C1_z_imag[IA_idx:]
# #
# # plt.plot(IA_c_C1_z_real, IA_c_C1_z_imag)
# # plt.grid()
# # plt.show()

# visualization_utils.nyquist(IA_C_C1_obj, IA_freqs, labels=["Water"], title="IA_C_C1")
# visualization_utils.nyquist(IA_C_Cice_obj, IA_freqs, labels=["ICE"], title="IA_C_Cice")
# visualization_utils.nyquist(IA_F_C1_obj, IA_freqs, labels=["Water"],  title="IA_F_C1")
# visualization_utils.nyquist(IA_F_Cice_obj, IA_freqs, labels=["ICE"],  title="IA_F_Cice")
#
# visualization_utils.nyquist(LCR_C_C1_obj, LCR_freqs, labels=["Water"], title="LCR_C_C1")
# visualization_utils.nyquist(LCR_C_Cice_obj, LCR_freqs, labels=["ICE"], title="LCR_C_Cice")
# visualization_utils.nyquist(LCR_F_C1_obj, LCR_freqs, labels=["Water"], title="LCR_F_C1")
# visualization_utils.nyquist(LCR_F_Cice_obj, LCR_freqs, labels=["ICE"], title="LCR_F_Cice")


plotdict = {}
# plotdict["data_IA_F"] = [IA_F_C1_obj, IA_F_Cice_obj]
# plotdict["data_IA_C"] = [IA_C_C1_obj, IA_C_Cice_obj]
# plotdict["data_LCR_F"] = [LCR_F_C1_obj, LCR_F_Cice_obj]
# plotdict["data_LCR_C"] = [LCR_C_C1_obj, LCR_C_Cice_obj]

for key, value in plotdict.items():

    if key == "data_IA_F":
        C0_obj = IA_F_C0_obj
        freqs = IA_freqs
        title = "IA Freezer"

    elif key == "data_IA_C":
        C0_obj = IA_C_C0_obj
        freqs = IA_freqs
        title = "IA Chiller"

    elif key == "data_LCR_F":
        C0_obj = LCR_F_C0_obj
        freqs = LCR_freqs
        title = "LCR Freezer"

    elif key == "data_LCR_C":
        C0_obj = LCR_C_C0_obj
        freqs = LCR_freqs
        title = "LCR Chiller"
    else:
        continue


    visualization_utils.permittivity_by_freq_logx(plotdict[key], C0_obj, freqs,
                                                  eps_func=characterization_utils.dielectric_params_corrected,
                                                  labels=["Water", "ICE"], title=title,yaxis_scale=1e5)

    visualization_utils.tan_delta_logx(plotdict[key], C0_obj, freqs,
                                       eps_func=characterization_utils.dielectric_params_corrected,
                                       labels=["Water", "ICE"], title=title)

    visualization_utils.nyquist(plotdict[key], freqs, labels=["Water", "ICE"], title=title)

    visualization_utils.conductivity_by_freq_logx(plotdict[key], C0_obj, freqs,
                                                  eps_func=characterization_utils.dielectric_params_corrected,
                                                  labels=["Water", "ICE"], title=title)

    visualization_utils.cole_cole_conductivity(plotdict[key], C0_obj, freqs,
                                               eps_func=characterization_utils.dielectric_params_corrected,
                                               labels=["Water", "ICE"], title=title)

    visualization_utils.cole_cole_permittivity(plotdict[key], C0_obj, freqs,
                                           eps_func=characterization_utils.dielectric_params_corrected,
                                           labels=["Water", "ICE"], title=title)

