from framework import file_lcr,file_ia, characterization_utils
import numpy as np
import csv
import os

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
        np.savetxt(path, data, delimiter="\t", header="\t".join(header), comments="", fmt="%.15g")
    return

#Impedance Analyzer load acquisition
spec_ia_obj = file_ia.read('../data/test_media_13_01/4294A_DataTransfer_0310.xls')
folderZview_IA = '../data/test_media_13_01/Zview'

IA_spec_air_obj = spec_ia_obj["c0"]
IA_spec_tap_obj= spec_ia_obj["tap"]
IA_spec_mineral_obj= spec_ia_obj["mineral"]
IA_spec_distilled_obj= spec_ia_obj["distilled"]
IA_spec_deionized_obj= spec_ia_obj["deionized"]

IA_freqs = IA_spec_air_obj.freqs

IA_tap_eps_real, IA_tap_eps_imag = characterization_utils.dielectric_params_corrected(IA_spec_tap_obj, IA_spec_air_obj, IA_freqs) #compute the spectrum based on the experimental data
IA_tap_z_real,   IA_tap_z_imag = characterization_utils.complex_impedance(IA_spec_tap_obj,IA_freqs) #compute the complex impedance based on the experimental data

IA_mineral_eps_real, IA_mineral_eps_imag = characterization_utils.dielectric_params_corrected(IA_spec_mineral_obj, IA_spec_air_obj, IA_freqs) #compute the spectrum based on the experimental data
IA_mineral_z_real,   IA_mineral_z_imag = characterization_utils.complex_impedance(IA_spec_mineral_obj, IA_freqs) #compute the complex impedance based on the experimental data

IA_distilled_eps_real, IA_distilled_eps_imag = characterization_utils.dielectric_params_corrected(IA_spec_distilled_obj, IA_spec_air_obj, IA_freqs) #compute the spectrum based on the experimental data
IA_distilled_z_real, IA_distilled_z_imag = characterization_utils.complex_impedance(IA_spec_distilled_obj, IA_freqs) #compute the complex impedance based on the experimental data

IA_deionized_eps_real, IA_deionized_eps_imag = characterization_utils.dielectric_params_corrected(IA_spec_deionized_obj, IA_spec_air_obj, IA_freqs) #compute the spectrum based on the experimental data
IA_deionized_z_real, IA_deionized_z_imag = characterization_utils.complex_impedance(IA_spec_deionized_obj, IA_freqs) #compute the complex impedance based on the experimental data

saveZfile(IA_freqs,IA_tap_z_real, IA_tap_z_imag, filename = "IA_tap", folder = folderZview_IA)
saveZfile(IA_freqs,IA_mineral_z_real, IA_mineral_z_imag, filename = "IA_mineral", folder = folderZview_IA)
saveZfile(IA_freqs,IA_distilled_z_real, IA_distilled_z_imag, filename = "IA_distilled", folder = folderZview_IA)
saveZfile(IA_freqs,IA_deionized_z_real, IA_deionized_z_imag, filename = "IA_deionized", folder = folderZview_IA)

# # LCR spectroscopy acquisition
LCR_spec_air_obj = file_lcr.read('../data/test_media_12_01/c0.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
LCR_spec_tap_obj = file_lcr.read('../data/test_media_12_01/tap.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)

folderZview_LCR = '../data/test_media_12_01/Zview'


LCR_freqs = LCR_spec_air_obj.freqs
#dielectric parameters
LCR_tap_eps_real, LCR_tap_eps_imag = characterization_utils.dielectric_params_corrected(LCR_spec_tap_obj, LCR_spec_air_obj, LCR_freqs) #compute the spectrum based on the experimental data
LCR_tap_z_real, LCR_tap_z_imag = characterization_utils.complex_impedance(LCR_spec_tap_obj, LCR_freqs) #compute the complex impedance based on the experimental data

saveZfile(LCR_freqs,LCR_tap_z_real, LCR_tap_z_imag, filename = "LCR_tap", folder = folderZview_LCR)

