from framework import file_lcr, characterization_utils
import numpy as np
from impedance.validation import linKK
import matplotlib.pyplot as plt

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

LCR_freqs = LCR_C_C0_obj.freqs
LCR_C_C1_z_real,   LCR_C_C1_z_imag = characterization_utils.complex_impedance(LCR_C_C1_obj,LCR_freqs)
LCR_C_Cice_z_real, LCR_C_Cice_z_imag = characterization_utils.complex_impedance(LCR_C_Cice_obj,LCR_freqs)
LCR_F_C1_z_real,   LCR_F_C1_z_imag = characterization_utils.complex_impedance(LCR_F_C1_obj,LCR_freqs)
LCR_F_Cice_z_real, LCR_F_Cice_z_imag = characterization_utils.complex_impedance(LCR_F_Cice_obj,LCR_freqs)

Z_C_C1 = LCR_C_C1_z_real - LCR_C_C1_z_imag*1j
Z_C_Cice = LCR_C_Cice_z_real - LCR_C_Cice_z_imag*1j
Z_F_C1 = LCR_F_C1_z_real - LCR_F_C1_z_imag*1j
Z_F_Cice = LCR_F_Cice_z_real - LCR_F_Cice_z_imag*1j


def kk_validate_impedance_py(f_Hz, Z, c=0.5, max_M=50):
    """
    Runs lin-KK test and returns KK-fitted spectrum + residuals.

    c: 0<c<1, controls tau range vs. f-range (common ~0.7–0.9)
    max_M: maximum number of RC elements to try
    """
    f_Hz = np.asarray(f_Hz, float).ravel()
    Z = np.asarray(Z, complex).ravel()
    # linKK returns:
    # M (chosen number of RC elements), Z_fit (KK-consistent), res_real, res_imag
    M, mu, Z_fit, res_real, res_imag = linKK(f_Hz, Z, c=c, max_M=max_M, fit_type='complex', add_cap=True)

    # handy scalar metrics
    rms_rel_re = np.sqrt(np.mean(res_real**2))
    rms_rel_im = np.sqrt(np.mean(res_imag**2))

    return {
        "M": M,
        "Z_fit": Z_fit,
        "res_real": res_real,
        "res_imag": res_imag,
        "rms_rel_re": rms_rel_re,
        "rms_rel_im": rms_rel_im,
    }

def plotLinkk(linKK, meas, freqs, title):

    Z_meas = np.asarray(meas, complex).ravel()
    Z_fit = np.asarray(linKK["Z_fit"], complex).ravel()
    res_re = np.asarray(linKK["res_real"], float).ravel()
    res_im = np.asarray(linKK["res_imag"], float).ravel()
    # ---- Nyquist ----

    plt.figure(figsize=(7, 6))
    plt.plot(Z_meas.real, -Z_meas.imag, "o", label="Measured")
    plt.plot(Z_fit.real, -Z_fit.imag, "-", label=f"KK fit (M={linKK['M']})")
    plt.xlabel("Z' [Ω]")
    plt.ylabel("-Z'' [Ω]")
    plt.title(title + " - Nyquist")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Residuals ----
    plt.figure(figsize=(8, 5))
    plt.semilogx(freqs, res_re, "o-", label="Residual Re (relative)")
    plt.semilogx(freqs, res_im, "s-", label="Residual Im (relative)")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Relative residual [-]")
    plt.title(title + " - Residuals"
              )
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


linKK_C_C1= kk_validate_impedance_py(LCR_freqs, Z_C_C1, c=0.5, max_M=50)
linKK_C_Cice= kk_validate_impedance_py(LCR_freqs, Z_C_Cice, c=0.5, max_M=50)
linKK_F_C1= kk_validate_impedance_py(LCR_freqs, Z_F_C1, c=0.5, max_M=50)
linKK_F_Cice= kk_validate_impedance_py(LCR_freqs, Z_F_Cice, c=0.5, max_M=50)

plotLinkk(linKK_C_C1, Z_C_C1, LCR_freqs, title = "LCR_C_C1")
plotLinkk(linKK_C_Cice, Z_C_Cice,LCR_freqs, title = "LCR_C_Cice")
plotLinkk(linKK_F_C1, Z_F_C1, LCR_freqs, title = "LCR_F_C1")
plotLinkk(linKK_F_Cice, Z_F_Cice, LCR_freqs, title = "LCR_F_Cice")
