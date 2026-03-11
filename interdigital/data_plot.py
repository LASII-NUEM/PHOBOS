from framework import batch_utils, characterization_utils, file_lcr
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

circuits = {"Longo2020": {"guess": np.array([1, 1, 1, 1, 1, 1, 1, 1]), "ice_scale": np.array([1e5, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1])},
            "Zurich2021": {"guess": np.array([1, 1, 1, 1, 1, 1]), "scale": np.array([1e4, 1e-8, 1, 1e5, 1e3, 1e-8])},
            "Zhang2024": {"guess": np.array([1, 1, 1, 1, 1, 1]), "scale": np.array([1e-9, 1, 1e6, 1e3, 1e2, 1e-8])},
            "Yang2025": {"guess": np.array([1, 1, 1, 1, 1, 1]), "scale": np.array([1e3, 1e4, 1e-9, 1, 1e4, 1e-8])},
            "Fouquet2005": {"guess": np.array([1, 1, 1, 1, 1, 1]), "scale": np.array([1e4, 1e3, 1e-8, 1, 1e4, 1e-4])},
            "Awayssa2025": {"guess": np.array([1, 1, 1, 1, 1]), "scale": np.array([1e3, 1e3, 1e-8, 1e-4, 1e-8])}} #list of circuits to attempt fitting the data

# 'batch_utils' package from framework is responsible for processing the raw data batch output by the PHOBOS acquisition system
# into a custom data structure to facilitate your signal processing routine.
dir_path = '../data/testICE_30_01_26' #relative path to the directory where all acquisition files are stored in the filesystem
data = batch_utils.BatchOrganizer(dir_path, circuits, freq_threshold=None, electrode="cell", hardware = "lcr", aggregate=np.mean, eps_func=characterization_utils.dielectric_params_corrected)
# Parameters:
# circuits: dictionary with initial arguments and scaling for impedance fitting (disregard it for now)
# freq_threshold: value to filter frequencies from (i.e., 100 filters starting from 100 Hz. OBS: None to skip filtering)
# electrode: which hardware cell was used on the test (note: 'cell' is the default for any single pair hardware)
# hardware: which hardware equipment was used on the test
# aggregate: pointer to the function that will be used to aggregate the sweeps per mode
# Learn more about the function reading its documentation at: '../framework/batch_utils.py'!

# Once the batch processing is complete, you can explore the data structures to familiarize yourself with it.
spec_air_obj = data.lcr_obj.spec_air_obj #SpectroscopyData with the air EIS sweep
spec_h2o_obj = data.lcr_obj.spec_h2o_obj #SpectroscopyData with the water EIS sweep
spec_ice_obj = data.lcr_obj.spec_ice_obj #SpectroscopyData with the ice EIS sweep
spec_fr_obj = data.lcr_obj.freerun_obj #PHOBOSData with the freerun acquisition EIS sweep

# Now, let's generate the plot to validate our experimental batch!
# generate the plots for the water EIS

#Permittivity by log of frequency
yaxis_scale = 1e5
plt.figure(1)
plt.suptitle('Permittivity')
plt.subplot(2, 1, 1)
leg = []
plt.plot(np.log10(spec_h2o_obj.freqs), spec_h2o_obj.eps_real/yaxis_scale)
leg.append('water')
plt.ylabel(f"ε' x {yaxis_scale}")
plt.xlabel(f'log(frequency)')
plt.legend(leg)
plt.grid()

plt.subplot(2, 1, 2)
leg = []
plt.plot(np.log10(spec_h2o_obj.freqs), spec_h2o_obj.eps_imag/yaxis_scale)
leg.append('water')
plt.ylabel(f"ε'' x {yaxis_scale}")
plt.xlabel(f'log(frequency)')
plt.legend(leg)
plt.grid()
plt.tight_layout()
plt.show()

#tan delta by log of frequency
plt.figure(2)
plt.title('tanδ')
leg = []
plt.plot(np.log10(spec_h2o_obj.freqs), spec_h2o_obj.eps_imag/spec_h2o_obj.eps_real) #tan_delta = eps''/eps'
leg.append('water')
plt.xlabel("log(frequency)")
plt.ylabel("tanδ")
plt.legend(leg)
plt.grid()
plt.tight_layout()
plt.show()

#Conductivity by log of frequency
plt.figure(3)
plt.suptitle('Conductivity')
leg = []
plt.plot(np.log10(spec_h2o_obj.freqs), spec_h2o_obj.sigma_real, color='tab:blue')
leg.append("σ' water")
plt.plot(np.log10(spec_h2o_obj.freqs), spec_h2o_obj.sigma_imag, color='tab:blue', linestyle='dotted')
leg.append("σ'' water")
plt.ylabel(f"σ'")
plt.xlabel(f'log(frequency)')
plt.legend(leg)
plt.grid()
plt.tight_layout()
plt.show()

#Cole-Cole permittivity
plt.figure(4)
leg = []
plt.title('Cole-Cole Permittivity')
plt.plot(spec_h2o_obj.eps_real, spec_h2o_obj.eps_imag)
leg.append('water')
plt.xlabel("ε'")
plt.ylabel("ε''")
plt.legend(leg)
plt.grid()
plt.tight_layout()
plt.show()

#Cole-Cole conductivity
plt.figure(5)
leg = []
plt.title('Cole-Cole Conductivity')
plt.plot(spec_h2o_obj.sigma_real, spec_h2o_obj.sigma_imag)
leg.append('water')
plt.xlabel("σ'")
plt.ylabel("σ''")
plt.legend(leg)
plt.grid()
plt.tight_layout()
plt.show()

#Nyquist (-Z'' by Z')
plt.figure(6)
leg = []
plt.title('Nyquist')
plt.plot(spec_h2o_obj.z_real, spec_h2o_obj.z_imag)
leg.append('water')
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.legend(leg)
plt.grid()
plt.tight_layout()
plt.show()

#generate the plots for ice EIS

#Permittivity by log of frequency
yaxis_scale = 1e5
plt.figure(7)
plt.suptitle('Permittivity')
plt.subplot(2, 1, 1)
leg = []
plt.plot(np.log10(spec_ice_obj.freqs), spec_ice_obj.eps_real/yaxis_scale)
leg.append('ice')
plt.ylabel(f"ε' x {yaxis_scale}")
plt.xlabel(f'log(frequency)')
plt.legend(leg)
plt.grid()

plt.subplot(2, 1, 2)
leg = []
plt.plot(np.log10(spec_ice_obj.freqs), spec_ice_obj.eps_imag/yaxis_scale)
leg.append('ice')
plt.ylabel(f"ε'' x {yaxis_scale}")
plt.xlabel(f'log(frequency)')
plt.legend(leg)
plt.grid()
plt.tight_layout()
plt.show()

#tan delta by log of frequency
plt.figure(8)
plt.title('tanδ')
leg = []
plt.plot(np.log10(spec_ice_obj.freqs), spec_ice_obj.eps_imag/spec_ice_obj.eps_real) #tan_delta = eps''/eps'
leg.append('ice')
plt.xlabel("log(frequency)")
plt.ylabel("tanδ")
plt.legend(leg)
plt.grid()
plt.tight_layout()
plt.show()

#Conductivity by log of frequency
plt.figure(9)
plt.suptitle('Conductivity')
leg = []
plt.plot(np.log10(spec_ice_obj.freqs), spec_ice_obj.sigma_real, color='tab:blue')
leg.append("σ' ice")
plt.plot(np.log10(spec_ice_obj.freqs), spec_ice_obj.sigma_imag, color='tab:blue', linestyle='dotted')
leg.append("σ'' ice")
plt.ylabel(f"σ'")
plt.xlabel(f'log(frequency)')
plt.legend(leg)
plt.grid()
plt.tight_layout()
plt.show()

#Cole-Cole permittivity
plt.figure(10)
leg = []
plt.title('Cole-Cole Permittivity')
plt.plot(spec_ice_obj.eps_real, spec_ice_obj.eps_imag)
leg.append('ice')
plt.xlabel("ε'")
plt.ylabel("ε''")
plt.legend(leg)
plt.grid()
plt.tight_layout()
plt.show()

#Cole-Cole conductivity
plt.figure(11)
leg = []
plt.title('Cole-Cole Conductivity')
plt.plot(spec_ice_obj.sigma_real, spec_ice_obj.sigma_imag)
leg.append('ice')
plt.xlabel("σ'")
plt.ylabel("σ''")
plt.legend(leg)
plt.grid()
plt.tight_layout()
plt.show()

#Nyquist (-Z'' by Z')
plt.figure(12)
leg = []
plt.title('Nyquist')
plt.plot(spec_ice_obj.z_real, spec_ice_obj.z_imag)
leg.append('ice')
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.legend(leg)
plt.grid()
plt.tight_layout()
plt.show()
