import numpy as np
from framework import file_lcr, characterization_utils, fitting_utils, data_types
import pickle
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class MediumData:
    def __init__(self, topology:dict, data_medium:data_types.SpectroscopyData, data_air:data_types.SpectroscopyData, media:str, eps_func=characterization_utils.dielectric_params_generic):
        '''
        :param topology: list of circuit topologies that output the modeled impedance of the expected circuit
        :param data_medium: SpectroscopyData structure for the frequency sweep in the media to be characterized
        :param data_air: SpectroscopyData structure for the frequency sweep in the air
        :param media: identifier of the media to be characterized
        :param eps_func: function to compute electrical permittivity
        '''

        #validate 'topology'
        if not isinstance(topology, dict):
            raise TypeError(f'[MediumData] "topology" must be a dictionary!')

        #validate 'data_medium'
        valid_types = [data_types.SpectroscopyData, data_types.CommercialCellData, data_types.FlangeData]
        if type(data_medium) not in valid_types:
            raise TypeError(f'[MediumData] "data_medium" must be a SpectroscopyData structure! Curr. type = {type(data_medium)}')

        #validate 'data_air'
        if not isinstance(data_air, data_types.SpectroscopyData):
            raise TypeError(f'[MediumData] "data_air" must be a SpectroscopyData structure! Curr. type = {type(data_air)}')

        self.topology = topology
        self.media = media
        self.spec_media_obj = data_medium #forward SpectroscopyData attributes to the MediumData class
        self.eps_real, self.eps_imag = eps_func(data_medium, data_air, data_medium.freqs) #compute the permittivity based on the 'eps_func' argument
        self.z_real, self.z_imag = characterization_utils.complex_impedance(data_medium, data_medium.freqs) #compute the impedance
        self.sigma_real, self.sigma_imag = characterization_utils.complex_conductivity(data_medium, data_air, data_medium.freqs, eps_func=eps_func) #compute the conductivity
        self.fit_data = self.batch_fit_circuit(topology, data_medium, data_medium.freqs)

    def batch_fit_circuit(self, topology:dict, data_medium:data_types.SpectroscopyData, freqs:np.ndarray):
        '''
        :param topology: list of circuit topologies that output the modeled impedance of the expected circuit
        :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
        :param freqs: array with the swept frequencies
        :return: a dictionary with the fitted data, the circuit parameters, and goodness of fit statistics
        '''

        if isinstance(data_medium, data_types.SpectroscopyData):
            linKK_obj = fitting_utils.LinearKramersKronig(data_medium, freqs, c=0.5, max_iter=100, add_capacitor=True, verbose=False) #validate the data with the Kramers-Kronig test before fitting

            #Kramers-Kronig threshold
            if linKK_obj.chi_square > 1e-2:
                print(f'[MediumData] {self.media} failed the Linear Kramers-Kronig test: x² = {linKK_obj.chi_square}')
                fitted_circuits = {"z_meas_real": self.z_real, "z_meas_imag": self.z_imag,
                                   "z_hat_real_KK": linKK_obj.z_hat_real, "z_hat_imag_KK": linKK_obj.z_hat_imag, "chi_square_KK": linKK_obj.chi_square,
                                   "params_KK": linKK_obj.fit_params, "M_KK": linKK_obj.fit_components}
            else:
                fitted_circuits = dict((circ_name, {"z_meas_real": self.z_real, "z_meas_imag": self.z_imag,
                                                    "z_hat_real_KK": linKK_obj.z_hat_real, "z_hat_imag_KK": linKK_obj.z_hat_imag, "chi_square_KK": linKK_obj.chi_square,
                                                    "params_KK": linKK_obj.fit_params, "M_KK": linKK_obj.fit_components,
                                                    "z_hat_real": None, "z_hat_imag": None, "nmse": None, "params": None, "t": None})
                                                    for circ_name in list(topology.keys())) #dictionary to store fitted circuits parameters and values to a dictionary
                for circuit in list(topology.keys()):
                    curr_fit_attributes = topology[circuit] #extract guess and scaling from the input topology argument
                    curr_fit_obj = fitting_utils.EquivalentCircuit(circuit, data_medium, freqs)

                    #fit using the gradient-based method (BFGS)
                    curr_fit_params_BFGS = curr_fit_obj.fit_circuit(curr_fit_attributes["guess"], curr_fit_attributes["scale"], method="BFGS")
                    z_hat_real_BFGS = curr_fit_params_BFGS.opt_fit.real #real part of the fitting
                    z_hat_imag_BFGS = curr_fit_params_BFGS.opt_fit.imag #imaginary part of the fitting
                    nmse_BFGS = curr_fit_params_BFGS.nmse_score
                    params_BFGS = curr_fit_params_BFGS.opt_params_scaled
                    t_BFGS = curr_fit_params_BFGS.t_elapsed

                    #fit using the non-linear-least-squares-based method (NLLS)
                    #hacky-fix: scipy's curve fit has an issue that when the total iterations are reached without convergence, it raises an error
                    try:
                        curr_fit_params_NLLS = curr_fit_obj.fit_circuit(curr_fit_attributes["guess"], curr_fit_attributes["scale"], method="NLLS")
                        z_hat_real_NLLS = curr_fit_params_NLLS.opt_fit.real
                        z_hat_imag_NLLS = curr_fit_params_NLLS.opt_fit.imag
                        nmse_NLLS = curr_fit_params_NLLS.nmse_score
                        params_NLLS = curr_fit_params_NLLS.opt_params_scaled
                        t_NLLS = curr_fit_params_NLLS.t_elapsed
                    except:
                        z_hat_real_NLLS = np.zeros(shape=(len(self.z_real), 1, len(data_medium.freqs)))
                        z_hat_imag_NLLS = np.zeros(shape=(len(self.z_imag), 1, len(data_medium.freqs)))
                        nmse_NLLS = 1
                        params_NLLS = np.zeros_like(params_BFGS)
                        t_NLLS = None

                    #store the optimization parameters into the fit dictionary
                    fitted_circuits[circuit]["z_hat_real"] = np.vstack((z_hat_real_BFGS, z_hat_real_NLLS)).T
                    fitted_circuits[circuit]["z_hat_imag"] = np.vstack((z_hat_imag_BFGS, z_hat_imag_NLLS)).T
                    fitted_circuits[circuit]["nmse"] = np.vstack((nmse_BFGS, nmse_NLLS)).T
                    fitted_circuits[circuit]["params"] = np.vstack((params_BFGS, params_NLLS)).T
                    fitted_circuits[circuit]["t"] = np.vstack((t_BFGS, t_NLLS)).T

            return fitted_circuits

        #if isinstance(data_medium, data_types.CommercialCellData):

class BatchLCR:
    def __init__(self, base_path, topology:dict, freq_threshold=None, n_samples_freq=1, n_samples_spectrum=3, sweeptype="cell", aggregate=None, eps_func=characterization_utils.dielectric_params_generic, timezone=-3, save=False):
        '''
        :param base_path: base path where all .csv files are stored
        :param topology: list of equivalent circuits to fit the data
        :param freq_threshold: value to filter frequencies from (i.e., 100 filters from 100 Hz. OBS: None to skip filtering)
        :param n_samples_freq: samples per freerun sweep
        :param n_samples_spectrum: samples per EIS sweep
        :param sweeptype: which hardware was used to acquire the signals
        :param aggregate: how to organize the data for each mode (None as default)
        :param timezone: timezone to convert unix timestamp to human timestamp
        :param save: flag to save the structure as a pickle file
        '''

        #validate 'base_path' argument
        if not os.path.exists(base_path):
            raise FileNotFoundError(f'[BatchLCR] Base path {base_path} does not exist!')

        #validate 'topology'
        if not isinstance(topology, dict):
            raise TypeError(f'[BatchLCR] "topology" must be a dictionary!')

        #validate 'freq_threshold'
        if not freq_threshold is None:
            if isinstance(freq_threshold, float) or isinstance(freq_threshold, int):
                if not freq_threshold >= 0:
                    raise ValueError(f'[BatchLCR] Frequency threshold cannot be smaller than zero!')
            else:
                raise TypeError(f'[BatchLCR] Frequency threshold must be an integer or float!')

        #process reference sweeps into SpectroscopyData objects
        lcr_files = [found_file for found_file in os.listdir(base_path) if found_file.endswith('.csv')] #list all LCR-based files prior to processing
        spec_objects = {"c0": None, "c1": None, "cice": None, "cthf": None, "ctest": None} #attribute the objects to sweep-based keys

        for lcr_file in lcr_files:
            filepath = os.path.join(base_path, lcr_file) #relative path of the EIS sweep
            try:
                eis_obj_key = lcr_file.replace('.csv', '') #key to extract the SpectroscopyData object and run the constructor
                eis_obj_key = eis_obj_key.replace('_', '') #remove underline to access the attributes
                if "test" in lcr_file:
                    spec_objects[eis_obj_key] = file_lcr.read(filepath, n_samples=n_samples_freq, sweeptype=sweeptype, acquisition_mode="freq", aggregate=aggregate, timezone=timezone) #FlangeData/CommercialCellData object

                    #filter frequency threshold
                    if freq_threshold:
                        freq_mask = spec_objects[eis_obj_key].freqs >= freq_threshold #frequency mask
                        spec_objects[eis_obj_key].freqs = spec_objects[eis_obj_key].freqs[freq_mask] #filter frequencies
                        spec_objects[eis_obj_key].n_freqs = len(spec_objects[eis_obj_key].freqs) #update number of swept frequencies
                        if aggregate is None:
                            spec_objects[eis_obj_key].Cp = spec_objects[eis_obj_key].Cp[:,:,:,freq_mask] #filter capacitance
                            spec_objects[eis_obj_key].Rp = spec_objects[eis_obj_key].Rp[:,:,:,freq_mask] #filter resistance
                        else:
                            spec_objects[eis_obj_key].Cp = spec_objects[eis_obj_key].Cp[:,:,freq_mask] #filter capacitance
                            spec_objects[eis_obj_key].Rp = spec_objects[eis_obj_key].Rp[:,:,freq_mask] #filter resistance
                else:
                    spec_objects[eis_obj_key] = file_lcr.read(filepath, n_samples=n_samples_spectrum, sweeptype=sweeptype, acquisition_mode="spectrum", aggregate=aggregate, timezone=timezone) #SpectroscopyData object for current EIS file

                    #filter frequency threshold
                    if freq_threshold:
                        freq_mask = spec_objects[eis_obj_key].freqs >= freq_threshold #frequency mask
                        spec_objects[eis_obj_key].freqs = spec_objects[eis_obj_key].freqs[freq_mask] #filter frequencies
                        spec_objects[eis_obj_key].n_freqs = len(spec_objects[eis_obj_key].freqs) #update number of swept frequencies
                        spec_objects[eis_obj_key].Cp = spec_objects[eis_obj_key].Cp[freq_mask] #filter capacitance
                        spec_objects[eis_obj_key].Rp = spec_objects[eis_obj_key].Rp[freq_mask] #filter resistance

            except Exception as e:
                print(f'[BatchLCR] Failed "{filepath}" with {e}! Processing data without it...')

        #attribute each processed SpectroscopyData object to their equivalent MediumData object
        if spec_objects["c0"] is not None:
            if spec_objects["c1"] is not None:
                self.spec_h2o_obj = MediumData(topology, spec_objects["c1"], spec_objects["c0"], "water", eps_func=eps_func) #object to store water EIS data
            if spec_objects["cice"] is not None:
                self.spec_ice_obj = MediumData(topology, spec_objects["cice"], spec_objects["c0"],"ice", eps_func=eps_func) #object to store ice EIS data
            if spec_objects["cthf"] is not None:
                self.spec_thf_obj = MediumData(topology, spec_objects["cthf"], spec_objects["c0"],"thf", eps_func=eps_func) #object to store ice EIS data
            if spec_objects["ctest"] is not None:
                self.freerun_obj = MediumData(topology, spec_objects["ctest"], spec_objects["c0"],"freerun", eps_func=eps_func) #object to store freerun sweep data


circuits = {"Longo2020": {"guess": np.array([1, 1, 1, 1, 1, 1, 1, 1]), "scale": np.array([1e3, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1])},
            "Zurich2021": {"guess": np.array([1, 1, 1, 1, 1, 1]), "scale": np.array([1e5, 1e-7, 1, 1e5, 1e3, 1e-8])},
            "Zhang2024": {"guess": np.array([1, 1, 1, 1, 1, 1]), "scale": np.array([1e-9, 1, 1e6, 1e2, 1e2, 1e-8])},
            "Yang2025": {"guess": np.array([1, 1, 1, 1, 1, 1]), "scale": np.array([1e4, 1e4, 1e-8, 1, 1e4, 1e-8])},
            "Fouquet2005": {"guess": np.array([1, 1, 1, 1, 1, 1]), "scale": np.array([1e4, 1e3, 1e-8, 1, 1e4, 1e-4])},
            "Awayssa2025": {"guess": np.array([1, 1, 1, 1, 1]), "scale": np.array([1e3, 1e3, 1e-5, 1e-9, 1e-8])}} #list of circuits to attempt fitting the data

batch_lcr_obj = BatchLCR('../data/testICE_30_01_26/', circuits, freq_threshold=100, aggregate=np.mean, eps_func=characterization_utils.dielectric_params_corrected)

#plot static ice vs. water
fig1, axs1 = plt.subplots(nrows=6, ncols=2, figsize=(10,10))

#longo2020
axs1[0,0].plot(batch_lcr_obj.spec_h2o_obj.fit_data["Longo2020"]["z_meas_real"], batch_lcr_obj.spec_h2o_obj.fit_data["Longo2020"]["z_meas_imag"], color="tab:blue", label="water measured")
axs1[0,0].plot(batch_lcr_obj.spec_h2o_obj.fit_data["Longo2020"]["z_hat_real"][:,0], -batch_lcr_obj.spec_h2o_obj.fit_data["Longo2020"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Longo2020 (BFGS)")
axs1[0,0].plot(batch_lcr_obj.spec_h2o_obj.fit_data["Longo2020"]["z_hat_real"][:,1], -batch_lcr_obj.spec_h2o_obj.fit_data["Longo2020"]["z_hat_real"][:,1], color="tab:green", linestyle="dashed", label="Longo2020 (NLLS)")
axs1[0,0].legend()
axs1[0,0].grid()
axs1[0,0].set_xlabel("Z'")
axs1[0,0].set_ylabel("Z''")
axs1[0,0].set_ylim([0,300])

axs1[0,1].plot(batch_lcr_obj.spec_ice_obj.fit_data["Longo2020"]["z_meas_real"], batch_lcr_obj.spec_ice_obj.fit_data["Longo2020"]["z_meas_imag"], color="tab:blue", label="water measured")
axs1[0,1].plot(batch_lcr_obj.spec_ice_obj.fit_data["Longo2020"]["z_hat_real"][:,0], -batch_lcr_obj.spec_ice_obj.fit_data["Longo2020"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Longo2020 (BFGS)")
axs1[0,1].plot(batch_lcr_obj.spec_ice_obj.fit_data["Longo2020"]["z_hat_real"][:,1], -batch_lcr_obj.spec_ice_obj.fit_data["Longo2020"]["z_hat_real"][:,1], color="tab:green", linestyle="dashed", label="Longo2020 (NLLS)")
axs1[0,1].legend()
axs1[0,1].grid()
axs1[0,1].set_xlabel("Z'")
axs1[0,1].set_ylabel("Z''")

#zurich2021
axs1[1,0].plot(batch_lcr_obj.spec_h2o_obj.fit_data["Zurich2021"]["z_meas_real"], batch_lcr_obj.spec_h2o_obj.fit_data["Zurich2021"]["z_meas_imag"], color="tab:blue", label="water measured")
axs1[1,0].plot(batch_lcr_obj.spec_h2o_obj.fit_data["Zurich2021"]["z_hat_real"][:,0], -batch_lcr_obj.spec_h2o_obj.fit_data["Zurich2021"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Zurich2021 (BFGS)")
axs1[1,0].plot(batch_lcr_obj.spec_h2o_obj.fit_data["Zurich2021"]["z_hat_real"][:,1], -batch_lcr_obj.spec_h2o_obj.fit_data["Zurich2021"]["z_hat_real"][:,1], color="tab:green", linestyle="dashed", label="Zurich2021 (NLLS)")
axs1[1,0].legend()
axs1[1,0].grid()
axs1[1,0].set_xlabel("Z'")
axs1[1,0].set_ylabel("Z''")
axs1[1,0].set_ylim([0,300])

axs1[1,1].plot(batch_lcr_obj.spec_ice_obj.fit_data["Zurich2021"]["z_meas_real"], batch_lcr_obj.spec_ice_obj.fit_data["Zurich2021"]["z_meas_imag"], color="tab:blue", label="water measured")
axs1[1,1].plot(batch_lcr_obj.spec_ice_obj.fit_data["Zurich2021"]["z_hat_real"][:,0], -batch_lcr_obj.spec_ice_obj.fit_data["Zurich2021"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Zurich2021 (BFGS)")
axs1[1,1].plot(batch_lcr_obj.spec_ice_obj.fit_data["Zurich2021"]["z_hat_real"][:,1], -batch_lcr_obj.spec_ice_obj.fit_data["Zurich2021"]["z_hat_real"][:,1], color="tab:green", linestyle="dashed", label="Zurich2021 (NLLS)")
axs1[1,1].legend()
axs1[1,1].grid()
axs1[1,1].set_xlabel("Z'")
axs1[1,1].set_ylabel("Z''")

#zhang2024
axs1[2,0].plot(batch_lcr_obj.spec_h2o_obj.fit_data["Zhang2024"]["z_meas_real"], batch_lcr_obj.spec_h2o_obj.fit_data["Zhang2024"]["z_meas_imag"], color="tab:blue", label="water measured")
axs1[2,0].plot(batch_lcr_obj.spec_h2o_obj.fit_data["Zhang2024"]["z_hat_real"][:,0], -batch_lcr_obj.spec_h2o_obj.fit_data["Zhang2024"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Zhang2024 (BFGS)")
axs1[2,0].plot(batch_lcr_obj.spec_h2o_obj.fit_data["Zhang2024"]["z_hat_real"][:,1], -batch_lcr_obj.spec_h2o_obj.fit_data["Zhang2024"]["z_hat_real"][:,1], color="tab:green", linestyle="dashed", label="Zhang2024 (NLLS)")
axs1[2,0].legend()
axs1[2,0].grid()
axs1[2,0].set_xlabel("Z'")
axs1[2,0].set_ylabel("Z''")
axs1[2,0].set_ylim([0,300])

axs1[2,1].plot(batch_lcr_obj.spec_ice_obj.fit_data["Zhang2024"]["z_meas_real"], batch_lcr_obj.spec_ice_obj.fit_data["Zhang2024"]["z_meas_imag"], color="tab:blue", label="water measured")
axs1[2,1].plot(batch_lcr_obj.spec_ice_obj.fit_data["Zhang2024"]["z_hat_real"][:,0], -batch_lcr_obj.spec_ice_obj.fit_data["Zhang2024"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Zhang2024 (BFGS)")
axs1[2,1].plot(batch_lcr_obj.spec_ice_obj.fit_data["Zhang2024"]["z_hat_real"][:,1], -batch_lcr_obj.spec_ice_obj.fit_data["Zhang2024"]["z_hat_real"][:,1], color="tab:green", linestyle="dashed", label="Zhang2024 (NLLS)")
axs1[2,1].legend()
axs1[2,1].grid()
axs1[2,1].set_xlabel("Z'")
axs1[2,1].set_ylabel("Z''")

#yang2025
axs1[3,0].plot(batch_lcr_obj.spec_h2o_obj.fit_data["Yang2025"]["z_meas_real"], batch_lcr_obj.spec_h2o_obj.fit_data["Yang2025"]["z_meas_imag"], color="tab:blue", label="water measured")
axs1[3,0].plot(batch_lcr_obj.spec_h2o_obj.fit_data["Yang2025"]["z_hat_real"][:,0], -batch_lcr_obj.spec_h2o_obj.fit_data["Yang2025"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Yang2025 (BFGS)")
axs1[3,0].plot(batch_lcr_obj.spec_h2o_obj.fit_data["Yang2025"]["z_hat_real"][:,1], -batch_lcr_obj.spec_h2o_obj.fit_data["Yang2025"]["z_hat_real"][:,1], color="tab:green", linestyle="dashed", label="Yang2025 (NLLS)")
axs1[3,0].legend()
axs1[3,0].grid()
axs1[3,0].set_xlabel("Z'")
axs1[3,0].set_ylabel("Z''")
axs1[3,0].set_ylim([0,300])

axs1[3,1].plot(batch_lcr_obj.spec_ice_obj.fit_data["Yang2025"]["z_meas_real"], batch_lcr_obj.spec_ice_obj.fit_data["Yang2025"]["z_meas_imag"], color="tab:blue", label="water measured")
axs1[3,1].plot(batch_lcr_obj.spec_ice_obj.fit_data["Yang2025"]["z_hat_real"][:,0], -batch_lcr_obj.spec_ice_obj.fit_data["Yang2025"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Yang2025 (BFGS)")
axs1[3,1].plot(batch_lcr_obj.spec_ice_obj.fit_data["Yang2025"]["z_hat_real"][:,1], -batch_lcr_obj.spec_ice_obj.fit_data["Yang2025"]["z_hat_real"][:,1], color="tab:green", linestyle="dashed", label="Yang2025 (NLLS)")
axs1[3,1].legend()
axs1[3,1].grid()
axs1[3,1].set_xlabel("Z'")
axs1[3,1].set_ylabel("Z''")

#Fouquet2005
axs1[4,0].plot(batch_lcr_obj.spec_h2o_obj.fit_data["Fouquet2005"]["z_meas_real"], batch_lcr_obj.spec_h2o_obj.fit_data["Fouquet2005"]["z_meas_imag"], color="tab:blue", label="water measured")
axs1[4,0].plot(batch_lcr_obj.spec_h2o_obj.fit_data["Fouquet2005"]["z_hat_real"][:,0], -batch_lcr_obj.spec_h2o_obj.fit_data["Fouquet2005"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Fouquet2005 (BFGS)")
axs1[4,0].plot(batch_lcr_obj.spec_h2o_obj.fit_data["Fouquet2005"]["z_hat_real"][:,1], -batch_lcr_obj.spec_h2o_obj.fit_data["Fouquet2005"]["z_hat_real"][:,1], color="tab:green", linestyle="dashed", label="Fouquet2005 (NLLS)")
axs1[4,0].legend()
axs1[4,0].grid()
axs1[4,0].set_xlabel("Z'")
axs1[4,0].set_ylabel("Z''")
axs1[4,0].set_ylim([0,300])

axs1[4,1].plot(batch_lcr_obj.spec_ice_obj.fit_data["Fouquet2005"]["z_meas_real"], batch_lcr_obj.spec_ice_obj.fit_data["Fouquet2005"]["z_meas_imag"], color="tab:blue", label="water measured")
axs1[4,1].plot(batch_lcr_obj.spec_ice_obj.fit_data["Fouquet2005"]["z_hat_real"][:,0], -batch_lcr_obj.spec_ice_obj.fit_data["Fouquet2005"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Fouquet2005 (BFGS)")
axs1[4,1].plot(batch_lcr_obj.spec_ice_obj.fit_data["Fouquet2005"]["z_hat_real"][:,1], -batch_lcr_obj.spec_ice_obj.fit_data["Fouquet2005"]["z_hat_real"][:,1], color="tab:green", linestyle="dashed", label="Fouquet2005 (NLLS)")
axs1[4,1].legend()
axs1[4,1].grid()
axs1[4,1].set_xlabel("Z'")
axs1[4,1].set_ylabel("Z''")

#Awayssa2025
axs1[5,0].plot(batch_lcr_obj.spec_h2o_obj.fit_data["Awayssa2025"]["z_meas_real"], batch_lcr_obj.spec_h2o_obj.fit_data["Awayssa2025"]["z_meas_imag"], color="tab:blue", label="water measured")
axs1[5,0].plot(batch_lcr_obj.spec_h2o_obj.fit_data["Awayssa2025"]["z_hat_real"][:,0], -batch_lcr_obj.spec_h2o_obj.fit_data["Awayssa2025"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Awayssa2025 (BFGS)")
axs1[5,0].plot(batch_lcr_obj.spec_h2o_obj.fit_data["Awayssa2025"]["z_hat_real"][:,1], -batch_lcr_obj.spec_h2o_obj.fit_data["Awayssa2025"]["z_hat_real"][:,1], color="tab:green", linestyle="dashed", label="Awayssa2025 (NLLS)")
axs1[5,0].legend()
axs1[5,0].grid()
axs1[5,0].set_xlabel("Z'")
axs1[5,0].set_ylabel("Z''")
axs1[5,0].set_ylim([0,300])

axs1[5,1].plot(batch_lcr_obj.spec_ice_obj.fit_data["Awayssa2025"]["z_meas_real"], batch_lcr_obj.spec_ice_obj.fit_data["Awayssa2025"]["z_meas_imag"], color="tab:blue", label="water measured")
axs1[5,1].plot(batch_lcr_obj.spec_ice_obj.fit_data["Awayssa2025"]["z_hat_real"][:,0], -batch_lcr_obj.spec_ice_obj.fit_data["Awayssa2025"]["z_hat_imag"][:,0], color="tab:orange", linestyle="dotted", label="Awayssa2025 (BFGS)")
axs1[5,1].plot(batch_lcr_obj.spec_ice_obj.fit_data["Awayssa2025"]["z_hat_real"][:,1], -batch_lcr_obj.spec_ice_obj.fit_data["Awayssa2025"]["z_hat_real"][:,1], color="tab:green", linestyle="dashed", label="Awayssa2025 (NLLS)")
axs1[5,1].legend()
axs1[5,1].grid()
axs1[5,1].set_xlabel("Z'")
axs1[5,1].set_ylabel("Z''")