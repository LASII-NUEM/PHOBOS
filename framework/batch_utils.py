import numpy as np
from framework import file_lcr, characterization_utils, fitting_utils, data_types
import pickle
import os

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
        self.media_obj = data_medium #forward SpectroscopyData attributes to the MediumData class
        self.eps_real, self.eps_imag = eps_func(data_medium, data_air, data_medium.freqs) #compute the permittivity based on the 'eps_func' argument
        self.z_real, self.z_imag = characterization_utils.complex_impedance(data_medium, data_medium.freqs) #compute the impedance
        self.sigma_real, self.sigma_imag = characterization_utils.complex_conductivity(data_medium, data_air, data_medium.freqs, eps_func=eps_func) #compute the conductivity

        if isinstance(data_medium, data_types.SpectroscopyData):
            self.fit_data = self.batch_fit_circuit(topology, data_medium, data_medium.freqs)

    def batch_fit_circuit(self, topology:dict, data_medium:data_types.SpectroscopyData, freqs:np.ndarray):
        '''
        :param topology: list of circuit topologies that output the modeled impedance of the expected circuit
        :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
        :param freqs: array with the swept frequencies
        :return: a dictionary with the fitted data, the circuit parameters, and goodness of fit statistics
        '''

        #validate the data with the Kramers-Kronig test before fitting
        linKK_obj = fitting_utils.LinearKramersKronig(data_medium, freqs, c=0.5, max_iter=100, add_capacitor=True, verbose=False)

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

class BatchLCR:
    def __init__(self, base_path, topology:dict, freq_threshold=None, n_samples_freq=1, n_samples_spectrum=3, sweeptype="cell", aggregate=None, eps_func=characterization_utils.dielectric_params_generic, timezone=-3):
        '''
        :param base_path: base path where all .csv files are stored
        :param topology: list of equivalent circuits to fit the data
        :param freq_threshold: value to filter frequencies from (i.e., 100 filters from 100 Hz. OBS: None to skip filtering)
        :param n_samples_freq: samples per freerun sweep
        :param n_samples_spectrum: samples per EIS sweep
        :param sweeptype: which hardware was used to acquire the signals
        :param aggregate: how to organize the data for each mode (None as default)
        :param timezone: timezone to convert unix timestamp to human timestamp
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
                    placeholder_temp_file = os.path.join(base_path, "c_temp.lvm")
                    spec_objects[eis_obj_key] = data_types.PHOBOSData(filepath, filename_temperature=placeholder_temp_file,
                                                                      n_samples=n_samples_freq, sweeptype=sweeptype, acquisition_mode="freq", aggregate=aggregate) #constructor of the PHOBOSData object

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
                            spec_objects[eis_obj_key].Cp_norm = spec_objects[eis_obj_key].Cp_norm[:,:,freq_mask] #filter normalized capacitance
                            spec_objects[eis_obj_key].Cp_agg = spec_objects[eis_obj_key].Cp_agg[:,freq_mask] #filter aggregated capacitance
                            spec_objects[eis_obj_key].agg_Cp_norm = spec_objects[eis_obj_key].agg_Cp_norm[:,freq_mask] #filter aggregated normalized capacitance
                            spec_objects[eis_obj_key].Rp = spec_objects[eis_obj_key].Rp[:,:,freq_mask] #filter resistance
                            spec_objects[eis_obj_key].Rp_norm = spec_objects[eis_obj_key].Rp_norm[:,:,freq_mask] #filter normalized resistance
                            spec_objects[eis_obj_key].Rp_agg = spec_objects[eis_obj_key].Rp_agg[:,freq_mask] #filter aggregated resistance
                            spec_objects[eis_obj_key].agg_Rp_norm = spec_objects[eis_obj_key].agg_Rp_norm[:,freq_mask] #filter aggregated normalized resistance
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
            self.spec_air_obj = spec_objects["c0"] #object to store air EIS data
            if spec_objects["c1"] is not None:
                self.spec_h2o_obj = MediumData(topology, spec_objects["c1"], spec_objects["c0"], "water", eps_func=eps_func) #object to store water EIS data
            if spec_objects["cice"] is not None:
                self.spec_ice_obj = MediumData(topology, spec_objects["cice"], spec_objects["c0"],"ice", eps_func=eps_func) #object to store ice EIS data
            if spec_objects["cthf"] is not None: #TODO: test later with THF (once the data exists)
                self.spec_thf_obj = MediumData(topology, spec_objects["cthf"], spec_objects["c0"],"thf", eps_func=eps_func) #object to store ice EIS data
            if spec_objects["ctest"] is not None:
                self.freerun_obj = spec_objects["ctest"] #object to store freerun sweep data

class BatchIA:
    def __init__(self):
        #TODO: repeat the BatchLCR routine to BatchIA
        self.data = None

class BatchOrganizer:
    def __init__(self, base_path, circuits:dict, freq_threshold=None, n_samples_freq=1, n_samples_spectrum=3, sweeptype="cell", aggregate=None, eps_func=characterization_utils.dielectric_params_generic, timezone=-3, save=True):
        '''
        :param base_path: base path where all .csv files are stored
        :param circuits: list of equivalent circuits to fit the data
        :param freq_threshold: value to filter frequencies from (i.e., 100 filters from 100 Hz. OBS: None to skip filtering)
        :param n_samples_freq: samples per freerun sweep
        :param n_samples_spectrum: samples per EIS sweep
        :param sweeptype: which hardware was used to acquire the signals
        :param aggregate: how to organize the data for each mode (None as default)
        :param timezone: timezone to convert unix timestamp to human timestamp
        :param save: flag to enable pickle dumping
        '''

        try:
            self.lcr_obj = BatchLCR(base_path, circuits, freq_threshold=freq_threshold,
                                    n_samples_freq=n_samples_freq, n_samples_spectrum=n_samples_spectrum, sweeptype=sweeptype,
                                    aggregate=aggregate, eps_func=eps_func, timezone=timezone) #process the batch for LCR data
            print(f'[BatchOrganizer] LCR batch processing finished!')
        except:
            print(f'[BatchOrganizer] LCR batch processing failed! Processing data without it...')

        self.ia_obj = BatchIA() #TODO: process the batch data for IA data

        #if True, save the attributes as a pickle file
        if save:
            pickle_file = f"batch_{base_path.split('/')[-1]}.pkl"
            self.full_relative_path = os.path.join(base_path, pickle_file)
            with open(self.full_relative_path, 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'[BatchOrganizer] Batch processing output saved at: {self.full_relative_path}')

