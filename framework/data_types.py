import numpy as np
import pandas as pd
import datetime
from framework import file_lcr, file_lvm

class FlangeData:
    def __init__(self, electrode_data:np.ndarray, swept_freqs:np.ndarray, n_samples=1, aggregate=None, timezone=-3):
        '''
        :param electrode_data: raw data output from the PHOBOS acquisition system
        :param swept_freqs: array with all swept frequencies
        :param n_samples: samples per pair swept
        :param aggregate: how to organize the data for each mode (None as default)
        :param timezone: timezone to convert unix timestamp to human timestamp
        '''

        #check if the raw electrode data is a numpy array
        if type(electrode_data) != np.ndarray:
            raise TypeError(f'[FlangeData] Raw electrode data must be a numpy array! Curr. type = {type(electrode_data)}')

        #check if the swept_frequencies argument is a list
        if type(swept_freqs) != np.ndarray:
            raise TypeError(f'[FlangeData] Swept frequencies data must be a numpy array! Curr. type = {type(swept_freqs)}')

        self.freqs = swept_freqs
        self.n_freqs = len(swept_freqs)

        #validate n_samples
        if not (n_samples > 0):
            raise ValueError(f'[FlangeData] n_samples = {n_samples}, must be > 0!')
        self.n_samples = n_samples

        #initialize class attributes
        self.n_modes = None #how many unique values exist in a group of 10 readings
        self.modes = None #emitter/receiver pairs
        self.unix_timestamps = None #unix timestamps
        self.human_timestamps = None #human timestamps
        self.Cp = None #capacitance readings
        self.Rp = None #resistance readings

        #organize the data based on the CSV format
        self.unix_timestamps = electrode_data[:,0] #update timestamps in [ms]
        valid_electrodes = electrode_data[:,2:] #filter the array from the first electrode reading

        #'flange' type sweeps might end before a loop is completed
        self.n_modes = len(set(electrode_data[:int(n_samples*10),1])) #update the attribute
        self.modes = electrode_data[:self.n_modes,1] #update the attribute
        self.modes = np.array([mode.replace('d:','') for mode in self.modes]) #remove "d:" from the string
        samples_per_loop = self.n_modes*self.n_samples #number of samples expected in a full sweep loop
        total_loops = np.floor(len(valid_electrodes)/samples_per_loop) #number of completed loops in the full acquisition
        last_valid_sample = int(total_loops*samples_per_loop) #index of the last sample in the last valid loop
        valid_electrodes = valid_electrodes[:last_valid_sample,:] #filter raw data up to the last valid sample
        self.unix_timestamps = self.unix_timestamps[:last_valid_sample] #filter raw timestamps up to the last valid sample

        if self.n_samples < 2:
            self.unix_timestamps = self.unix_timestamps[np.newaxis,:] #add a new dimension to avoid computing for all samples

        self.unix_timestamps = np.reshape(self.unix_timestamps, [self.n_samples, int(total_loops), self.n_modes]) #[samples, modes, readings]

        #process capacitance and resistance separately
        idx_cp = np.arange(0,int(2*len(self.freqs)),2) #indexes of each capacitance reading
        self.Cp = valid_electrodes[:, idx_cp] #update capacitance readings
        self.Cp = np.reshape(self.Cp, [self.n_samples, int(total_loops), self.n_modes, len(self.freqs)]) #[samples, readings, modes, freqs]
        idx_rp = np.arange(1,int(2*len(self.freqs)),2) #indexes of each resistance reading
        self.Rp = valid_electrodes[:, idx_rp] #update resistance readings
        self.Rp = np.reshape(self.Rp, [self.n_samples, int(total_loops), self.n_modes, len(self.freqs)]) #[samples, readings, modes, freqs]

        #apply aggregation if required on the "modes" axis
        if aggregate is not None:
            self.unix_timestamps = aggregate(self.unix_timestamps, axis=2) #mean over the modes
            self.unix_timestamps = self.unix_timestamps[-1,:] #use the timestamps of the last registered mode per loop
            self.Cp = aggregate(self.Cp, axis=0) #mean over the n_samples
            self.Rp = aggregate(self.Rp, axis=0) #mean over the n_samples
        else:
            self.unix_timestamps = self.unix_timestamps[0,-1,:] #use the timestamps of the last registered mode per loop

        #convert unix timestamps to human timestamps
        self.unix_timestamps /= 1000 #[ms] to [s]
        self.unix_timestamps += timezone*3600 #convert timezone
        self.human_timestamps = pd.to_datetime(self.unix_timestamps, unit='s').to_numpy()

class CommercialCellData:
    def __init__(self, electrode_data:np.ndarray, swept_freqs:np.ndarray, n_samples=1, aggregate=None, timezone=-3):
        '''
        :param electrode_data: raw data output from the PHOBOS acquisition system
        :param swept_freqs: array with all swept frequencies
        :param n_samples: samples per pair swept
        :param aggregate: how to organize the data for each mode (None as default)
        :param timezone: timezone to convert unix timestamp to human timestamp
        '''

        #check if the raw electrode data is a numpy array
        if type(electrode_data) != np.ndarray:
            raise TypeError(f'[CommercialCell] Raw electrode data must be a numpy array! Curr. type = {type(electrode_data)}')

        #check if the swept_frequencies argument is a list
        if type(swept_freqs) != np.ndarray:
            raise TypeError(f'[CommercialCell] Swept frequencies data must be a numpy array! Curr. type = {type(swept_freqs)}')

        self.freqs = swept_freqs
        self.n_freqs = len(swept_freqs)

        #validate n_samples
        if not (n_samples > 0):
            raise ValueError(f'[CommercialCell] n_samples = {n_samples}, must be > 0!')
        self.n_samples = n_samples

        #initialize class attributes
        self.n_modes = None #number of emitter/receiver pairs
        self.modes = None #emitter/receiver pairs
        self.unix_timestamps = None #unix timestamps
        self.human_timestamps = None #human timestamps
        self.Cp = None #capacitance readings
        self.Rp = None #resistance readings

        #organize the data based on the CSV format
        self.unix_timestamps = electrode_data[:,0] #update timestamps in [ms]
        valid_electrodes = electrode_data[:,2:] #filter the array from the first electrode reading

        #'cell' type sweeps might end before a loop is completed
        self.n_modes = 1 #only one e/r is used
        self.modes = ["1-2"] #update the attribute
        samples_per_loop = self.n_modes*self.n_samples #number of samples expected in a full sweep loop
        total_loops = np.floor(len(valid_electrodes)/samples_per_loop) #number of completed loops in the full acquisition
        last_valid_sample = int(total_loops*samples_per_loop) #index of the last sample in the last valid loop
        valid_electrodes = valid_electrodes[:last_valid_sample,:] #filter raw data up to the last valid sample
        self.unix_timestamps = self.unix_timestamps[:last_valid_sample] #filter raw timestamps up to the last valid sample

        if self.n_samples < 2:
            self.unix_timestamps = self.unix_timestamps[np.newaxis,:] #add a new dimension to avoid computing for all samples

        self.unix_timestamps = np.reshape(self.unix_timestamps, [self.n_samples, int(total_loops), self.n_modes]) #[samples, modes, readings]

        #process capacitance and resistance separately
        idx_cp = np.arange(0,int(2*len(self.freqs)),2) #indexes of each capacitance reading
        self.Cp = valid_electrodes[:, idx_cp] #update capacitance readings
        self.Cp = np.reshape(self.Cp, [self.n_samples, int(total_loops), self.n_modes, len(self.freqs)]) #[samples, readings, modes, freqs]
        idx_rp = np.arange(1,int(2*len(self.freqs)),2) #indexes of each resistance reading
        self.Rp = valid_electrodes[:, idx_rp] #update resistance readings
        self.Rp = np.reshape(self.Rp, [self.n_samples, int(total_loops), self.n_modes, len(self.freqs)]) #[samples, readings, modes, freqs]

        #apply aggregation if required on the "samples" axis
        if aggregate is not None:
            self.unix_timestamps = aggregate(self.unix_timestamps, axis=2) #mean over the modes
            self.unix_timestamps = self.unix_timestamps[-1,:] #use the timestamps of the last registered mode per loop
            self.Cp = aggregate(self.Cp, axis=0) #mean over the n_samples
            self.Rp = aggregate(self.Rp, axis=0) #mean over the n_samples
        else:
            self.unix_timestamps = self.unix_timestamps[0,:,-1] #use the timestamps of the last registered mode per loop

        #convert unix timestamps to human timestamps
        self.unix_timestamps /= 1000 #[ms] to [s]
        self.unix_timestamps += timezone*3600 #convert timezone
        self.human_timestamps = pd.to_datetime(self.unix_timestamps, unit='s').to_numpy()

class SpectroscopyData:
    def __init__(self, electrode_data:np.ndarray, swept_freqs:np.ndarray, n_samples=1, aggregate=None, hardware="phobos"):
        '''
        :param electrode_data: raw data output from the PHOBOS acquisition system
        :param swept_freqs: array with all swept frequencies
        :param n_samples: samples per pair swept (1 by default)
        :param aggregate: how to organize the data for each mode (None as default)
        :param hardware: which hardware was used to acquire the signals (different file formats)
        '''

        #check if the raw electrode data is a numpy array
        if type(electrode_data) != np.ndarray:
            raise TypeError(f'[SpectroscopyData] Raw electrode data must be a numpy array! Curr. type = {type(electrode_data)}')

        #check if the swept_frequencies argument is a list
        if type(swept_freqs) != np.ndarray:
            raise TypeError(f'[SpectroscopyData] Swept frequencies data must be a numpy array! Curr. type = {type(swept_freqs)}')

        self.freqs = swept_freqs
        self.n_freqs = len(swept_freqs)

        #validate n_samples
        if not (n_samples > 0):
            raise ValueError(f'[SpectroscopyData] n_samples = {n_samples}, must be > 0!')
        self.n_samples = n_samples

        #validade hardware input
        hardware = hardware.lower() #convert to lowercase
        valid_hardware = ['phobos', 'admx'] #list of valid sweep types
        if hardware not in valid_hardware:
            raise ValueError(f'[SpectroscopyData] hardware = {hardware} not implemented! Try: {valid_hardware}')

        #initialize class attributes
        self.Cp = None #capacitance readings
        self.Rp = None #resistance readings

        if hardware == 'phobos':
            #organize the data based on the CSV format
            valid_electrodes = electrode_data[:,2:] #filter the array from the first electrode reading

            #process capacitance and resistance separately
            idx_cp = np.arange(0,int(2*len(self.freqs)),2) #indexes of each capacitance reading
            self.Cp = valid_electrodes[:, idx_cp] #update capacitance readings
            idx_rp = np.arange(1,int(2*len(self.freqs)),2) #indexes of each resistance reading
            self.Rp = valid_electrodes[:, idx_rp] #update resistance readings

            #apply aggregation if required on the "modes" axis
            if aggregate is not None:
                self.Cp = aggregate(self.Cp, axis=0) #mean over the n_samples
                self.Rp = aggregate(self.Rp, axis=0) #mean over the n_samples

        elif hardware == 'admx':
            self.freqs *= 1000 #kHz to Hz
            self.Cp = electrode_data[:,-2].astype('float') #update capacitance readings
            self.Rp = electrode_data[:,-1].astype('float') #update resistance readings

class TemperatureData:
    def __init__(self, thermo_data:np.ndarray, abs_timestamp:datetime.datetime, ):
        '''
        :param thermo_data: raw temperature data output from the LVM file
        :param abs_timestamp: absolute timestamp of the first acquired sample
        '''

        #check if the thermocouple data input is a numpy array
        if type(thermo_data) != np.ndarray:
            raise TypeError(f'[TemperatureData] Raw thermocouple data must be a numpy array! Curr. type = {type(thermo_data)}')

        #check if the absolute timestamp input is a datetime object
        if type(abs_timestamp) != datetime.datetime:
            raise TypeError(f'[TemperatureData] Absolute timestamp must be a datetime object! Curr. type = {type(abs_timestamp)}')

        n_samples = len(thermo_data) #number of acquired samples per thermocouple
        self.relative_timestamp = np.zeros((n_samples,)) #array that stores the relative timestamps from the raw file
        self.human_timestamp = np.zeros((n_samples,), dtype=datetime.datetime)
        self.measured_temp = np.zeros((n_samples, 4)) #array that stores the measured temperatures for each thermocouple
        for i in range(0, n_samples):
            raw_line = thermo_data[i] #process each sample line at a time
            raw_line = raw_line.replace(',', '.') #use '.' as decimal notation
            raw_line = raw_line.strip('\n') #remove any line breakers
            raw_line = raw_line.split('\t') #split on tabs

            #if a line has less than 5 elements, skip
            if len(raw_line) < 5:
                continue

            self.relative_timestamp[i] = float(raw_line[0]) #update timestamps
            relative_delta = datetime.timedelta(seconds=self.relative_timestamp[i]-self.relative_timestamp[0]) #time delta for the current sample
            self.human_timestamp[i] = abs_timestamp + relative_delta #human timestamp from the absolute
            self.measured_temp[i,:] = [float(temp_sample) for temp_sample in raw_line[1:]] #update temperature readings

        self.human_timestamp = self.human_timestamp.astype('datetime64') #convert from datetime object to numpy datetime
        self.n_sensors = np.shape(self.measured_temp)[1] #number of available thermocouples

class PHOBOSData:
    def __init__(self, filename_electrode:str, filename_temperature=None, n_samples=1, normalize=True, sweeptype="flange", aggregate=None, timezone=-3):
        '''
        :param filename_electrode: path where the .csv is stored
        :param filename_temperature: path where the .lvm is stored
        :param n_samples: samples per pair swept
        :param normalize: apply media-based normalization
        :param sweeptype: how the data is expected to be organized ('flange' for 10-mode sweep or 'spectrum' for full spectroscopy)
        :param aggregate: how to organize the data for each mode (None as default)
        :param timezone: timezone to convert unix timestamp to human timestamp
        '''

        #process electrode data into its custom structure
        electrode_data = file_lcr.read(filename_electrode, n_samples, sweeptype=sweeptype, aggregate=aggregate, timezone=timezone)
        self.Cp = electrode_data.Cp #capacitance
        self.Rp = electrode_data.Rp #resistance
        self.freqs = electrode_data.freqs #swept frequencies
        self.n_freqs = electrode_data.n_freqs #number of swept frequencies
        self.electrode_human_timestamps = electrode_data.human_timestamps #human timestamps
        self.modes = electrode_data.modes #emitter/receiver modes
        self.n_modes = electrode_data.n_modes #number of emitter/receiver modes

        #process temperature data into its custom structure
        try:
            temperature_data = file_lvm.read(filename_temperature)
            self.thermo_readings = temperature_data.measured_temp #temperatures acquired by each thermocouple
            self.temp_human_timestamp = temperature_data.human_timestamp #human timestamps
            self.n_thermosensors = temperature_data.n_sensors #number of thermocouples

            #aligning both sources in time
            er_start = self.electrode_human_timestamps[0]  # timestamp of the first mode sample
            idx_delta = np.argmin(np.abs(self.temp_human_timestamp - er_start))  # index that minimizes the time delta
            self.temp_human_timestamp = self.temp_human_timestamp[idx_delta:]  # slice the temperature timestamps
            self.thermo_readings = self.thermo_readings[idx_delta:, :]  # slice the temperature readings
        except:
            print(f'[PHOBOSData] No temperature file found, processing data without it!')

        #media-based normalization
        if normalize:
            self.Cp_norm = np.zeros_like(self.Cp)
            self.avg_Cp_norm = np.zeros((len(self.Cp),self.n_freqs))
            self.Rp_norm = np.zeros_like(self.Rp)
            self.avg_Rp_norm = np.zeros((len(self.Rp), self.n_freqs))
            smooth_win = 3 #size of the window to compute the moving average filter
            filter_kernel = np.ones(smooth_win)/smooth_win #moving average filter kernel
            ref_win = 10 #size of the window to compute the reference value for each medium
            for f in range(0, self.n_freqs):
                for m in range(0, self.n_modes):
                    Cp_line = self.Cp[:,m,f] #all capacitance readings for a mode at the given freq
                    Cp_line = np.convolve(Cp_line, filter_kernel, "same") #moving average filter
                    Rp_line = self.Rp[:,m,f] #all resistance readings for a mode at the given freq
                    Rp_line = np.convolve(Rp_line, filter_kernel, "same") #moving average filter
                    n_points = len(Cp_line)
                    idx_water = np.arange(1, np.min([ref_win, n_points]),1) #points to use as water reference
                    idx_ice = np.arange(n_points-ref_win, n_points, 1) #point to use as ice reference
                    water_ref_Cp = np.median(Cp_line[idx_water])
                    water_ref_Rp = np.median(Rp_line[idx_water])
                    ice_ref_Cp = np.median(Cp_line[idx_ice])
                    ice_ref_Rp = np.median(Rp_line[idx_ice])
                    self.Cp_norm[:,m,f] = 1 - (Cp_line - water_ref_Cp)/(ice_ref_Cp - water_ref_Cp)
                    self.Rp_norm[:,m,f] = (Rp_line - water_ref_Rp)/(ice_ref_Rp - water_ref_Rp)

                #average the normalized signals over the modes
                self.avg_Cp_norm[:,f] = np.mean(self.Cp_norm[:,:,f], axis=1)
                self.avg_Rp_norm[:, f] = np.mean(self.Rp_norm[:, :, f], axis=1)