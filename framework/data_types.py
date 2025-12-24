import numpy as np
import pandas as pd
import datetime

class FlangeData:
    def __init__(self, electrode_data:np.ndarray, swept_freqs:np.ndarray, n_samples:int, aggregate=None, timezone=-3):
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
        self.Cp = np.reshape(self.Cp, [self.n_samples, int(total_loops), self.n_modes, len(self.freqs)]) #[samples, modes, readings, freqs]
        idx_rp = np.arange(1,int(2*len(self.freqs)),2) #indexes of each resistance reading
        self.Rp = valid_electrodes[:, idx_rp] #update resistance readings
        self.Rp = np.reshape(self.Rp, [self.n_samples, int(total_loops), self.n_modes, len(self.freqs)]) #[samples, modes, readings, freqs]

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

class SpectroscopyData:
    def __init__(self, electrode_data:np.ndarray, swept_freqs:np.ndarray, n_samples:int, aggregate=None):
        '''
        :param electrode_data: raw data output from the PHOBOS acquisition system
        :param swept_freqs: array with all swept frequencies
        :param n_samples: samples per pair swept
        :param aggregate: how to organize the data for each mode (None as default)
        '''

        #check if the raw electrode data is a numpy array
        if type(electrode_data) != np.ndarray:
            raise TypeError(f'[SpectroscopyData] Raw electrode data must be a numpy array! Curr. type = {type(electrode_data)}')

        #check if the swept_frequencies argument is a list
        if type(swept_freqs) != np.ndarray:
            raise TypeError(f'[SpectroscopyData] Swept frequencies data must be a numpy array! Curr. type = {type(swept_freqs)}')

        self.freqs = swept_freqs

        #validate n_samples
        if not (n_samples > 0):
            raise ValueError(f'[SpectroscopyData] n_samples = {n_samples}, must be > 0!')
        self.n_samples = n_samples

        #initialize class attributes
        self.Cp = None #capacitance readings
        self.Rp = None #resistance readings

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