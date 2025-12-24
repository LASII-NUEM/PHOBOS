import pandas as pd
import csv
import numpy as np
import os
from framework import data_types

def read(filename:str, n_samples:int, sweeptype="flange", aggregate=None, timezone=-3):
    '''
    :param filename: path where the .csv is stored
    :param n_samples: samples per pair swept
    :param sweeptype: how the data is expected to be organized ('flange' for 10-mode sweep or 'spectrum' for full spectroscopy)
    :param aggregate: how to organize the data for each mode (None as default)
    :param timezone: timezone to convert unix timestamp to human timestamp
    '''

    #check if the filename exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'[file_phobos] Filename {filename} does not exist!')

    #validate n_samples
    if not (n_samples>0):
        raise ValueError(f'[file_phobos] n_samples = {n_samples}, must be > 0!')

    #validate sweeptype
    sweeptype = sweeptype.lower() #convert to lowercase
    valid_sweeps = ['flange', 'spectrum'] #list of valid sweep types
    if sweeptype not in valid_sweeps:
        raise ValueError(f'[file_phobos] sweeptype = {sweeptype} not implemented! Try: {valid_sweeps}')

    #infer the swept frequencies from the file header
    with open(filename, 'r') as f:
        reader = csv.DictReader(f) #read only the header
        header_data = reader.fieldnames
    f.close() #close the file

    #process the raw data output from the PHOBOS acquisition system into a custom data structure
    raw_data = pd.read_csv(filename).to_numpy() #process the raw data output from the PHOBOS acquisition system

    if sweeptype == "flange":
        swept_freqs = [float(freq.replace(" ", "").replace("Cp", "")) for freq in header_data if 'Cp' in freq]
        swept_freqs = np.array(swept_freqs)  # convert to numpy array
        data = data_types.FlangeData(raw_data, swept_freqs, n_samples, aggregate=aggregate, timezone=timezone)
    elif sweeptype == "spectrum":
        swept_freqs = [float(freq.replace(" ", "").replace("Z", "")) for freq in header_data if 'Z' in freq]
        swept_freqs = np.array(swept_freqs)  # convert to numpy array
        data = data_types.SpectroscopyData(raw_data, swept_freqs, n_samples, aggregate=aggregate)

    return data
