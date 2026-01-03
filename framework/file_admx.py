import pandas as pd
import csv
import numpy as np
import os
from framework import data_types

def read(filename:str, sweeptype="spectrum"):
    '''
    :param filename: path where the .csv is stored
    :param sweeptype: how the data is expected to be organized ('flange' for 10-mode sweep or 'spectrum' for full spectroscopy)
    :param aggregate: how to organize the data for each mode (None as default)
    :param timezone: timezone to convert unix timestamp to human timestamp
    '''

    #check if the filename exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'[file_admx] Filename {filename} does not exist!')

    #validate sweeptype
    sweeptype = sweeptype.lower() #convert to lowercase
    valid_sweeps = ['flange', 'spectrum'] #list of valid sweep types
    if sweeptype not in valid_sweeps:
        raise ValueError(f'[file_admx] sweeptype = {sweeptype} not implemented! Try: {valid_sweeps}')

    #process the raw data output from the ADMX2001 acquisition system into a custom data structure
    raw_data = pd.read_csv(filename).to_numpy() #process the raw data output from the ADMX2001 acquisition system

    if sweeptype == "flange":
        data = None #TODO: generate flange-based files
    elif sweeptype == "spectrum":
        swept_freqs = raw_data[:,0].astype('float') #extract the frequencies array
        data = data_types.SpectroscopyData(raw_data, swept_freqs, hardware='admx')

    return data