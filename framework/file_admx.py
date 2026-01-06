import pandas as pd
import csv
import numpy as np
import os
from framework import data_types

def read(filename:str, sweeptype="spectrum", acquisition_mode="freq"):
    '''
    :param filename: path where the .csv is stored
    :param sweeptype: which hardware was used to acquire the signals
    :param acquisition_mode: how the data is expected to be organized ('flange' for 10-mode sweep or 'spectrum' for full spectroscopy)
    '''

    #check if the filename exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'[file_admx] Filename {filename} does not exist!')

    #validate sweeptype
    sweeptype = sweeptype.lower() #convert to lowercase
    valid_sweeps = ['flange', 'cell', 'custom'] #list of valid sweep types
    if sweeptype not in valid_sweeps:
        raise ValueError(f'[file_admx] sweeptype = {sweeptype} not implemented! Try: {valid_sweeps}')

    #validate acquisition_mode
    acquisition_mode = acquisition_mode.lower() #convert to lowercase
    valid_acqs = ['freq', 'spectrum'] #list of valid sweep types
    if acquisition_mode not in valid_acqs:
        raise ValueError(f'[file_admx] acquisition_mode = {acquisition_mode} not implemented! Try: {valid_acqs}')

    #process the raw data output from the ADMX2001 acquisition system into a custom data structure
    raw_data = pd.read_csv(filename).to_numpy() #process the raw data output from the ADMX2001 acquisition system

    if sweeptype == "flange":
        if acquisition_mode == "spectrum":
            data = None  # TODO: generate flange-based spectrum files
        elif acquisition_mode == "freq":
            data = None  # TODO: generate flange-based frequency files
    elif sweeptype == "cell":
        if acquisition_mode == "spectrum":
            swept_freqs = raw_data[:,0].astype('float') #extract the frequencies array
            data = data_types.SpectroscopyData(raw_data, swept_freqs, sweeptype=sweeptype, hardware='admx')
        elif acquisition_mode == "freq":
            data = None #TODO: generate cell-based frequency files

    return data