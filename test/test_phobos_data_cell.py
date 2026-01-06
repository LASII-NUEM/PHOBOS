import numpy as np
from framework import data_types

#read the files
filename_flange = '../data/testICE_12_12_25/c_test.csv'
phobos_obj = data_types.PHOBOSData(filename_flange, n_samples=1, sweeptype="cell", acquisition_mode="freq", aggregate=np.mean)