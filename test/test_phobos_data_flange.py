import numpy as np
from framework import data_types

#read the files
filename_flange = '../data/testICE_02_12_25/c_test.csv'
filename_temp = '../data/testICE_02_12_25/c_temp.lvm'
phobos_obj = data_types.PHOBOSData(filename_flange, filename_temperature=filename_temp, n_samples=1, sweeptype="flange", acquisition_mode="freq", aggregate=np.mean)