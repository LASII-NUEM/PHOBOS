from framework import file_lcr
import numpy as np

#PHOBOS relay acquisition
filename = '../data/testICE_12_12_25/c_test.csv' #path to the raw data CSV file
relay_obj = file_lcr.read(filename, n_samples=1, sweeptype="cell", aggregate=np.mean)