from framework import file_csv
import numpy as np

#PHOBOS relay acquisition
filename = '../data/testICE_02_12_25/c_test.csv' #path to the raw data CSV file
relay_obj = file_csv.read(filename, 1, sweeptype="flange", aggregate=np.mean)

