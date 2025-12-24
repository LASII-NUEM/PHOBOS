from framework import file_csv
import numpy as np

#PHOBOS spectroscopy acquisition
spec_air_obj = file_csv.read('../data/testICE_10_12_25/c0.csv', 3, "spectrum", np.mean)
spec_h2o_obj = file_csv.read('../data/testICE_10_12_25/c1.csv', 3, "spectrum", np.mean)
spec_ice_obj = file_csv.read('../data/testICE_10_12_25/cice.csv', 3, "spectrum", np.mean)