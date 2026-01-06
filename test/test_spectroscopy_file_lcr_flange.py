from framework import file_lcr
import numpy as np

#PHOBOS spectroscopy acquisition
spec_air_obj = file_lcr.read('../data/testICE_03_12_25/c0.csv', n_samples=2, sweeptype="flange", acquisition_mode="spectrum", aggregate=np.mean)
spec_h2o_obj = file_lcr.read('../data/testICE_03_12_25/c1.csv', n_samples=2, sweeptype="flange", acquisition_mode="spectrum", aggregate=np.mean)
spec_ice_obj = file_lcr.read('../data/testICE_03_12_25/cice.csv', n_samples=2, sweeptype="flange", acquisition_mode="spectrum", aggregate=np.mean)