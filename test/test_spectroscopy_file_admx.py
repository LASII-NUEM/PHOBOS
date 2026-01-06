from framework import file_admx
import numpy as np

#ADMX2001 spectroscopy acquisition
spec_air_obj = file_admx.read('../data/testICE_16_12_25/c0.csv', sweeptype="cell", acquisition_mode="spectrum")
h2o_air_obj = file_admx.read('../data/testICE_16_12_25/c1.csv', sweeptype="cell", acquisition_mode="spectrum")