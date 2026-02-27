import numpy as np
from framework import batch_utils
import os
import time

circuits = {"Longo2020": {"guess": np.array([1, 1, 1, 1, 1, 1, 0.5, 1]),
                          "scale_BFGS": np.array([1e5, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]),
                          "scale_NLLS": np.array([1e5, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]),
                          "scale_DLS": np.array([1e4, 1e-7, 1e6, 1e-2, 1e4, 1e-1, 1, 1]),
                          "scale_SIMPLEX": np.array([1e5, 1e-6, 1e7, 1e-2, 1e4, 1e-1, 1, 1])},
            "Zurich2021": {"guess": np.array([1, 1, 0.5, 1, 1, 1]),
                          "scale_BFGS": np.array([1e4, 1e-8, 1, 1e5, 1e3, 1e-8]),
                          "scale_NLLS": np.array([1e5, 1e-8, 1, 1e5, 1e3, 1e-8]),
                          "scale_DLS": np.array([1e5, 1e-8, 1, 1e4, 1e3, 1e-8]),
                          "scale_SIMPLEX": np.array([1e5, 1e-8, 1, 1e5, 1e3, 1e-7])},
            "Zhang2024": {"guess": np.array([1, 0.5, 1, 1, 1, 1]),
                          "scale_BFGS": np.array([1e-9, 1, 1e6, 1e3, 1e2, 1e-8]),
                          "scale_NLLS": np.array([1e-9, 1, 1e6, 1e3, 1e2, 1e-8]),
                          "scale_DLS":  np.array([1e-9, 1, 1e6, 1e3, 1e4, 1e-8]),
                          "scale_SIMPLEX": np.array([1e-7, 1, 1e5, 1e5, 1e4, 1e-8])},
            "Yang2025": {"guess": np.array([1, 1, 1, 0.5, 1, 1]),
                          "scale_BFGS": np.array([1e4, 1e5, 1e-9, 1, 1e4, 1e-8]),
                          "scale_NLLS": np.array([1e4, 1e5, 1e-9, 1, 1e4, 1e-8]),
                          "scale_DLS":  np.array([1e4, 1e4, 1e-9, 1, 1e4, 1e-8]),
                          "scale_SIMPLEX": np.array([1e3, 1e4, 1e-9, 1, 1e4, 1e-8])}} #list of circuits to attempt fitting the data

batch_fit_obj = batch_utils.BatchImpedanceFit('../data/testICE_30_01_26/c_test.csv', circuits, freq_threshold=100, aggregate=np.mean, save=True)